"""Behavior cloning на человеческих реплеях .osr."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import model as M
from .autoplay import autoplay
from .beatmap import Beatmap, mods_to_string as mods_str, parse_osu
from .env import DT
from .osr import MODS_UNSUPPORTED, read_osr, resample
from .sim import MOVE_SCALE, MapPool, Simulator


def trajectory_dataset(bm: Beatmap, times, pos, key, click_t=None, click_pos=None):
    """Прогнать траекторию через симулятор -> (obs [T,D], move [T,2], key [T]), счёт."""
    sim = Simulator(MapPool([bm]), 1)
    sim.reset([0], 0, 0, times[0], pos[0])
    T = len(times) - 1
    obs = np.zeros((T, sim.obs().shape[1]), np.float32)
    for k in range(T):
        obs[k] = sim.obs()[0]
        sim.advance(times[k + 1], pos[k + 1][None], key[k + 1:k + 2],
                    None if click_t is None else click_t[k + 1:k + 2],
                    None if click_pos is None else click_pos[k + 1][None])
    move = np.clip((pos[1:] - pos[:-1]) / MOVE_SCALE, -M.MAX_ACTION, M.MAX_ACTION)
    return obs, move.astype(np.float32), key[1:].astype(np.float32), sim.counts[0].copy()


def grid(bm: Beatmap):
    return np.arange(max(bm.objects[0].time - 1000, 0.0), bm.objects[-1].end_time + 500, DT)


def _replay_job(rp: str, osu_path: str):
    """Один реплей -> (obs, move, key, skill, строка лога) или (None, причина пропуска)."""
    r = read_osr(rp)
    bm = parse_osu(osu_path, r.mods)
    if len(bm.objects) < 2:
        return None, "пустая карта"
    times = grid(bm)
    frames = r.frames.copy()
    frames[:, 0] /= bm.rate  # кадры во времени карты -> реальные мс
    times = times[times <= frames[-1, 0]]  # фейл/выход: дальше кадров нет
    if len(times) < 600:
        return None, "короче 10 с"
    pos, key, ct, cp = resample(frames, times)
    o, mv, k, counts = trajectory_dataset(bm, times, pos, key, ct, cp)
    real = (r.n300, r.n100, r.n50, r.nmiss)
    tot = max(sum(real), 1)
    skill = ((300 * r.n300 + 100 * r.n100 + 50 * r.n50) / (300 * tot), r.nmiss / tot)
    line = f"  {mods_str(r.mods):6s} {Path(rp).name[:44]:44s} sim {counts.tolist()} | игра {list(real)}"
    return (o.astype(np.float16), mv, k, skill), line


def _autoplay_job(bm: Beatmap):
    times = grid(bm)
    pos, key = autoplay(bm, times)
    o, mv, k, _ = trajectory_dataset(bm, times, pos, key)
    return (o.astype(np.float16), mv, k, (1.0, 0.0)), None


def build_dataset(replays: list[str], md5_to_osu: dict[str, str], out: str,
                  autoplay_maps: list[Beatmap] = (), workers: int | None = None):
    import os
    from concurrent.futures import ProcessPoolExecutor
    seen, skipped, jobs = set(), {}, []
    for rp in replays:  # заголовки читаем здесь: фильтры и дубликаты
        try:
            r = read_osr(rp)
        except Exception as e:
            print(f"  пропуск {Path(rp).name}: {e}")
            continue
        why = ("не std" if r.mode != 0 else "моды RX/Auto/AP/SO" if r.mods & MODS_UNSUPPORTED
               else "карты нет в Songs" if r.beatmap_md5 not in md5_to_osu
               else "дубликат" if r.replay_md5 in seen else "пустой" if len(r.frames) < 10 else None)
        if why:
            skipped[why] = skipped.get(why, 0) + 1
            continue
        seen.add(r.replay_md5)
        jobs.append((rp, md5_to_osu[r.beatmap_md5]))
    # тяжёлая часть (прогон через симулятор) — параллельно, одно ядро оставляем живой игре
    workers = workers or max(1, (os.cpu_count() or 2) // 2 - 1)
    print(f"реплеев к обработке: {len(jobs)}, процессов: {workers}", flush=True)
    seqs, skills = [], []
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(_replay_job, rp, osu) for rp, osu in jobs]
        futs += [ex.submit(_autoplay_job, bm) for bm in autoplay_maps]
        for i, fut in enumerate(futs):
            try:
                res, info = fut.result()
            except Exception as e:
                skipped["ошибка"] = skipped.get("ошибка", 0) + 1
                print(f"  ошибка: {e}")
                continue
            if res is None:
                skipped[info] = skipped.get(info, 0) + 1
                continue
            seqs.append(res[:3])
            skills.append(res[3])
            if info:
                print(info, flush=True)
            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(futs)}]", flush=True)
    if skipped:
        print("пропущено:", ", ".join(f"{k}: {v}" for k, v in skipped.items()))
    if not seqs:
        raise SystemExit("нет подходящих реплеев (std, без HR/DT/EZ/HT, карта найдена в Songs)")
    lens = np.array([len(s[0]) for s in seqs])
    np.savez_compressed(out, obs=np.concatenate([s[0] for s in seqs]).astype(np.float16),
                        move=np.concatenate([s[1] for s in seqs]), key=np.concatenate([s[2] for s in seqs]),
                        lens=lens, skill=np.array(skills, np.float32))
    print(f"датасет: {len(seqs)} траекторий, {lens.sum()} шагов -> {out}")


def train_bc(dataset: str, out: str, init: str | None = None, steps: int = 20000, batch: int = 64,
             seq: int = 256, burn_in: int = 32, lr: float = 3e-4, device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = np.load(dataset)
    obs = torch.as_tensor(d["obs"], device=device)  # float16, в float32 по батчам
    move = torch.as_tensor(d["move"], device=device)
    key = torch.as_tensor(d["key"], device=device)
    lens = d["lens"]
    # условие уровня на каждый шаг (для синтетики/старых датасетов — идеальная игра)
    skill = d["skill"] if "skill" in d else np.tile([1.0, 0.0], (len(lens), 1))
    cond = torch.as_tensor(np.repeat(M.skill_features(skill[:, 0], skill[:, 1]), lens, 0),
                           device=device, dtype=torch.float16)
    starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
    L = seq + burn_in
    ok = lens > L
    if not ok.any():
        raise SystemExit("траектории короче окна seq+burn_in")
    starts, lens = starts[ok], lens[ok]
    w = lens / lens.sum()
    net = M.load(init, device) if init else M.AlyaNet().to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    ar = torch.arange(L, device=device)
    for it in range(1, steps + 1):
        s = np.random.choice(len(lens), batch, p=w)
        off = starts[s] + (np.random.rand(batch) * (lens[s] - L)).astype(np.int64)
        idx = torch.as_tensor(off, device=device)[None] + ar[:, None]          # [L,B]
        mu, kl, _, _ = net(torch.cat([obs[idx], cond[idx]], -1).float(), net.init_hidden(batch, device))
        mu, kl = mu[burn_in:], kl[burn_in:]
        tm, tk = move[idx][burn_in:], key[idx][burn_in:]
        nd = torch.distributions.Normal(mu, net.log_std.clamp(-4, 1).exp())
        loss_move = -nd.log_prob(tm).sum(-1).mean()
        tki = M.key_index(tk)
        loss_key = F.cross_entropy(kl.reshape(-1, 4), tki.reshape(-1))
        loss = loss_move + loss_key
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if it % 200 == 0 or it == steps:
            with torch.no_grad():
                mae = (mu - tm).abs().mean().item() * MOVE_SCALE
                kacc = (kl.argmax(-1) == tki).float().mean().item()
            print(f"bc {it:6d} | loss {loss.item():.4f} | move MAE {mae:.2f}px | key acc {kacc:.3f}", flush=True)
        if it % 2000 == 0 or it == steps:
            M.save(net, out, kind="bc", step=it)
    return net
