"""Рекуррентный PPO в векторизованном симуляторе."""
from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch

from . import model as M
from .env import OsuEnv
from .sim import MapPool


def train_ppo(maps, out: str, init: str | None = None, n_envs: int = 256, horizon: int = 128,
              updates: int = 2000, lr: float = 1e-4, gamma: float = 0.99, lam: float = 0.95,
              clip: float = 0.2, epochs: int = 4, minibatches: int = 4, ent_coef: float = 0.002,
              vf_coef: float = 0.5, save_every: int = 50, device: str | None = None,
              kl_coef: float = 0.05, acc_range=(0.90, 1.0), miss_range=(0.0, 0.04),
              ref_path: str | None = None, wasted_cost: float = 0.15, smooth_cost: float = 0.0,
              hard_frac: float = 0.0, cp_cost: float = 0.1, timing_cost: float = 0.0):
    """kl_coef — якорь к BC-модели (init): держит человеческий стиль, 0 = чистый PPO.
    Каждой среде на эпизод выдаётся случайный уровень (acc, miss) из диапазонов."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    pool = MapPool(maps)
    env = OsuEnv(pool, n_envs, wasted_cost=wasted_cost, smooth_cost=smooth_cost, hard_frac=hard_frac,
                 cp_cost=cp_cost, timing_cost=timing_cost)
    net = M.load(init, device) if init else M.AlyaNet().to(device)
    net.train()
    ref = None
    if (ref_path or init) and kl_coef > 0:  # якорь стиля: ref (обычно BC) или init
        ref = M.load(ref_path, device).eval() if ref_path else copy.deepcopy(net).eval()
        for p in ref.parameters():
            p.requires_grad_(False)
    opt = torch.optim.Adam(net.parameters(), lr=lr, eps=1e-5)
    rng = np.random.default_rng(0)

    def sample_cond(k):
        return M.skill_features(rng.uniform(*acc_range, k), rng.uniform(*miss_range, k))

    Path(out).parent.mkdir(parents=True, exist_ok=True)

    env.reset()
    cond = sample_cond(n_envs)
    obs = torch.as_tensor(M.with_cond(env.obs(), cond), device=device)
    h = net.init_hidden(n_envs, device)
    reset_next = torch.zeros(n_envs, device=device)
    T, N = horizon, n_envs
    for upd in range(1, updates + 1):
        t_start = time.time()
        h0 = h.clone()
        buf_obs = torch.zeros(T, N, obs.shape[1], device=device)
        buf_move = torch.zeros(T, N, 2, device=device)
        buf_key = torch.zeros(T, N, 2, device=device)
        buf_logp = torch.zeros(T, N, device=device)
        buf_val = torch.zeros(T, N, device=device)
        buf_rew = torch.zeros(T, N, device=device)
        buf_done = torch.zeros(T, N, device=device)
        buf_reset = torch.zeros(T, N, device=device)
        judge = np.zeros(4, np.int64)
        clicks = wasted = 0
        corners = []
        timing = 0.0
        for t in range(T):
            h = h * (1 - reset_next).view(1, -1, 1)
            move, key, logp, v, h = net.act(obs, h)
            o2, r, done, info = env.step(move.cpu().numpy(), key.cpu().numpy())
            judge += info["judge"].sum(0)
            clicks += info["click"].sum()
            wasted += info["wasted"].sum()
            corners.append(info["corner"])
            timing += info["timing"].sum()
            buf_obs[t], buf_move[t], buf_key[t] = obs, move, key
            buf_logp[t], buf_val[t], buf_reset[t] = logp, v, reset_next
            buf_rew[t] = torch.as_tensor(r, device=device)
            buf_done[t] = torch.as_tensor(done, device=device, dtype=torch.float32)
            reset_next = buf_done[t]
            if done.any():
                cond[done] = sample_cond(int(done.sum()))
            obs = torch.as_tensor(M.with_cond(o2, cond), device=device)
        with torch.no_grad():
            _, _, last_v, _ = net(obs.unsqueeze(0), h * (1 - reset_next).view(1, -1, 1))
            last_v = last_v[0]
        adv = torch.zeros(T, N, device=device)
        gae = torch.zeros(N, device=device)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_val[t + 1]
            nd = 1 - buf_done[t]
            delta = buf_rew[t] + gamma * nv * nd - buf_val[t]
            gae = delta + gamma * lam * nd * gae
            adv[t] = gae
        ret = adv + buf_val

        # обновление: минибатчи по средам, целые последовательности длины T
        env_ids = np.arange(N)
        stats = []
        for _ in range(epochs):
            np.random.shuffle(env_ids)
            for mb in np.array_split(env_ids, minibatches):
                mb = torch.as_tensor(mb, device=device)
                mu, kl, v, _ = net(buf_obs[:, mb], h0[:, mb], buf_reset[:, mb])
                nd_, kd_ = net.dists(mu, kl)
                logp = nd_.log_prob(buf_move[:, mb]).sum(-1) + kd_.log_prob(M.key_index(buf_key[:, mb]))
                ent = nd_.entropy().sum(-1) + kd_.entropy()
                a = adv[:, mb]
                a = (a - a.mean()) / (a.std() + 1e-8)
                ratio = (logp - buf_logp[:, mb]).exp()
                pg = -torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a).mean()
                vl = (v - ret[:, mb]).pow(2).mean()
                loss = pg + vf_coef * vl - ent_coef * ent.mean()
                kl_ref = torch.zeros((), device=device)
                if ref is not None:
                    with torch.no_grad():
                        rmu, rkl, _, _ = ref(buf_obs[:, mb], h0[:, mb], buf_reset[:, mb])
                    rnd, rkd = ref.dists(rmu, rkl)
                    kl_ref = (torch.distributions.kl_divergence(nd_, rnd).sum(-1)
                              + torch.distributions.kl_divergence(kd_, rkd)).mean()
                    loss = loss + kl_coef * kl_ref
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
                stats.append((pg.item(), vl.item(), ent.mean().item(), kl_ref.item()))
        h = h.detach()

        fps = T * N / (time.time() - t_start)
        tot = max(judge.sum(), 1)
        recent = env.ep_stats[-200:]
        ep_acc = np.mean([a for a, _ in recent]) if recent else float("nan")
        pg, vl, en, klr = np.mean(stats, 0)
        print(f"upd {upd:5d} | rew/step {buf_rew.mean().item():+.4f} | 300 {judge[0]/tot:.2f} "
              f"100 {judge[1]/tot:.2f} 50 {judge[2]/tot:.2f} miss {judge[3]/tot:.2f} лишн.клик {wasted/max(clicks,1):.2f} "
              f"углы {np.mean(np.concatenate(corners) > 0.5) * 1000:.1f}‰ "
              f"тайминг {timing / max(judge[:3].sum(), 1):.2f}w300 | "
              f"ep_acc {ep_acc:.3f} | pg {pg:+.3f} vl {vl:.3f} ent {en:.2f} kl_bc {klr:.3f} | {fps:.0f} st/s", flush=True)
        if upd % save_every == 0 or upd == updates:
            M.save(net, out, kind="ppo", update=upd)
    return net
