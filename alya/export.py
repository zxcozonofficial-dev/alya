"""Прогон модели по карте в симуляторе и сохранение .osr для просмотра в osu!."""
from __future__ import annotations

import numpy as np
import torch

from . import model as M
from .autoplay import autoplay
from .beatmap import Beatmap
from .bc import grid
from .doodle import BreakDoodler
from .humanize import CurveBend
from .keys import KeyAssigner
from .osr import K1, K2, M1, M2, SMOKE, Replay, write_osr
from .beatmap import SPINNER
from .env import next_step
from .sim import MOVE_SCALE, MapPool, Simulator, clamp_cursor


def play_map(bm: Beatmap, net: M.AlyaNet | None, deterministic=True, device="cpu", acc=0.97, miss=0.005,
             doodle=False, curve=0.0):
    """-> (times, pos, key, simulator, smoke). net=None — автоплей. doodle — рисовать дымом в перерывах,
    curve — сила изгиба траектории между нотами (в возвращаемых позициях; симулятор и модель — без него)."""
    times = grid(bm)
    sim = Simulator(MapPool([bm]), 1)
    if net is None:
        pos, key = autoplay(bm, times)
        sim.reset([0], 0, 0, times[0], pos[0])
        for k in range(1, len(times)):
            sim.advance(times[k], pos[k][None], key[k:k + 1])
        return times, pos, key, sim, np.zeros(len(times), bool)
    # шаги как в обучении и живой игре: по 16 мс, но шаг с нотой — ровно на неё
    note_t = np.array([o.time for o in bm.objects if o.kind != SPINNER])
    t_list = [float(times[0])]
    while t_list[-1] < times[-1]:
        t_list.append(next_step(t_list[-1], note_t))
    times = np.array(t_list)
    pos = np.zeros((len(times), 2), np.float32)
    key = np.zeros((len(times), 2), bool)
    smoke = np.zeros(len(times), bool)
    pos[0] = (256, 192)
    keys = KeyAssigner(1)
    doodler = BreakDoodler(bm, seed=0) if doodle else None
    bend = CurveBend(bm, curve, seed=0)
    shown = pos.copy()
    sim.reset([0], 0, 0, times[0], pos[0])
    h = net.init_hidden(1, device)
    cond = M.skill_features(acc, miss)
    for k in range(1, len(times)):
        obs = torch.as_tensor(M.with_cond(sim.obs(), cond), device=device)
        move, kp, _, _, h = net.act(obs, h, deterministic)
        pos[k] = clamp_cursor(pos[k - 1] + np.clip(move.cpu().numpy()[0], -5, 5) * MOVE_SCALE)
        want = kp.cpu().numpy() > 0.5
        if doodler is not None:
            p, smoke[k] = doodler.step(times[k], pos[k])
            if p is not None:
                pos[k], want[:] = p, False
        shown[k] = pos[k] if smoke[k] else bend(times[k], pos[k])
        key[k] = keys(times[k], want)[0]
        sim.advance(times[k], pos[k][None], key[k:k + 1])
    return times, shown, key, sim, smoke


def to_replay(bm: Beatmap, times, pos, key, sim: Simulator, player="Alya", smoke=None) -> Replay:
    k = key[:, 0] * (K1 | M1) + key[:, 1] * (K2 | M2)
    if smoke is not None:
        k = k + smoke * SMOKE
    c = sim.counts[0]
    score = int(300 * c[0] + 100 * c[1] + 50 * c[2])
    # в .osr время кадров — время карты (для DT/HT отличается от реального)
    frames = np.column_stack([times * bm.rate, pos[:, 0], pos[:, 1], k])
    return Replay(beatmap_md5=bm.md5, player=player, n300=int(c[0]), n100=int(c[1]), n50=int(c[2]),
                  nmiss=int(c[3]), score=score, max_combo=int(sim.max_combo[0]),
                  perfect=bool(c[3] == 0), mods=bm.mods, frames=frames)


def export(bm: Beatmap, net, out: str, deterministic=True, device="cpu", player="Alya", acc=0.97, miss=0.005):
    times, pos, key, sim, smoke = play_map(bm, net, deterministic, device, acc, miss, doodle=True, curve=0.2)
    r = to_replay(bm, times, pos, key, sim, player, smoke)
    write_osr(r, out)
    return r, sim.accuracy()[0]
