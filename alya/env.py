"""RL-обёртка над симулятором: шаг 16 мс, действие = смещение курсора + клавиша."""
from __future__ import annotations

import numpy as np

from .keys import KeyAssigner
from .sim import JMISS, MOVE_SCALE, MapPool, Simulator, clamp_cursor

DT = 16.0
# шаг, на который приходится нота, ставится ровно на неё (как в живой игре, live._next_step):
# если нота через 0.5–1.5 DT — шаг заканчивается в момент ноты
SNAP_LO, SNAP_HI = 0.5 * DT, 1.5 * DT


def next_step(t: float, note_t: np.ndarray) -> float:
    """Время следующего шага для одной карты (note_t — времена голов нот по возрастанию)."""
    i = int(np.searchsorted(note_t, t + 1e-3))
    if i < len(note_t) and SNAP_LO <= note_t[i] - t <= SNAP_HI:
        return float(note_t[i])
    return t + DT
# плавность аима: штраф за резкий поворот на скорости. corner = min(скорость до, после) / CORNER_NORM
# * (1 - cos угла) / 2; штраф ~ corner². Разворот на 180° на полном ходу дорогой, а тот же
# поворот, растянутый в дугу на несколько шагов или с торможением у ноты, — почти бесплатный.
CORNER_NORM = 20.0
# «трудные» места для старта эпизода: стримы (медиана интервала до следующих нот < STREAM_GAP мс),
# двойки 1/8 (интервал <= BURST_GAP мс) — реальные мс, с учётом DT/HT — и слайдеры с повторами
STREAM_GAP = 115.0
BURST_GAP = 65.0
REWARD = np.array([1.0, 0.4, 0.15, -1.0], np.float32)  # 300 / 100 / 50 / miss


class OsuEnv:
    def __init__(self, pool: MapPool, n: int, max_steps: int = 2000, shaping: float = 0.3,
                 click_cost: float = 0.01, wasted_cost: float = 0.15, smooth_cost: float = 0.0, seed: int = 0,
                 hard_frac: float = 0.0, cp_cost: float = 0.1, timing_cost: float = 0.0):
        self.pool, self.n = pool, n
        self.sim = Simulator(pool, n)
        self.max_steps, self.shaping, self.click_cost = max_steps, shaping, click_cost
        self.wasted_cost, self.smooth_cost = wasted_cost, smooth_cost
        self.hard_frac, self.cp_cost, self.timing_cost = hard_frac, cp_cost, timing_cost
        self.hard = self._hard_starts()
        self.rng = np.random.default_rng(seed)
        self.steps = np.zeros(n, np.int64)
        self.phi = np.zeros(n, np.float32)
        self.keys = KeyAssigner(n)
        self.vel = np.zeros((n, 2), np.float32)
        self.ep_stats = []  # (accuracy, n_judged) законченных эпизодов

    def _potential(self):
        """-расстояние до цели, если цель близко по времени."""
        s = self.sim
        tgt = s.target()
        d = np.linalg.norm(tgt - s.pos, axis=1)
        obj = np.where(s.body >= 0, s.body, s.head)
        dt = s._g(self.pool.t, obj) - s.t
        near = (s.body >= 0) | (dt < 600)
        return np.where(near & ~s.finished(), -d / 100.0, 0).astype(np.float32)

    def _hard_starts(self):
        """По каждой карте — объекты, с которых начинается плотный участок."""
        P, out = self.pool, []
        for m in range(len(P.maps)):
            n = int(P.n_obj[m])
            t = P.t[m, :n]
            g = np.diff(t)
            hard = np.zeros(n, bool)
            for i in range(max(n - 9, 0)):
                w = g[i:i + 8]
                hard[i] = np.median(w) < STREAM_GAP or w[:4].min() <= BURST_GAP
            hard |= P.slides[m, :n] >= 2
            # старт за пару нот до участка, чтобы успеть на него зайти
            out.append(np.maximum(np.nonzero(hard)[0] - 2, 0))
        return out

    def reset(self, idx=None):
        idx = np.arange(self.n) if idx is None else np.asarray(idx)
        if len(idx) == 0:
            return
        P = self.pool
        mid = self.rng.integers(0, len(P.maps), len(idx))
        start = (self.rng.random(len(idx)) * np.maximum(P.n_obj[mid] - 20, 1)).astype(np.int64)
        start[self.rng.random(len(idx)) < 0.2] = 0
        for k in np.nonzero(self.rng.random(len(idx)) < self.hard_frac)[0]:
            h = self.hard[mid[k]]
            if len(h):
                start[k] = h[self.rng.integers(len(h))]
        t0 = P.t[mid, start] - self.rng.uniform(300, 1500, len(idx))
        pos = np.stack([self.rng.uniform(0, 512, len(idx)), self.rng.uniform(0, 384, len(idx))], 1)
        self.sim.reset(idx, mid, start, t0, pos.astype(np.float32))
        self.steps[idx] = 0
        self.keys.reset(idx)
        self.vel[idx] = 0
        self.phi[idx] = self._potential()[idx]

    def obs(self):
        return self.sim.obs()

    def step(self, move: np.ndarray, key: np.ndarray):
        s = self.sim
        move = np.clip(move, -5, 5) * MOVE_SCALE
        pos = clamp_cursor(s.pos + move)
        P = self.pool
        th = s._g(P.t, s.head)
        dt_note = th - s.t
        snap = (s.head < P.n_obj[s.map_id]) & (dt_note >= SNAP_LO) & (dt_note <= SNAP_HI)
        t_new = np.where(snap, th, s.t + DT)
        key = self.keys(t_new, key > 0.5)
        vel = pos - s.pos
        sp0, sp1 = np.linalg.norm(self.vel, axis=1), np.linalg.norm(vel, axis=1)
        cos = (vel * self.vel).sum(1) / (sp0 * sp1 + 1e-6)
        corner = np.minimum(sp0, sp1) / CORNER_NORM * (1 - cos) / 2
        self.vel = vel
        ev = s.advance(t_new, pos, key)
        r = ev["judge"].astype(np.float32) @ REWARD
        r += 0.1 * ev["cp_hit"] - self.cp_cost * ev["cp_miss"]
        r += 0.15 * ev["spin"]
        r -= self.click_cost * ev["click"] + self.wasted_cost * ev["wasted"]
        r -= self.smooth_cost * corner ** 2
        # точность по времени: даже внутри окна 300 лучше жать ближе к моменту ноты
        r -= self.timing_cost * ev["timing"]
        phi = self._potential()
        r += self.shaping * (phi - self.phi)
        self.phi = phi
        self.steps += 1
        done = s.finished() | (self.steps >= self.max_steps)
        info = {"judge": ev["judge"], "click": ev["click"], "wasted": ev["wasted"], "corner": corner, "timing": ev["timing"]}
        if done.any():
            d = np.nonzero(done)[0]
            for i in d:
                tot = s.counts[i].sum()
                if tot:
                    self.ep_stats.append((s.accuracy()[i], s.counts[i, JMISS] / tot))
            self.reset(d)
        return self.obs(), r, done, info
