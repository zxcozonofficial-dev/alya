"""Рисование дымом (клавиша smoke, по умолчанию C) в перерывах карты.

В длинной паузе между нотами курсор забирается у модели: подлетает к началу рисунка,
обводит его с зажатым smoke и отдаёт управление за preempt+250 мс до следующей ноты,
чтобы модель успела долететь до неё.
"""
from __future__ import annotations

import numpy as np

from .beatmap import Beatmap

PEN_SPEED = 0.45     # osu!px/мс — скорость обводки
TRAVEL_MS = 220.0    # перелёт между штрихами без дыма
MIN_BREAK = 1500.0   # короче — не рисуем
STEP_MS = 4.0
SIZES = {1: 90.0, 2: 75.0, 3: 60.0}  # размер рисунка (osu!px) от их числа в паузе


def _circle(cx, cy, r, n=40, a0=0.0, a1=2 * np.pi):
    a = np.linspace(a0, a1, n)
    return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)


def _heart():
    t = np.linspace(0, 2 * np.pi, 80)
    x = 16 * np.sin(t) ** 3
    y = -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))
    return [np.stack([x, y], 1) / 17]


def _star():
    a = -np.pi / 2 + np.arange(6) * 4 * np.pi / 5
    return [np.stack([np.cos(a), np.sin(a)], 1)]


def _smiley():
    return [_circle(0, 0, 1, 60), _circle(-0.35, -0.3, 0.08, 10), _circle(0.35, -0.3, 0.08, 10),
            _circle(0, 0.05, 0.55, 24, 0.2 * np.pi, 0.8 * np.pi)]


def _cat():
    head = _circle(0, 0.1, 0.8, 50)
    ears = np.array([[-0.75, -0.15], [-0.6, -0.95], [-0.25, -0.62], [0.25, -0.62], [0.6, -0.95], [0.75, -0.15]])
    face = [_circle(-0.3, 0, 0.07, 8), _circle(0.3, 0, 0.07, 8),
            np.array([[-0.2, 0.3], [-0.1, 0.38], [0, 0.3], [0.1, 0.38], [0.2, 0.3]])]
    whiskers = [np.array([[-0.35, 0.25], [-1.05, 0.15]]), np.array([[0.35, 0.25], [1.05, 0.15]])]
    return [head, ears] + face + whiskers


def _spiral():
    a = np.linspace(0, 6 * np.pi, 120)
    return [np.stack([np.cos(a), np.sin(a)], 1) * (a / a[-1])[:, None]]


def _infinity():
    t = np.linspace(0, 2 * np.pi, 80)
    d = 1 + np.sin(t) ** 2
    return [np.stack([np.cos(t) / d, np.sin(t) * np.cos(t) / d], 1)]


SHAPES = {"heart": _heart, "star": _star, "smiley": _smiley, "cat": _cat, "spiral": _spiral,
          "infinity": _infinity}


def _length(p):
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


def _resample(p, n):
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    u = np.linspace(0, s[-1], max(n, 2))
    return np.stack([np.interp(u, s, p[:, 0]), np.interp(u, s, p[:, 1])], 1)


class BreakDoodler:
    def __init__(self, bm: Beatmap, seed: int | None = None):
        self.starts = np.array([o.time for o in bm.objects], np.float64)
        self.ends = np.maximum.accumulate(np.array([o.end_time for o in bm.objects], np.float64))
        self.preempt = bm.preempt
        self.rng = np.random.default_rng(seed)
        self.done = set()   # номера пауз, где уже рисовали
        self.plan = None    # (times, pos [k,2], smoke [k])

    def _make_plan(self, t0, p0, t_end):
        rng, t = self.rng, t0
        times, pos, smoke = [t0], [np.asarray(p0, np.float64)], [False]

        def go(p, ms, on):
            nonlocal t
            k = max(int(ms / STEP_MS), 1)
            a = pos[-1]
            if on:
                pts = _resample(p, k + 1)[1:]
            else:  # плавный перелёт к началу штриха
                u = np.linspace(0, 1, k + 1)[1:, None]
                pts = a + (p[0] - a) * (u * u * (3 - 2 * u))
            times.extend(t + STEP_MS * np.arange(1, k + 1))
            pos.extend(pts)
            smoke.extend([on] * k)
            t += STEP_MS * k

        def need(names, size):
            return sum(sum(_length(s) * size for s in SHAPES[m]()) / PEN_SPEED + TRAVEL_MS * len(SHAPES[m]())
                       for m in names)

        # сколько рисунков (до трёх) успеваем — ставим их в ряд, чтобы не налезали друг на друга
        names = []
        for name in rng.permutation(list(SHAPES)):
            if len(names) == 3 or need(names + [name], SIZES[len(names) + 1]) > t_end - t:
                break
            names.append(name)
        n = len(names)
        for j, name in enumerate(names):
            size = SIZES[n] * rng.uniform(0.85, 1.0)
            c = np.array([256 + (j - (n - 1) / 2) * 165 + rng.uniform(-15, 15), 192 + rng.uniform(-40, 40)])
            ang = rng.uniform(-0.25, 0.25)
            rot = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            for s in SHAPES[name]():
                s = s @ rot.T * size + c
                # лёгкое дрожание руки
                ph = rng.uniform(0, 2 * np.pi, 2)
                k = np.arange(len(s))[:, None]
                s = s + 1.5 * np.sin(0.35 * k + ph)
                go(s, TRAVEL_MS, False)
                go(s, _length(s) / PEN_SPEED, True)
        if len(times) < 3:
            return None
        return np.array(times), np.array(pos, np.float32), np.array(smoke)

    def step(self, t: float, pos):
        """-> (позиция курсора или None — управляет модель, зажат ли smoke)."""
        if self.plan is not None:
            ts, ps, sm = self.plan
            if t <= ts[-1]:
                k = int(np.searchsorted(ts, t))
                return ps[k], bool(sm[k])
            self.plan = None
        i = int(np.searchsorted(self.starts, t))
        if i >= len(self.starts) or i in self.done:
            return None, False
        prev_end = self.ends[i - 1] if i else -np.inf
        t_end = self.starts[i] - self.preempt - 250
        if t < prev_end + 300 or t_end - t < MIN_BREAK:
            return None, False
        self.done.add(i)
        self.plan = self._make_plan(t, pos, t_end)
        return self.step(t, pos) if self.plan is not None else (None, False)
