"""Изгиб траектории между нотами: прыжки идут дугой, а «туда-обратно» складывается в овал.

Поверх курсора модели добавляется смещение поперёк линии «конец прошлой ноты → следующая
нота»: A · sin²(π·u), где u — доля пройденного времени в середине пути: GUARD интервала у каждой
ноты курсор идёт без изгиба, потому что модель жмёт чуть раньше или позже ноты (попадания не страдают), на слайдерах и спиннерах изгиба нет. Сторона изгиба по ходу
движения одна и та же, поэтому 1→2 и 2→1 огибают разные стороны и рисуют петлю.
Модель видит курсор без изгиба (иначе она стала бы с ним бороться).
"""
from __future__ import annotations

import numpy as np

from .beatmap import SPINNER, Beatmap

MIN_GAP, MAX_GAP = 60.0, 700.0   # мс: двойки 1/8 и долгие паузы не гнём (стримы гнутся слабо — прыжок короткий)
MAX_BEND = 70.0                 # osu!px
GUARD = 0.2                     # доля интервала у каждой ноты без изгиба
FAST_GAP = 250.0                # на прыжках быстрее этого дуга меньше


class CurveBend:
    def __init__(self, bm: Beatmap, strength: float = 0.2, seed: int | None = None):
        objs = bm.objects
        self.t = np.array([o.time for o in objs])
        self.te = np.array([o.end_time for o in objs])
        self.xy = np.array([(o.x, o.y) for o in objs], np.float64)
        self.end_xy = np.array([o.end_pos() for o in objs], np.float64)
        self.spinner = np.array([o.kind == SPINNER for o in objs])
        self.strength = strength
        rng = np.random.default_rng(seed)
        # у каждого прыжка своя амплитуда (±30%); сторона меняется редко — после пауз
        self.amp = strength * rng.uniform(0.7, 1.3, len(objs))
        side = np.empty(len(objs))
        s = rng.choice([-1.0, 1.0])
        for i in range(len(objs)):
            if i and self.t[i] - self.te[i - 1] > 1500 and rng.random() < 0.5:
                s = -s
            side[i] = s
        self.side = side

    def offset(self, t: float) -> np.ndarray:
        """Смещение курсора (osu!px) в момент t."""
        if self.strength <= 0:
            return np.zeros(2)
        i = int(np.searchsorted(self.t, t, side="right"))  # следующая нота
        if i == 0 or i >= len(self.t):
            return np.zeros(2)
        p = i - 1
        t0, t1 = self.te[p], self.t[i]
        gap = t1 - t0
        if t < t0 or gap < MIN_GAP or gap > MAX_GAP or self.spinner[p] or self.spinner[i]:
            return np.zeros(2)
        a, b = self.end_xy[p], self.xy[i]
        d = b - a
        dist = float(np.hypot(*d))
        if dist < 20:
            return np.zeros(2)
        normal = np.array([-d[1], d[0]]) / dist * self.side[i]
        u = (t - t0 - GUARD * gap) / ((1 - 2 * GUARD) * gap)
        if not 0 < u < 1:
            return np.zeros(2)
        speed = np.clip((gap - MIN_GAP) / (FAST_GAP - MIN_GAP), 0.4, 1.0)
        bend = min(self.amp[i] * dist, MAX_BEND) * speed * np.sin(np.pi * u) ** 2
        return normal * bend

    def __call__(self, t: float, pos) -> np.ndarray:
        return np.asarray(pos, np.float64) + self.offset(t)
