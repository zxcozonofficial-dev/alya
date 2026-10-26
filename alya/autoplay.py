"""Идеальный автоплей по карте — эталон для проверки симулятора и пример данных для BC."""
from __future__ import annotations

import math

import numpy as np

from .beatmap import SLIDER, SPINNER, Beatmap


def autoplay(bm: Beatmap, times: np.ndarray):
    """Вернуть (pos [T,2], keys [T,2]) на сетке times."""
    objs = bm.objects
    pos = np.zeros((len(times), 2), np.float32)
    key = np.zeros(len(times), bool)
    j = 0
    last_pos, last_t = (256.0, 192.0), -1e9
    for i, t in enumerate(times):
        while j < len(objs) and objs[j].end_time < t - 8:
            last_pos, last_t = objs[j].end_pos(), objs[j].end_time
            j += 1
        if j >= len(objs):
            pos[i] = last_pos
            continue
        o = objs[j]
        if o.time - 8 <= t <= o.end_time + 8 and o.kind != 0:
            key[i] = True
            if o.kind == SLIDER:
                pos[i] = o.ball_pos(min(t, o.end_time))
            else:  # спиннер: крутим по кругу на максимальной скорости
                a = (t - o.time) * 0.05 * bm.rate  # лимит вращения — во времени игры
                pos[i] = (256 + 60 * math.cos(a), 192 + 60 * math.sin(a))
            continue
        if -8 <= t - o.time < 8:
            key[i] = True
        # линейно от конца прошлого объекта к голове следующего
        # прилетаем к ноте к моменту нажатия (до 8 мс раньше её времени)
        arrive = o.time - 8
        span = max(arrive - max(last_t, arrive - 1000), 1)
        f = min(max((t - (arrive - span)) / span, 0), 1)
        pos[i] = (last_pos[0] + (o.x - last_pos[0]) * f, last_pos[1] + (o.y - last_pos[1]) * f)
    # каждая нота — новое нажатие; клавиши чередуются K1/K2, как у человека
    taps = np.zeros(len(times), bool)
    ot = np.array([o.time for o in objs])
    near = np.searchsorted(ot, times - 8, side="right")   # первая нота с time > t - 8
    ok = near < len(ot)
    taps[ok] = (ot[near[ok]] - times[ok]) <= 8
    keys = np.zeros((len(times), 2), bool)
    col = 1
    for i in range(len(times)):
        if key[i] and (taps[i] or i == 0 or not key[i - 1]):
            col ^= 1
        keys[i, col] = key[i]
    return pos, keys
