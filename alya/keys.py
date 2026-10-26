"""Какой палец жмёт.

Модель решает, когда нажать и сколько держать, а клавишу выбирает правило, как у людей:
если с прошлого нажатия прошло меньше ALT_GAP мс — другим пальцем (стримы, быстрые прыжки),
иначе — основным (K1). Нажатие = клавиша, которой не было на прошлом шаге в выходе модели.
Одна и та же клавиша никогда не жмётся чаще, чем раз в ALT_GAP мс, — поэтому отдельный
лимит пальца в среде больше не нужен.
"""
from __future__ import annotations

import numpy as np

ALT_GAP = 180.0


class KeyAssigner:
    def __init__(self, n: int, alt_gap: float = ALT_GAP):
        self.n, self.alt_gap = n, alt_gap
        self.prev = np.zeros((n, 2), bool)    # что хотела модель на прошлом шаге
        self.out = np.zeros((n, 2), bool)     # что реально зажато
        self.last_t = np.full(n, -1e9)
        self.last_f = np.zeros(n, np.int64)

    def reset(self, idx=None):
        idx = slice(None) if idx is None else np.asarray(idx)
        self.prev[idx] = False
        self.out[idx] = False
        self.last_t[idx] = -1e9
        self.last_f[idx] = 0

    def __call__(self, t, want) -> np.ndarray:
        """t [n] — время шага, want [n,2] — клавиши из модели. -> [n,2] клавиши для игры."""
        want = np.asarray(want, bool).reshape(self.n, 2)
        t = np.broadcast_to(np.asarray(t, np.float64), (self.n,))
        ar = np.arange(self.n)
        tap = (want & ~self.prev).any(1)
        hold = want.any(1)
        f = np.where(t - self.last_t < self.alt_gap, 1 - self.last_f, 0)
        f = np.where(self.out[ar, f], 1 - f, f)  # зажатую клавишу нельзя нажать ещё раз
        new = np.zeros_like(self.out)
        new[ar, f] = True
        out = np.where(tap[:, None], new, self.out & hold[:, None])
        self.last_t = np.where(tap, t, self.last_t)
        self.last_f = np.where(tap, f, self.last_f)
        self.prev, self.out = want.copy(), out.copy()
        return out
