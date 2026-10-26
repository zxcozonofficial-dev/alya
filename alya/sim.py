"""Векторизованный симулятор osu!standard (без игры). N сред шагают одновременно.

Интерфейс абсолютный: advance(t_new, cursor_new, key_new) — так один и тот же код
используется в RL (шаг 16 мс), при разборе человеческих реплеев (BC) и в живой игре.
"""
from __future__ import annotations

import numpy as np

from .beatmap import CIRCLE, SLIDER, SPINNER, PATH_K, Beatmap

N_NEXT = 8           # сколько следующих нот видит модель
# dx, dy, dt, dt_end, circle, slider, spinner, active, aux, valid,
# путь слайдера наперёд: шарик через 100 и 200 мс (dx, dy), конец слайдера (dx, dy)
OBJ_F = 16
GLOB_F = 9           # vx, vy, x, y, key1, radius, preempt, w50, key2
OBS_DIM = N_NEXT * OBJ_F + GLOB_F
MOVE_SCALE = 32.0    # 1.0 действия = 32 osu!px за шаг
FOLLOW_MULT = 2.4    # радиус follow circle
EARLY_MISS = 400.0   # клик раньше окна 50, но ближе 400 мс = miss (как в lazer)
MAX_SPIN = 0.05      # рад/мс (~477 rpm)
CENTER = np.array([256.0, 192.0], dtype=np.float32)

J300, J100, J50, JMISS = 0, 1, 2, 3


class MapPool:
    """Карты, упакованные в паддированные массивы для векторного доступа."""

    def __init__(self, maps: list[Beatmap]):
        assert maps, "пустой пул карт"
        self.maps = maps
        P = len(maps)
        M = max(len(m.objects) for m in maps) + N_NEXT + 1
        self.n_obj = np.array([len(m.objects) for m in maps], dtype=np.int64)
        self.x = np.zeros((P, M), np.float32)
        self.y = np.zeros((P, M), np.float32)
        self.t = np.full((P, M), 1e9, np.float64)
        self.te = np.full((P, M), 1e9, np.float64)
        self.kind = np.full((P, M), -1, np.int64)
        self.slides = np.ones((P, M), np.int64)
        self.path_ptr = np.zeros((P, M), np.int64)
        self.cp_start = np.zeros((P, M), np.int64)
        self.cp_count = np.zeros((P, M), np.int64)
        self.spins_req = np.ones((P, M), np.float32)
        paths = [np.zeros((PATH_K, 2), np.float32)]
        cps = [1e18]
        for i, m in enumerate(maps):
            for j, o in enumerate(m.objects):
                self.x[i, j], self.y[i, j] = o.x, o.y
                self.t[i, j], self.te[i, j] = o.time, o.end_time
                self.kind[i, j], self.slides[i, j] = o.kind, o.slides
                if o.kind == SLIDER:
                    self.path_ptr[i, j] = len(paths)
                    paths.append(o.path)
                    self.cp_start[i, j] = len(cps)
                    self.cp_count[i, j] = len(o.checkpoints)
                    cps.extend(o.checkpoints)
                elif o.kind == SPINNER:
                    secs = (o.end_time - o.time) / 1000 * m.rate  # требование считается во времени карты
                    self.spins_req[i, j] = max(int(secs * m.spins_per_second), 1)
        self.paths = np.stack(paths)
        self.cps = np.array(cps, np.float64)
        self.radius = np.array([m.radius for m in maps], np.float32)
        self.rate = np.array([m.rate for m in maps], np.float32)
        self.preempt = np.array([m.preempt for m in maps], np.float32)
        w = np.array([m.windows for m in maps], np.float32)
        self.w300, self.w100, self.w50 = w[:, 0], w[:, 1], w[:, 2]


class Simulator:
    def __init__(self, pool: MapPool, n: int):
        self.pool, self.n = pool, n
        self.map_id = np.zeros(n, np.int64)
        self.t = np.zeros(n, np.float64)
        self.pos = np.tile(CENTER, (n, 1))
        self.vel = np.zeros((n, 2), np.float32)
        self.key = np.zeros((n, 2), bool)   # K1, K2
        self.head = np.zeros(n, np.int64)        # первый объект с неоценённой головой
        self.body = np.full(n, -1, np.int64)     # активный слайдер/спиннер
        self.body_head_hit = np.zeros(n, bool)
        self.tracking = np.zeros(n, bool)       # курсор «держит» шарик слайдера
        self.cp_ptr = np.zeros(n, np.int64)
        self.cp_end = np.zeros(n, np.int64)
        self.cp_hits = np.zeros(n, np.int64)
        self.spin_rot = np.zeros(n, np.float32)
        self.combo = np.zeros(n, np.int64)
        self.max_combo = np.zeros(n, np.int64)
        self.counts = np.zeros((n, 4), np.int64)

    # ---------------------------------------------------------- состояние
    def reset(self, idx, map_id, start_obj, t0, pos):
        idx = np.asarray(idx)
        self.map_id[idx] = map_id
        self.head[idx] = start_obj
        self.t[idx] = t0
        self.pos[idx] = pos
        self.vel[idx] = 0
        self.key[idx] = False
        self.body[idx] = -1
        self.combo[idx] = 0
        self.max_combo[idx] = 0
        self.counts[idx] = 0

    def finished(self) -> np.ndarray:
        return (self.head >= self.pool.n_obj[self.map_id]) & (self.body < 0)

    def _g(self, arr, obj):
        return arr[self.map_id, np.clip(obj, 0, arr.shape[1] - 1)]

    def ball(self, obj, t):
        """Позиция шарика слайдера obj (по средам) в момент t."""
        return self.ball_at(self.map_id, obj, t)

    def ball_at(self, mid, obj, t):
        """То же для произвольных плоских массивов (карта, объект, время) одной длины."""
        P = self.pool
        obj = np.clip(obj, 0, P.x.shape[1] - 1)
        t0, t1 = P.t[mid, obj], P.te[mid, obj]
        slides = P.slides[mid, obj]
        dur = np.maximum(t1 - t0, 1e-6)
        span = dur / slides
        el = np.clip(t - t0, 0, dur)
        i = np.minimum(np.floor(el / span), slides - 1)
        f = el / span - i
        f = np.where(i % 2 == 1, 1 - f, f)
        pos = f * (PATH_K - 1)
        j = np.minimum(pos.astype(np.int64), PATH_K - 2)
        w = (pos - j)[:, None].astype(np.float32)
        ptr = P.path_ptr[mid, obj]
        return P.paths[ptr, j] * (1 - w) + P.paths[ptr, j + 1] * w

    def _judge(self, mask, j, ev):
        if not mask.any():
            return
        idx = np.nonzero(mask)[0]
        j = j[idx] if np.ndim(j) else np.full(len(idx), j)
        np.add.at(self.counts, (idx, j), 1)
        np.add.at(ev, (idx, j), 1)
        hit = j != JMISS
        self.combo[idx] = np.where(hit, self.combo[idx] + 1, 0)
        self.max_combo[idx] = np.maximum(self.max_combo[idx], self.combo[idx])

    def _finish_body(self, mask, ev):
        """Оценить слайдер/спиннер целиком и освободить слот body."""
        mask = mask & (self.body >= 0)
        if not mask.any():
            return
        P = self.pool
        kind = self._g(P.kind, self.body)
        total = 1 + self._g(P.cp_count, self.body)
        ratio = (self.body_head_hit + self.cp_hits) / total
        js = np.where(ratio >= 1, J300, np.where(ratio >= 0.5, J100, np.where(ratio > 0, J50, JMISS)))
        rot = self.spin_rot / (2 * np.pi) / self._g(P.spins_req, self.body)
        jsp = np.where(rot >= 1, J300, np.where(rot >= 0.9, J100, np.where(rot >= 0.5, J50, JMISS)))
        self._judge(mask, np.where(kind == SPINNER, jsp, js), ev)
        self.body[mask] = -1

    def _start_body(self, mask, obj, head_hit, ev):
        mask = mask.copy()
        self._finish_body(mask & (self.body >= 0), ev)  # 2B: старое тело закрываем
        self.body[mask] = obj[mask]
        self.body_head_hit[mask] = head_hit[mask]
        self.tracking[mask] = head_hit[mask]
        self.cp_ptr[mask] = self._g(self.pool.cp_start, obj)[mask]
        self.cp_end[mask] = self.cp_ptr[mask] + self._g(self.pool.cp_count, obj)[mask]
        self.cp_hits[mask] = 0
        self.spin_rot[mask] = 0

    # ---------------------------------------------------------- шаг
    def advance(self, t_new, pos_new, key_new, click_t=None, click_pos=None):
        """Продвинуть все среды. Возвращает dict событий (массивы по средам).

        click_t / click_pos — точные время и позиция клика внутри шага (из реплея);
        по умолчанию клик считается в момент t_new в позиции pos_new.
        """
        P = self.pool
        n = self.n
        t_new = np.broadcast_to(np.asarray(t_new, np.float64), (n,)).copy()
        pos_new = np.asarray(pos_new, np.float32).reshape(n, 2)
        key_new = np.asarray(key_new, bool).reshape(n, 2)
        dt = np.maximum(t_new - self.t, 1e-3)
        # каждая нажатая клавиша — отдельный клик (K1+K2 разом = два клика, как в osu!)
        click = (key_new & ~self.key).sum(1)
        held, held_old = key_new.any(1), self.key.any(1)
        ev = np.zeros((n, 4), np.int64)
        cp_hit = np.zeros(n, np.int64)
        cp_miss = np.zeros(n, np.int64)
        timing = np.zeros(n, np.float32)  # сумма |ошибка| / окно 300 по попаданиям этого шага
        r = P.radius[self.map_id]

        # --- тело: тики/повторы/конец слайдера
        is_slider = (self.body >= 0) & (self._g(P.kind, self.body) == SLIDER)
        # слежение за шариком, как в игре: пока держишь — радиус follow circle (2.4r),
        # потерял — вернуть можно только подведя курсор к самому шарику (r)
        if is_slider.any():
            d_ball = np.linalg.norm(pos_new - self.ball(self.body, t_new), axis=1)
            self.tracking = is_slider & held & (d_ball <= np.where(self.tracking, FOLLOW_MULT * r, r))
        for _ in range(64):
            m = is_slider & (self.cp_ptr < self.cp_end)
            cp_t = np.where(m, P.cps[np.minimum(self.cp_ptr, len(P.cps) - 1)], 1e18)
            m &= cp_t <= t_new
            if not m.any():
                break
            ball = self.ball(self.body, cp_t)
            ok = m & self.tracking
            self.cp_hits += ok
            cp_hit += ok
            cp_miss += m & ~ok
            self.combo[m & ~ok] = 0
            self.cp_ptr += m
        # --- тело: вращение спиннера
        is_spin = (self.body >= 0) & (self._g(P.kind, self.body) == SPINNER)
        a0 = np.arctan2(self.pos[:, 1] - CENTER[1], self.pos[:, 0] - CENTER[0])
        a1 = np.arctan2(pos_new[:, 1] - CENTER[1], pos_new[:, 0] - CENTER[0])
        da = np.abs((a1 - a0 + np.pi) % (2 * np.pi) - np.pi)
        da = np.minimum(da, MAX_SPIN * dt * P.rate[self.map_id]).astype(np.float32)  # лимит во времени игры
        spin_delta = np.where(is_spin & held & held_old, da, 0)
        self.spin_rot += spin_delta
        # --- конец тела
        body_end = self._g(P.te, self.body)
        self._finish_body((self.body >= 0) & (t_new >= body_end), ev)

        # --- головы (до двух за шаг: плотные стримы)
        ct = t_new if click_t is None else np.asarray(click_t, np.float64).reshape(n)
        cpos = pos_new if click_pos is None else np.asarray(click_pos, np.float32).reshape(n, 2)
        clicked = click.copy()
        for _ in range(3):
            h = self.head
            alive = h < P.n_obj[self.map_id]
            kind = self._g(P.kind, h)
            th = self._g(P.t, h)
            # спиннер начинается сам
            sp = alive & (kind == SPINNER) & (t_new >= th)
            if sp.any():
                self._start_body(sp, h, np.ones(n, bool), ev)
                self.head += sp
                continue
            hittable = alive & (kind != SPINNER)
            hx = np.stack([self._g(P.x, h), self._g(P.y, h)], 1)
            inside = np.linalg.norm(cpos - hx, axis=1) <= r
            err = ct - th
            aerr = np.abs(err)
            w300, w100, w50 = P.w300[self.map_id], P.w100[self.map_id], P.w50[self.map_id]
            press = hittable & (clicked > 0) & inside
            hit = press & (aerr <= w50)
            early = press & (err < -w50) & (err >= -EARLY_MISS)
            jh = np.where(aerr <= w300, J300, np.where(aerr <= w100, J100, J50))
            late = hittable & ~hit & ~early & (err > w50)
            done = hit | early | late
            if not done.any():
                break
            circ = kind == CIRCLE
            self._judge(done & circ & hit, jh, ev)
            self._judge(done & circ & ~hit, JMISS, ev)
            sl = done & (kind == SLIDER)
            if sl.any():
                self.combo[sl & ~hit] = 0
                self.combo[sl & hit] += 1
                self._start_body(sl, h, hit, ev)
            timing += np.where(hit, aerr / w300, 0).astype(np.float32)
            clicked -= hit | early   # один клик — одна нота
            self.head += done

        self.vel = (pos_new - self.pos).astype(np.float32)
        self.pos = pos_new.copy()
        self.key = key_new.copy()
        self.t = t_new
        # wasted — клики, не задевшие ни одной ноты (даблтапы, клики в пустоту)
        return {"judge": ev, "cp_hit": cp_hit, "cp_miss": cp_miss, "spin": spin_delta, "click": click,
                "wasted": clicked, "timing": timing}

    # ---------------------------------------------------------- наблюдение
    def target(self) -> np.ndarray:
        """Куда сейчас стоит вести курсор: шарик, центр спиннера или следующая голова."""
        P = self.pool
        has_body = self.body >= 0
        obj = np.where(has_body, self.body, self.head)
        tgt = np.stack([self._g(P.x, obj), self._g(P.y, obj)], 1)
        kind = self._g(P.kind, obj)
        sl = has_body & (kind == SLIDER)
        if sl.any():
            tgt[sl] = self.ball(obj, self.t)[sl]
        return tgt

    def obs(self) -> np.ndarray:
        P = self.pool
        n = self.n
        mid = self.map_id[:, None]
        has_body = self.body >= 0
        start = np.where(has_body, self.body, self.head)
        objs = start[:, None] + np.arange(N_NEXT)[None]
        objs_c = np.clip(objs, 0, P.x.shape[1] - 1)
        valid = objs < P.n_obj[mid]
        t = self.t[:, None]
        to = P.t[mid, objs_c]
        te = P.te[mid, objs_c]
        valid &= (to - t) <= P.preempt[mid]          # только видимые ноты (AR)
        kind = P.kind[mid, objs_c]
        tx = P.x[mid, objs_c].copy()
        ty = P.y[mid, objs_c].copy()
        active = np.zeros((n, N_NEXT), bool)
        active[:, 0] = has_body
        aux = np.zeros((n, N_NEXT), np.float32)
        sl = has_body & (kind[:, 0] == SLIDER)
        if sl.any():
            b = self.ball(self.body, self.t)
            tx[sl, 0], ty[sl, 0] = b[sl, 0], b[sl, 1]
            done = (self.cp_ptr - self._g(P.cp_start, self.body)) / np.maximum(self._g(P.cp_count, self.body), 1)
            aux[sl, 0] = done[sl]
        sp = has_body & (kind[:, 0] == SPINNER)
        if sp.any():
            aux[sp, 0] = np.minimum(self.spin_rot / (2 * np.pi) / self._g(P.spins_req, self.body), 1.5)[sp]
        f = np.zeros((n, N_NEXT, OBJ_F), np.float32)
        f[..., 0] = (tx - self.pos[:, :1]) / 256
        f[..., 1] = (ty - self.pos[:, 1:]) / 256
        f[..., 2] = np.clip((to - t) / 1000, -0.5, 3)
        f[..., 3] = np.clip((te - t) / 1000, -0.5, 3)
        f[..., 4] = kind == CIRCLE
        f[..., 5] = kind == SLIDER
        f[..., 6] = kind == SPINNER
        f[..., 7] = active
        f[..., 8] = aux
        # путь слайдера наперёд (для круга/спиннера — его же позиция): одним векторным вызовом
        is_sl = (kind == SLIDER).reshape(-1)
        midf = np.repeat(self.map_id, N_NEXT)
        of = objs_c.reshape(-1)
        base = np.maximum(t, to).reshape(-1)
        tq = np.concatenate([base + 100.0, base + 200.0, te.reshape(-1)])
        b = self.ball_at(np.tile(midf, 3), np.tile(of, 3), tq).reshape(3, n, N_NEXT, 2)
        sl3 = is_sl.reshape(n, N_NEXT)[None]
        bx = np.where(sl3, b[..., 0], tx[None])
        by = np.where(sl3, b[..., 1], ty[None])
        for j in range(3):
            f[..., 10 + 2 * j] = (bx[j] - self.pos[:, :1]) / 256
            f[..., 11 + 2 * j] = (by[j] - self.pos[:, 1:]) / 256
        f *= valid[..., None]
        f[..., 9] = valid
        g = np.zeros((n, GLOB_F), np.float32)
        g[:, 0:2] = self.vel / MOVE_SCALE
        g[:, 2] = self.pos[:, 0] / 512 - 0.5
        g[:, 3] = self.pos[:, 1] / 384 - 0.5
        g[:, 4] = self.key[:, 0]
        g[:, 8] = self.key[:, 1]
        g[:, 5] = P.radius[self.map_id] / 64
        g[:, 6] = P.preempt[self.map_id] / 1000
        g[:, 7] = P.w50[self.map_id] / 200
        return np.concatenate([f.reshape(n, -1), g], 1)

    def accuracy(self) -> np.ndarray:
        c = self.counts
        tot = np.maximum(c.sum(1), 1)
        return (300 * c[:, 0] + 100 * c[:, 1] + 50 * c[:, 2]) / (300 * tot)


def clamp_cursor(pos: np.ndarray) -> np.ndarray:
    return np.clip(pos, [-64, -48], [576, 432]).astype(np.float32)
