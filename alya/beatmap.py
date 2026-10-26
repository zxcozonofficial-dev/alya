"""Парсер .osu (osu!standard): Difficulty, TimingPoints, HitObjects, пути слайдеров."""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CIRCLE, SLIDER, SPINNER = 0, 1, 2
# биты модов osu!
EZ, HR, DT, HT, NC = 2, 16, 64, 256, 512
PATH_K = 32  # точек на равномерно перепараметризованный путь слайдера
PLAYFIELD_CENTER = (256.0, 192.0)


@dataclass
class HitObject:
    x: float
    y: float
    time: float
    kind: int
    end_time: float
    new_combo: bool = False
    slides: int = 1
    length: float = 0.0
    curve_type: str = ""
    path: np.ndarray | None = None          # [PATH_K, 2], равные шаги по длине дуги
    checkpoints: list[float] = field(default_factory=list)  # тики, повторы, конец

    @property
    def span_duration(self) -> float:
        return (self.end_time - self.time) / max(self.slides, 1)

    def ball_pos(self, t: float) -> tuple[float, float]:
        """Позиция шарика слайдера в момент t."""
        if self.kind != SLIDER:
            return self.x, self.y
        return slider_ball(self.path, self.time, self.end_time, self.slides, t)

    def end_pos(self) -> tuple[float, float]:
        if self.kind != SLIDER:
            return self.x, self.y
        p = self.path[-1] if self.slides % 2 == 1 else self.path[0]
        return float(p[0]), float(p[1])


@dataclass
class Beatmap:
    path: str
    md5: str
    title: str = ""
    artist: str = ""
    version: str = ""
    mode: int = 0
    hp: float = 5.0
    cs: float = 5.0
    od: float = 5.0
    ar: float = 5.0
    slider_multiplier: float = 1.4
    slider_tick_rate: float = 1.0
    stack_leniency: float = 0.7
    format_version: int = 14
    mods: int = 0
    rate: float = 1.0          # DT 1.5 / HT 0.75; все времена объектов — реальные мс
    timing_points: list[tuple[float, float, bool]] = field(default_factory=list)
    objects: list[HitObject] = field(default_factory=list)

    @property
    def radius(self) -> float:
        return 54.4 - 4.48 * self.cs

    @property
    def preempt(self) -> float:
        ar = self.ar
        return (1200 + 600 * (5 - ar) / 5 if ar < 5 else 1200 - 750 * (ar - 5) / 5) / self.rate

    @property
    def windows(self) -> tuple[float, float, float]:
        od, r = self.od, self.rate
        return (80 - 6 * od) / r, (140 - 8 * od) / r, (200 - 10 * od) / r

    @property
    def spins_per_second(self) -> float:
        od = self.od
        return 5 - 2 * (5 - od) / 5 if od < 5 else 5 + 2.5 * (od - 5) / 5

    def __repr__(self) -> str:
        return (f"Beatmap({self.artist} - {self.title} [{self.version}], "
                f"{len(self.objects)} obj, CS{self.cs} AR{self.ar} OD{self.od})")


# ---------------------------------------------------------------- кривые

def _bezier(points: np.ndarray, n: int) -> np.ndarray:
    ts = np.linspace(0.0, 1.0, n)[:, None, None]
    p = np.broadcast_to(points[None], (n,) + points.shape).copy()
    for k in range(len(points) - 1, 0, -1):  # де Кастельжо
        p = p[:, :k] * (1 - ts) + p[:, 1:k + 1] * ts
    return p[:, 0]


def _perfect_circle(a, b, c, n: int = 64) -> np.ndarray | None:
    a, b, c = (np.asarray(v, dtype=np.float64) for v in (a, b, c))
    d = 2 * (a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1]))
    if abs(d) < 1e-3:
        return None
    aa, bb, cc = a @ a, b @ b, c @ c
    center = np.array([
        (aa * (b[1] - c[1]) + bb * (c[1] - a[1]) + cc * (a[1] - b[1])) / d,
        (aa * (c[0] - b[0]) + bb * (a[0] - c[0]) + cc * (b[0] - a[0])) / d,
    ])
    r = np.linalg.norm(a - center)
    t0 = math.atan2(a[1] - center[1], a[0] - center[0])
    t1 = math.atan2(c[1] - center[1], c[0] - center[0])
    while t1 < t0:
        t1 += 2 * math.pi
    direction, rng = 1.0, t1 - t0
    ortho = np.array([c[1] - a[1], -(c[0] - a[0])])
    if ortho @ (b - a) < 0:
        direction, rng = -1.0, 2 * math.pi - rng
    th = t0 + direction * np.linspace(0, 1, n) * rng
    return center + r * np.stack([np.cos(th), np.sin(th)], axis=1)


def _catmull(points: np.ndarray, n: int = 32) -> np.ndarray:
    out = []
    for i in range(len(points) - 1):
        p1, p2 = points[i], points[i + 1]
        p0 = points[i - 1] if i > 0 else p1
        p3 = points[i + 2] if i + 2 < len(points) else 2 * p2 - p1
        t = np.linspace(0, 1, n)[:, None]
        out.append(0.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                          + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.concatenate(out)


def build_polyline(curve_type: str, pts: np.ndarray) -> np.ndarray:
    if len(pts) < 2:
        return np.repeat(pts, 2, axis=0)
    if curve_type == "L":
        return pts
    if curve_type == "P" and len(pts) == 3:
        arc = _perfect_circle(*pts)
        if arc is not None:
            return arc
    if curve_type == "C":
        return _catmull(pts)
    # Безье (и фолбэк для P): красные якоря = повторённые точки делят сегменты
    segs, cur = [], [pts[0]]
    for p in pts[1:]:
        if np.allclose(p, cur[-1]):
            if len(cur) > 1:
                segs.append(np.array(cur))
            cur = [p]
        else:
            cur.append(p)
    if len(cur) > 1:
        segs.append(np.array(cur))
    if not segs:
        return np.repeat(pts[:1], 2, axis=0)
    out = [_bezier(s, max(8, 12 * len(s))) for s in segs]
    return np.concatenate(out)


def resample_path(poly: np.ndarray, length: float, k: int = PATH_K) -> np.ndarray:
    """Обрезать/продлить полилинию до length и взять k точек через равные расстояния."""
    poly = np.asarray(poly, dtype=np.float64)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    keep = np.concatenate([[True], seg > 1e-9])
    poly = poly[keep]
    if len(poly) < 2:
        return np.repeat(poly[:1], k, axis=0).astype(np.float32)
    seg = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    if length <= 0:
        length = cum[-1]
    if cum[-1] < length:  # продлить последний сегмент линейно
        d = poly[-1] - poly[-2]
        poly = np.vstack([poly, poly[-1] + d / np.linalg.norm(d) * (length - cum[-1])])
        cum = np.append(cum, length)
    s = np.linspace(0, length, k)
    x = np.interp(s, cum, poly[:, 0])
    y = np.interp(s, cum, poly[:, 1])
    return np.stack([x, y], axis=1).astype(np.float32)


def slider_ball(path: np.ndarray, t0: float, t1: float, slides: int, t: float):
    dur = max(t1 - t0, 1e-6)
    span = dur / slides
    el = min(max(t - t0, 0.0), dur)
    i = min(int(el // span), slides - 1)
    f = el / span - i
    if i % 2 == 1:
        f = 1 - f
    pos = f * (len(path) - 1)
    j = min(int(pos), len(path) - 2)
    w = pos - j
    p = path[j] * (1 - w) + path[j + 1] * w
    return float(p[0]), float(p[1])


# ---------------------------------------------------------------- парсинг

def _timing_at(points, t):
    """(beat_length красной линии, множитель SV) в момент t."""
    beat, sv = None, 1.0
    for time, bl, inherited in points:
        if time > t and beat is not None:
            break
        if not inherited:
            beat, sv = bl, 1.0
        elif bl < 0:
            sv = min(max(-100.0 / bl, 0.1), 10.0)
    return (beat if beat else 500.0), sv


def parse_osu(path: str | Path, mods: int = 0) -> Beatmap:
    """mods — битовая маска osu! (учитываются EZ/HR/DT/NC/HT)."""
    raw = Path(path).read_bytes()
    bm = Beatmap(path=str(path), md5=hashlib.md5(raw).hexdigest())
    text = raw.decode("utf-8-sig", errors="replace")
    section = ""
    ar_set = False
    obj_lines = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("osu file format v"):
            try:
                bm.format_version = int(line.rsplit("v", 1)[1])
            except ValueError:
                pass
            continue
        if not line or line.startswith("//"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1]
            continue
        if section in ("General", "Metadata", "Difficulty"):
            if ":" not in line:
                continue
            k, v = (s.strip() for s in line.split(":", 1))
            if section == "General" and k == "Mode":
                bm.mode = int(v)
            elif section == "General" and k == "StackLeniency":
                bm.stack_leniency = float(v)
            elif section == "Metadata":
                if k == "Title": bm.title = v
                elif k == "Artist": bm.artist = v
                elif k == "Version": bm.version = v
            elif section == "Difficulty":
                f = float(v)
                if k == "HPDrainRate": bm.hp = f
                elif k == "CircleSize": bm.cs = f
                elif k == "OverallDifficulty": bm.od = f
                elif k == "ApproachRate": bm.ar, ar_set = f, True
                elif k == "SliderMultiplier": bm.slider_multiplier = f
                elif k == "SliderTickRate": bm.slider_tick_rate = f
        elif section == "TimingPoints":
            p = line.split(",")
            if len(p) < 2:
                continue
            bl = float(p[1])
            inherited = (len(p) > 6 and p[6].strip() == "0") or bl < 0
            bm.timing_points.append((float(p[0]), bl, inherited))
        elif section == "HitObjects":
            obj_lines.append(line)
    if not ar_set:
        bm.ar = bm.od
    bm.timing_points.sort(key=lambda p: (p[0], p[2]))  # красная перед зелёной в то же время
    for line in obj_lines:
        o = _parse_object(bm, line)
        if o is not None:
            bm.objects.append(o)
    bm.objects.sort(key=lambda o: o.time)
    apply_difficulty_mods(bm, mods)
    apply_stacking(bm)
    apply_rate(bm)
    return bm


def mods_from_string(s: str) -> int:
    names = {"NM": 0, "NF": 1, "EZ": EZ, "TD": 4, "HD": 8, "HR": HR, "SD": 32, "DT": DT, "HT": HT,
             "NC": NC | DT, "FL": 1024, "SO": 4096, "PF": 16384 | 32}
    s = s.upper().replace("+", "")
    return sum(names[s[i:i + 2]] for i in range(0, len(s), 2)) if s else 0


def mods_to_string(mods: int) -> str:
    out = [n for n, b in (("EZ", EZ), ("HD", 8), ("HR", HR), ("HT", HT), ("FL", 1024)) if mods & b]
    out.insert(2 if len(out) >= 2 else len(out), "NC" if mods & NC else "DT" if mods & DT else "")
    return "".join(out) or "NM"


def apply_difficulty_mods(bm: Beatmap, mods: int):
    """EZ/HR меняют CS/AR/OD/HP, HR ещё и отражает карту по вертикали (как в stable — до стаков)."""
    bm.mods = mods
    if mods & HR:
        bm.cs = min(bm.cs * 1.3, 10.0)
        bm.ar, bm.od, bm.hp = (min(v * 1.4, 10.0) for v in (bm.ar, bm.od, bm.hp))
        for o in bm.objects:
            if o.kind != SPINNER:
                o.y = 384 - o.y
            if o.path is not None:
                o.path = o.path.copy()
                o.path[:, 1] = 384 - o.path[:, 1]
    elif mods & EZ:
        bm.cs, bm.ar, bm.od, bm.hp = bm.cs * 0.5, bm.ar * 0.5, bm.od * 0.5, bm.hp * 0.5


def apply_rate(bm: Beatmap):
    """DT/NC/HT: перевести времена объектов в реальные мс; окна и preempt делятся на rate в свойствах."""
    rate = 1.5 if bm.mods & (DT | NC) else 0.75 if bm.mods & HT else 1.0
    if rate == 1.0:
        return
    bm.rate = rate
    for o in bm.objects:
        o.time /= rate
        o.end_time /= rate
        o.checkpoints = [c / rate for c in o.checkpoints]


def apply_stacking(bm: Beatmap):
    """Смещение стопок нот (алгоритм stable для формата v6+; для старых — упрощённо он же)."""
    objs = bm.objects
    n = len(objs)
    if n < 2:
        return
    height = [0] * n
    threshold = bm.preempt * bm.stack_leniency
    stack_dist = 3.0

    def dist(a, b):
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def path_end(o):
        return (float(o.path[-1][0]), float(o.path[-1][1])) if o.kind == SLIDER else (o.x, o.y)

    for i in range(n - 1, 0, -1):
        oi = objs[i]
        if height[i] != 0 or oi.kind == SPINNER:
            continue
        cur = i
        if oi.kind == CIRCLE:
            k = i
            while k > 0:
                k -= 1
                on = objs[k]
                if on.kind == SPINNER:
                    continue
                if objs[cur].time - on.end_time > threshold:
                    break
                if on.kind == SLIDER and dist(path_end(on), (objs[cur].x, objs[cur].y)) < stack_dist:
                    offset = height[cur] - height[k] + 1
                    for j in range(k + 1, i + 1):
                        if dist(path_end(on), (objs[j].x, objs[j].y)) < stack_dist:
                            height[j] -= offset
                    break
                if dist((on.x, on.y), (objs[cur].x, objs[cur].y)) < stack_dist:
                    height[k] = height[cur] + 1
                    cur = k
        elif oi.kind == SLIDER:
            k = i
            while k > 0:
                k -= 1
                on = objs[k]
                if on.kind == SPINNER:
                    continue
                if objs[cur].time - on.time > threshold:
                    break
                if dist(path_end(on), (objs[cur].x, objs[cur].y)) < stack_dist:
                    height[k] = height[cur] + 1
                    cur = k
    scale = (1.0 - 0.7 * (bm.cs - 5) / 5) / 2
    for o, hgt in zip(objs, height):
        if hgt == 0 or o.kind == SPINNER:
            continue
        off = hgt * scale * -6.4
        o.x += off
        o.y += off
        if o.path is not None:
            o.path = o.path + np.float32(off)


def _parse_object(bm: Beatmap, line: str) -> HitObject | None:
    p = line.split(",")
    if len(p) < 4:
        return None
    x, y, t, typ = float(p[0]), float(p[1]), float(p[2]), int(p[3])
    nc = bool(typ & 4)
    if typ & 1:
        return HitObject(x, y, t, CIRCLE, t, nc)
    if typ & 8:
        end = float(p[5]) if len(p) > 5 else t
        return HitObject(256.0, 192.0, t, SPINNER, max(end, t), nc)
    if typ & 2:
        curve = p[5].split("|")
        ctype = curve[0] if curve[0] in "LPBC" and curve[0] else "B"
        pts = [(x, y)]
        for c in curve[1:]:
            if ":" in c:
                a, b = c.split(":")[:2]
                pts.append((float(a), float(b)))
        slides = max(int(p[6]), 1) if len(p) > 6 else 1
        length = float(p[7]) if len(p) > 7 else 0.0
        poly = build_polyline(ctype, np.array(pts, dtype=np.float64))
        if length <= 0:
            length = float(np.linalg.norm(np.diff(poly, axis=0), axis=1).sum())
        path = resample_path(poly, length)
        beat, sv = _timing_at(bm.timing_points, t)
        px_per_beat = bm.slider_multiplier * 100 * sv
        span = length / px_per_beat * beat
        o = HitObject(x, y, t, SLIDER, t + span * slides, nc, slides, length, ctype, path)
        o.checkpoints = _checkpoints(o, span, px_per_beat / bm.slider_tick_rate)
        return o
    return None  # mania hold и прочее


def _checkpoints(o: HitObject, span: float, tick_dist: float) -> list[float]:
    cps = []
    min_from_end = 10.0 / span * o.length if span > 0 else 0.0  # тики ближе 10 мс к концу не ставятся
    for i in range(o.slides):
        ticks = []
        d = tick_dist
        while tick_dist > 0 and d < o.length - min_from_end and len(ticks) < 256:
            ticks.append(d)
            d += tick_dist
        base = o.time + i * span
        for d in ticks:
            frac = d / o.length if i % 2 == 0 else 1 - d / o.length
            cps.append(base + frac * span)
        if i < o.slides - 1:
            cps.append(base + span)  # повтор
    # legacy: последний тик на 36 мс раньше конца, но не раньше середины
    cps.append(max(o.time + (o.end_time - o.time) / 2, o.end_time - 36))
    return sorted(cps)
