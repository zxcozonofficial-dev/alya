"""Чтение и запись реплеев .osr (osu! stable). Кадры сжаты LZMA (формат alone)."""
from __future__ import annotations

import hashlib
import io
import lzma
import struct
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# биты клавиш в кадре
M1, M2, K1, K2, SMOKE = 1, 2, 4, 8, 16
# моды, при которых реплей не годится для BC: RX, Auto, SpunOut, Autopilot, Target Practice
MODS_UNSUPPORTED = 128 | 2048 | 4096 | 8192 | 8388608


@dataclass
class Replay:
    mode: int = 0
    version: int = 20240101
    beatmap_md5: str = ""
    player: str = "Alya"
    replay_md5: str = ""
    n300: int = 0
    n100: int = 0
    n50: int = 0
    ngeki: int = 0
    nkatu: int = 0
    nmiss: int = 0
    score: int = 0
    max_combo: int = 0
    perfect: bool = False
    mods: int = 0
    life_bar: str = ""
    timestamp: int = 0
    online_id: int = 0
    seed: int = 0
    # кадры: абсолютное время (мс), x, y, клавиши
    frames: np.ndarray = field(default_factory=lambda: np.zeros((0, 4)))


def _read_uleb(f):
    out = shift = 0
    while True:
        b = f.read(1)[0]
        out |= (b & 0x7F) << shift
        if not b & 0x80:
            return out
        shift += 7


def _read_str(f):
    if f.read(1)[0] == 0:
        return ""
    return f.read(_read_uleb(f)).decode("utf-8", errors="replace")


def _write_str(f, s: str):
    if not s:
        f.write(b"\x00")
        return
    b = s.encode("utf-8")
    f.write(b"\x0b")
    n = len(b)
    while True:
        byte = n & 0x7F
        n >>= 7
        f.write(bytes([byte | (0x80 if n else 0)]))
        if not n:
            break
    f.write(b)


def read_osr(path) -> Replay:
    with open(path, "rb") as f:
        r = Replay()
        r.mode = f.read(1)[0]
        r.version = struct.unpack("<i", f.read(4))[0]
        r.beatmap_md5 = _read_str(f)
        r.player = _read_str(f)
        r.replay_md5 = _read_str(f)
        (r.n300, r.n100, r.n50, r.ngeki, r.nkatu, r.nmiss) = struct.unpack("<6H", f.read(12))
        r.score = struct.unpack("<i", f.read(4))[0]
        r.max_combo = struct.unpack("<H", f.read(2))[0]
        r.perfect = bool(f.read(1)[0])
        r.mods = struct.unpack("<i", f.read(4))[0]
        r.life_bar = _read_str(f)
        r.timestamp = struct.unpack("<q", f.read(8))[0]
        n = struct.unpack("<i", f.read(4))[0]
        data = lzma.decompress(f.read(n), format=lzma.FORMAT_ALONE).decode("ascii")
        rest = f.read(8)
        if len(rest) == 8:
            r.online_id = struct.unpack("<q", rest)[0]
    frames, t = [], 0.0
    for part in data.split(","):
        if not part:
            continue
        w, x, y, k = part.split("|")
        w = float(w)
        if w == -12345:
            r.seed = int(k)
            continue
        t += w
        frames.append((t, float(x), float(y), int(k)))
    r.frames = np.array(frames, dtype=np.float64).reshape(-1, 4)
    # первые два служебных кадра stable (y=-500)
    while len(r.frames) and r.frames[0, 2] <= -500:
        r.frames = r.frames[1:]
    return r


def write_osr(r: Replay, path):
    parts = ["0|256|-500|0", "-1|256|-500|0"]
    t_prev = -1.0
    for t, x, y, k in r.frames:
        w = int(round(t)) - int(round(t_prev))
        t_prev = t
        parts.append(f"{w}|{x:.4f}|{y:.4f}|{int(k)}")
    parts.append(f"-12345|0|0|{r.seed}")
    data = lzma.compress((",".join(parts) + ",").encode("ascii"), format=lzma.FORMAT_ALONE)
    if not r.timestamp:  # .NET ticks
        r.timestamp = int((time.time() + 62135596800) * 10_000_000)
    if not r.replay_md5:
        r.replay_md5 = hashlib.md5(f"{r.max_combo}osu{r.player}{r.beatmap_md5}{r.score}{r.mods}{r.timestamp}".encode()).hexdigest()
    f = io.BytesIO()
    f.write(bytes([r.mode]))
    f.write(struct.pack("<i", r.version))
    _write_str(f, r.beatmap_md5)
    _write_str(f, r.player)
    _write_str(f, r.replay_md5)
    f.write(struct.pack("<6H", r.n300, r.n100, r.n50, r.ngeki, r.nkatu, r.nmiss))
    f.write(struct.pack("<i", r.score))
    f.write(struct.pack("<H", r.max_combo))
    f.write(bytes([int(r.perfect)]))
    f.write(struct.pack("<i", r.mods))
    _write_str(f, r.life_bar)
    f.write(struct.pack("<q", r.timestamp))
    f.write(struct.pack("<i", len(data)))
    f.write(data)
    f.write(struct.pack("<q", r.online_id))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out:
        out.write(f.getvalue())


def resample(frames: np.ndarray, times: np.ndarray, dt: float = 16.0):
    """Кадры реплея -> сетка times.

    Позиция — линейная интерполяция. Клавиши [T,2] (K1/M1, K2/M2): нажата, если была
    зажата хоть раз в окне (t - dt/2, t + dt/2] — квантование по центру шага без запаздывания.
    """
    ft, fx, fy = frames[:, 0], frames[:, 1], frames[:, 2]
    bits = frames[:, 3].astype(np.int64)
    pos = np.stack([np.interp(times, ft, fx), np.interp(times, ft, fy)], 1).astype(np.float32)
    lo_t, hi_t = times - dt / 2, times + dt / 2
    idx = np.searchsorted(ft, hi_t, side="right") - 1
    lo = np.searchsorted(ft, lo_t, side="right")
    hi = idx + 1
    key = np.zeros((len(times), 2), bool)
    click_t = times.astype(np.float64).copy()
    first = np.full(len(times), np.inf)
    for c, mask in enumerate((M1 | K1, M2 | K2)):
        pressed = (bits & mask) != 0
        cum = np.concatenate([[0], np.cumsum(pressed)])
        key[:, c] = np.where(idx >= 0, pressed[np.clip(idx, 0, None)], False)
        key[:, c] |= (cum[hi] - cum[lo]) > 0
        # точное время первого нажатия этой клавиши в окне шага
        rise = np.nonzero(pressed & ~np.concatenate([[False], pressed[:-1]]))[0]
        j = np.searchsorted(ft[rise], lo_t, side="right")
        ok = j < len(rise)
        rt = np.where(ok, ft[rise[np.minimum(j, len(rise) - 1)]] if len(rise) else np.inf, np.inf)
        first = np.minimum(first, np.where(rt <= hi_t, rt, np.inf))
    has = np.isfinite(first)
    click_t[has] = first[has]
    click_pos = np.stack([np.interp(click_t, ft, fx), np.interp(click_t, ft, fy)], 1).astype(np.float32)
    return pos, key, click_t, click_pos
