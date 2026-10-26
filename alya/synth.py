"""Генератор случайных .osu карт (для тестов и стартового обучения без своих карт)."""
from __future__ import annotations

import math
import random
from pathlib import Path


def make_map(seed: int, n_objects: int = 120, bpm: float | None = None,
             cs: float | None = None, ar: float | None = None, od: float | None = None) -> str:
    rng = random.Random(seed)
    bpm = bpm or rng.uniform(110, 210)
    cs = cs if cs is not None else round(rng.uniform(3, 5), 1)
    ar = ar if ar is not None else round(rng.uniform(7, 9.5), 1)
    od = od if od is not None else round(rng.uniform(5, 9), 1)
    beat = 60000 / bpm
    sm = 1.6
    lines = [
        "osu file format v14", "", "[General]", "AudioFilename: audio.mp3", "Mode: 0", "",
        "[Metadata]", f"Title:synthetic {seed}", "Artist:Alya", "Creator:Alya",
        f"Version:bpm{bpm:.0f} cs{cs} ar{ar} od{od}", "",
        "[Difficulty]", "HPDrainRate:5", f"CircleSize:{cs}", f"OverallDifficulty:{od}",
        f"ApproachRate:{ar}", f"SliderMultiplier:{sm}", "SliderTickRate:1", "",
        "[TimingPoints]", f"1000,{beat:.6f},4,2,0,60,1,0", "", "[HitObjects]",
    ]
    t = 1000 + 2 * beat
    x, y = 256.0, 192.0
    ang = rng.uniform(0, 2 * math.pi)

    def step(dist):
        nonlocal x, y, ang
        for _ in range(20):
            ang += rng.uniform(-1.8, 1.8)
            nx, ny = x + dist * math.cos(ang), y + dist * math.sin(ang)
            if 30 <= nx <= 482 and 30 <= ny <= 354:
                x, y = nx, ny
                return
        x, y = rng.uniform(60, 452), rng.uniform(60, 324)

    i = 0
    while i < n_objects:
        r = rng.random()
        nc = 4 if i % 6 == 0 else 0
        if r < 0.08 and i > 4 and i % 40 > 10:
            dur = beat * rng.choice([2, 4])
            lines.append(f"256,192,{int(t)},{8 | 4},0,{int(t + dur)}")
            t += dur + beat
            i += 1
            continue
        if r < 0.38:
            length = rng.choice([1, 1, 2]) * sm * 100 / 2
            slides = rng.choice([1, 1, 1, 2])
            kind = rng.choice("LPB")
            a = rng.uniform(0, 2 * math.pi)
            ex, ey = x + length * math.cos(a), y + length * math.sin(a)
            ex, ey = min(max(ex, 10), 502), min(max(ey, 10), 374)
            if kind == "L":
                pts = f"L|{int(ex)}:{int(ey)}"
            else:
                mx = (x + ex) / 2 + rng.uniform(-40, 40)
                my = (y + ey) / 2 + rng.uniform(-40, 40)
                pts = f"{kind}|{int(mx)}:{int(my)}|{int(ex)}:{int(ey)}"
            lines.append(f"{int(x)},{int(y)},{int(t)},{2 | nc},0,{pts},{slides},{length:.2f}")
            dur = length / (sm * 100) * beat * slides
            if slides % 2 == 1:
                x, y = ex, ey
            t += dur + beat / 2
        else:
            lines.append(f"{int(x)},{int(y)},{int(t)},{1 | nc},0,0:0:0:0:")
            t += beat / rng.choice([1, 2, 2, 2, 4]) if rng.random() < 0.85 else beat
        step(rng.uniform(60, 220))
        i += 1
    return "\n".join(lines) + "\n"


def write_maps(folder: str | Path, count: int, start_seed: int = 0, **kw) -> list[Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for s in range(start_seed, start_seed + count):
        p = folder / f"synthetic_{s}.osu"
        p.write_text(make_map(s, **kw), encoding="utf-8")
        paths.append(p)
    return paths
