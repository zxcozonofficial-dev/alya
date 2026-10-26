"""Загрузка карт: папка Songs, отдельные .osu или синтетика."""
from __future__ import annotations

import json
import random
from pathlib import Path

from .beatmap import parse_osu


def find_osu(paths: list[str]) -> list[Path]:
    out = []
    for p in paths:
        p = Path(p)
        out.extend(sorted(p.rglob("*.osu")) if p.is_dir() else [p])
    return out


def load_maps(paths: list[str], limit: int | None = None, max_objects: int = 3000,
              min_objects: int = 50, seed: int = 0, verbose=True, mods_choices=(0,)):
    """mods_choices — моды, из которых для каждой карты выбирается случайный (аугментация)."""
    files = find_osu(paths)
    rng = random.Random(seed)
    rng.shuffle(files)
    maps = []
    for f in files:
        if limit and len(maps) >= limit:
            break
        try:
            bm = parse_osu(f, rng.choice(mods_choices))
        except Exception as e:  # битые карты не должны валить обучение
            if verbose:
                print(f"  пропуск {f.name}: {e}")
            continue
        if bm.mode != 0 or not (min_objects <= len(bm.objects) <= max_objects):
            continue
        maps.append(bm)
    if verbose:
        print(f"загружено карт: {len(maps)} (из {len(files)} файлов)")
    return maps


def md5_index(songs: str, cache: str | None = None) -> dict[str, str]:
    """md5 .osu -> путь (кэшируется в json)."""
    import hashlib
    if cache and Path(cache).exists():
        return json.loads(Path(cache).read_text(encoding="utf-8"))
    idx = {}
    for f in find_osu([songs]):
        idx[hashlib.md5(f.read_bytes()).hexdigest()] = str(f)
    if cache:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        Path(cache).write_text(json.dumps(idx), encoding="utf-8")
    return idx
