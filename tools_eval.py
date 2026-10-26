"""Сравнение моделей на одних и тех же картах: точность, промахи, чередование клавиш, слайдеры, резкие углы."""
import sys, glob, os, json, random, ctypes
import numpy as np
sys.path.insert(0, '.')
ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
from alya.osr import read_osr, resample, MODS_UNSUPPORTED
from alya.bc import grid
from alya.beatmap import parse_osu, mods_to_string
from alya.sim import Simulator, MapPool
from alya import model as M
from alya.export import play_map
from alya.live import _OwnErrors

O = os.environ['LOCALAPPDATA'] + '/osu!'
idx = json.load(open(glob.glob('data/md5_*.json')[0], encoding='utf-8'))
models = {name: M.load(path, 'cuda').eval() for name, path in (a.split('=') for a in sys.argv[1:])}


def taps(times, key):
    out = []
    for c in range(2):
        r = np.nonzero(key[1:, c] & ~key[:-1, c])[0] + 1
        out += [(times[i], c) for i in r]
    return sorted(out)


def alt_rate(tp, max_gap=160):
    pairs = [(c0 != c1) for (t0, c0), (t1, c1) in zip(tp, tp[1:]) if t1 - t0 < max_gap]
    return (sum(pairs) / len(pairs)) if pairs else float('nan')


def corners(pos):
    """Резкие развороты на скорости (>90°, обе скорости >6 px/шаг) на 1000 шагов."""
    v = np.diff(pos, axis=0)
    sp = np.linalg.norm(v, axis=1)
    cos = (v[1:] * v[:-1]).sum(1) / (sp[1:] * sp[:-1] + 1e-6)
    return ((cos < 0) & (np.minimum(sp[1:], sp[:-1]) > 6)).mean() * 1000


def timing(bm, times, pos, key):
    """Ошибки нажатий по нотам (мс карты, как на UR-баре): доля в окне 300 и UR."""
    o = _OwnErrors(bm)
    rise = np.nonzero((key[1:] & ~key[:-1]).any(1))[0] + 1
    for i in rise:
        o.press(times[i], pos[i, 0], pos[i, 1])
    e = np.array(o.errors) if o.errors else np.zeros(1)
    return np.mean(np.abs(e) <= bm.windows[0]), np.std(e * bm.rate) * 10


def cp_misses(bm, times, pos, key):
    sim = Simulator(MapPool([bm]), 1)
    sim.reset([0], 0, 0, times[0], pos[0])
    miss = 0
    for k in range(1, len(times)):
        miss += sim.advance(times[k], pos[k][None], key[k:k + 1])['cp_miss'][0]
    return miss


reps = sorted(glob.glob(O + '/Replays/*.osr'))
random.Random(1).shuffle(reps)
seen, rows = set(), []
for rp in reps:
    try:
        r = read_osr(rp)
    except Exception:
        continue
    if r.mode or r.mods & MODS_UNSUPPORTED or r.beatmap_md5 not in idx or r.beatmap_md5 in seen:
        continue
    seen.add(r.beatmap_md5)
    bm = parse_osu(idx[r.beatmap_md5], r.mods)
    tot = r.n300 + r.n100 + r.n50 + r.nmiss
    if tot < len(bm.objects) * 0.9:
        continue
    pacc = (300 * r.n300 + 100 * r.n100 + 50 * r.n50) / (300 * tot)
    fr = r.frames.copy()
    fr[:, 0] /= bm.rate
    hp = resample(fr, grid(bm))[0]
    line = f'{mods_to_string(r.mods):6s} {bm.title[:24]:24s} | ты {pacc*100:5.1f}% ({r.nmiss:3d}) угл.{corners(hp):4.1f}'
    for name, net in models.items():
        times, pos, key, sim, _ = play_map(bm, net, True, 'cuda', pacc, r.nmiss / tot)
        acc = sim.accuracy()[0]
        alt = alt_rate(taps(times, key))
        cpm = cp_misses(bm, times, pos, key)
        crn = corners(pos)
        in300, ur = timing(bm, times, pos, key)
        rows.append((name, acc, sim.counts[0, 3], alt, cpm, crn, in300, ur))
        line += f' | {name} {acc*100:5.1f}% ({sim.counts[0, 3]:3d}) чер.{alt*100:3.0f}% сорв.{cpm:3d} угл.{crn:4.1f} UR {ur:4.0f}'
    print(line, flush=True)
    if len(seen) >= 10 and sum(1 for _ in rows) >= 10 * len(models):
        break
print('--- среднее')
for name in models:
    a = np.array([x[1:] for x in rows if x[0] == name], float)
    print(f'{name}: точность {a[:, 0].mean()*100:.1f}%, промахов {a[:, 1].sum():.0f}, '
          f'чередование {np.nanmean(a[:, 2])*100:.0f}%, сорванных тиков {a[:, 3].sum():.0f}, '
          f'резких углов {a[:, 4].mean():.2f}‰, в окне 300 {a[:, 5].mean()*100:.0f}%, UR {a[:, 6].mean():.0f}')
