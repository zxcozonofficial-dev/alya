"""Живая игра: время песни и карта из tosu (websocket), курсор и клавиши через SendInput.

Только для оффлайна / локальных скоров. Отправка таких скоров на сервер нарушает правила osu!.
F8 — аварийная остановка. Esc — пауза сразу, не дожидаясь tosu: бот отпускает клавиши и
не трогает мышь, пока игра не продолжится (или выход в меню — тогда карта закончена).
"""
from __future__ import annotations

import ctypes
import ctypes.wintypes as wt
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from . import model as M
from .beatmap import SPINNER, parse_osu
from .doodle import BreakDoodler
from .env import DT, next_step
from .humanize import CurveBend
from .keys import KeyAssigner
from .sim import MOVE_SCALE, MapPool, Simulator, clamp_cursor

user32 = ctypes.WinDLL("user32", use_last_error=True)

# ---------------------------------------------------------------- SendInput
INPUT_MOUSE, INPUT_KEYBOARD = 0, 1
MOUSEEVENTF_MOVE, MOUSEEVENTF_ABSOLUTE, MOUSEEVENTF_VIRTUALDESK = 0x1, 0x8000, 0x4000
KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE = 0x2, 0x8
VK_F8, VK_ESCAPE = 0x77, 0x1B
SCANCODES = {c: s for c, s in zip("qwertyuiop", range(0x10, 0x1A))}
SCANCODES.update({c: s for c, s in zip("asdfghjkl", range(0x1E, 0x27))})
SCANCODES.update({c: s for c, s in zip("zxcvbnm", range(0x2C, 0x33))})


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wt.LONG), ("dy", wt.LONG), ("mouseData", wt.DWORD), ("dwFlags", wt.DWORD),
                ("time", wt.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wt.WORD), ("wScan", wt.WORD), ("dwFlags", wt.DWORD), ("time", wt.DWORD),
                ("dwExtraInfo", ctypes.c_size_t)]


class _U(ctypes.Union):
    _fields_ = [("mi", MOUSEINPUT), ("ki", KEYBDINPUT), ("pad", ctypes.c_byte * 32)]


class INPUT(ctypes.Structure):
    _fields_ = [("type", wt.DWORD), ("u", _U)]


def _send(*inputs):
    arr = (INPUT * len(inputs))(*inputs)
    user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


def move_mouse(x: float, y: float):
    vx, vy = user32.GetSystemMetrics(76), user32.GetSystemMetrics(77)
    vw, vh = user32.GetSystemMetrics(78), user32.GetSystemMetrics(79)
    i = INPUT(type=INPUT_MOUSE)
    i.u.mi = MOUSEINPUT(int((x - vx) * 65535 / max(vw - 1, 1)), int((y - vy) * 65535 / max(vh - 1, 1)), 0,
                        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK, 0, 0)
    _send(i)


def key_event(scan: int, up: bool):
    i = INPUT(type=INPUT_KEYBOARD)
    i.u.ki = KEYBDINPUT(0, scan, KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0), 0, 0)
    _send(i)


def find_osu_window():
    found = []

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n and user32.IsWindowVisible(hwnd):
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if buf.value.startswith("osu!") and "tosu" not in buf.value.lower():
                found.append(hwnd)
        return True

    user32.EnumWindows(cb, 0)
    return found[0] if found else None


def playfield_transform(hwnd, client="stable", offset_y=0.0):
    """osu!px -> экранные пиксели.

    stable: поле высотой 0.8 окна, сдвинуто вниз на 8 osu!px.
    lazer: поле вписано (4:3) в 0.8 окна, без сдвига (сдвиг только при выравнивании по сториборду).
    offset_y — ручная поправка в osu!px, если калибровка показала смещение.
    """
    rc = wt.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(rc))
    pt = wt.POINT(0, 0)
    user32.ClientToScreen(hwnd, ctypes.byref(pt))
    w, h = rc.right, rc.bottom
    if client == "lazer":
        s = min(0.8 * w / 512, 0.8 * h / 384)
        shift = 0.0
    else:
        s = h / 480.0
        shift = 8.0
    ox = pt.x + (w - 512 * s) / 2
    oy = pt.y + (h - 384 * s) / 2 + (shift + offset_y) * s
    return lambda x, y: (ox + x * s, oy + y * s)


# ---------------------------------------------------------------- tosu
class Tosu:
    """Состояние игры из tosu (/websocket/v2).

    Время песни tosu присылает раз в POLL_RATE мс (по умолчанию 150), и каждое значение
    приходит с задержкой. Храним смещение «время песни − часы» и берём медиану за последние
    WINDOW_S секунд: она устойчива к редким выбросам. Постоянную задержку убирает автоподстройка по
    ошибкам попаданий (offset_ms).

    В начале карты (~1–1.5 с) и на загрузке время стоит на месте. Если оно повторилось
    три раза подряд (~0.45 с), историю стираем и song_time() возвращает None, пока время снова не пойдёт:
    иначе бот экстраполирует часы от замёрзшего значения и мажет первые ноты.
    Если два значения подряд расходятся с медианой больше чем на JUMP_MS — время прыгнуло
    (старт трека, ретрай, перемотка), историю начинаем заново. Так же — при переходе времени
    через 0: отсчёт до начала музыки идёт по другим часам, со сдвигом ~35–40 мс.
    """

    JUMP_MS = 40.0
    WINDOW_S = 3.0

    def __init__(self, host="127.0.0.1", port=24050):
        self.url = f"ws://{host}:{port}/websocket/v2"
        self.state = None
        self.paused = False
        self.client = "stable"
        self.map_path = None
        self.rate = 1.0
        self.mods = 0
        self.logged_in = None
        self.connected = False
        self.offset_ms = 0.0  # ручная поправка: + = модель действует раньше
        self.hits = {}        # счёт самой игры (play.hits) на последнем кадре в состоянии play
        self.hit_errors = []  # ошибки попаданий из игры, мс (+ = поздно)
        self.time_log = []    # (perf_counter, время песни от tosu) — для разбора таймингов
        self.hits_log = []    # (время песни, miss, 100, 50) — когда игра засчитала промахи
        self.manual_offset = 0.0
        self.auto_offset = False
        self._offsets = []  # (perf_counter, offset)
        self._last_live = None
        self._jumps = 0
        self._frozen = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        from websockets.sync.client import connect
        warned = False
        while True:
            try:
                with connect(self.url, open_timeout=3) as ws:
                    self.connected = True
                    for msg in ws:
                        self._on_v2(json.loads(msg))
            except Exception as e:
                self.connected = False
                if not warned:
                    print(f"[tosu] нет соединения с {self.url} ({e.__class__.__name__}) — tosu запущен?")
                    warned = True
                time.sleep(2)

    def _on_v2(self, d):
        self.state = (d.get("state") or {}).get("number")
        self.paused = bool((d.get("game") or {}).get("paused"))
        self.client = str(d.get("client") or "stable").lower()
        folders = d.get("folders") or {}
        path = (d.get("directPath") or {}).get("beatmapFile") or ""
        if path and not Path(path).is_absolute():  # путь относительно папки Songs
            path = str(Path(folders.get("songs") or "") / path)
        self.map_path = path or None
        mods = (d.get("play") or {}).get("mods") or {}
        self.rate = float(mods.get("rate") or 1.0)
        self.mods = int(mods.get("number") or 0)
        status = (d.get("profile") or {}).get("userStatus") or {}
        name = str(status.get("name", "")).lower()
        self.logged_in = True if name == "connected" else False if name in ("guest", "disconnected") else None
        live = ((d.get("beatmap") or {}).get("time") or {}).get("live")
        if self.state == 2:
            play = d.get("play") or {}
            self.hits = dict(play.get("hits") or {})
            self.hit_errors = list(play.get("hitErrorArray") or [])
            if live is not None:
                self.time_log.append((time.perf_counter(), float(live)))
                h = self.hits
                self.hits_log.append((float(live), h.get("0", 0), h.get("100", 0), h.get("50", 0)))
        if live is not None:
            live = float(live)
            now = time.perf_counter()
            off = live - now * 1000 * self.rate
            with self._lock:
                frozen = live == self._last_live  # время стоит: старт трека, загрузка, пауза
                if self._last_live is not None and self._last_live < 0 <= live:
                    self._offsets.clear()  # началась музыка
                self._last_live = live
                if frozen:
                    self._frozen += 1
                    if self._frozen >= 3:
                        self._offsets.clear()
                    return
                self._frozen = 0
                if self._offsets:
                    med = float(np.median([o for _, o in self._offsets]))
                    if abs(off - med) > self.JUMP_MS:
                        self._jumps += 1
                        if self._jumps < 2:
                            return  # одиночный выброс (задержка доставки) — пропускаем
                        self._offsets.clear()
                    self._jumps = 0
                self._offsets.append((now, off))
                self._offsets = [(t, o) for t, o in self._offsets if now - t < self.WINDOW_S]

    def song_time(self):
        with self._lock:
            if not self._offsets:
                return None
            off = float(np.median([o for _, o in self._offsets]))
        return time.perf_counter() * 1000 * self.rate + off + self.offset_ms * self.rate

    def sim_time(self):
        """Реальные мс (как в симуляторе): время песни / скорость DT/HT."""
        t = self.song_time()
        return None if t is None else t / self.rate


class DryInput:
    """Вместо SendInput — только подсчёт действий (проверка без управления мышью)."""

    def __init__(self):
        self.moves = 0
        self.presses = 0

    def move(self, x, y):
        self.moves += 1

    def key(self, scan, up):
        self.presses += not up


class RealInput:
    def move(self, x, y):
        move_mouse(x, y)

    def key(self, scan, up):
        key_event(scan, up)


OFFSET_FILE = Path("ckpt/live_offset.json")


def _load_auto_offset(client):
    try:
        return float(json.loads(OFFSET_FILE.read_text(encoding="utf-8")).get(client, 0.0))
    except Exception:
        return 0.0


def _save_auto_offset(client, value):
    try:
        d = json.loads(OFFSET_FILE.read_text(encoding="utf-8")) if OFFSET_FILE.exists() else {}
    except Exception:
        d = {}
    d[client] = round(value, 1)
    OFFSET_FILE.write_text(json.dumps(d), encoding="utf-8")


def _hires_timer():
    try:
        ctypes.WinDLL("winmm").timeBeginPeriod(1)  # sleep с точностью ~1 мс вместо 15
    except Exception:
        pass


def _wait_until(tosu, target):
    """Ждать момента target. False — ждать нечего: игра закончилась или время прыгнуло назад
    (в начале карты tosu ещё какое-то время отдаёт время превью из меню выбора)."""
    while True:
        now = tosu.sim_time()
        if now is None or tosu.state != 2 or now < target - 500:
            return False
        if now >= target:
            return True
        if target - now > 3:
            time.sleep(0.001)


def _next_step(t, note_t):
    """Следующий шаг бота: обычно через DT, но если нота через 0.5–1.5 DT — ровно на неё
    (так же шагает и обучение): у модели есть шаг точно в момент ноты, длина шагов 8–24 мс."""
    return next_step(t, note_t)


class _OwnErrors:
    """Ошибки попаданий по часам самого бота: (время нажатия − время ноты) для нажатий по нотам.

    Ошибка в игре = ошибка самой модели + расхождение часов бота с игрой. Автоподстройка
    должна убирать только второе: поздние нажатия модели на стримах — не повод сдвигать часы.
    """

    def __init__(self, bm):
        objs = [o for o in bm.objects if o.kind != SPINNER]
        self.t = np.array([o.time for o in objs])
        self.xy = np.array([(o.x, o.y) for o in objs], np.float32).reshape(-1, 2)
        self.r, self.w50 = bm.radius, bm.windows[2]
        self.ptr = 0
        self.errors = []

    def press(self, t, x, y):
        while self.ptr < len(self.t) and self.t[self.ptr] < t - self.w50:
            self.ptr += 1
        j = self.ptr
        while j < len(self.t) and self.t[j] <= t + self.w50:
            if np.hypot(*(self.xy[j] - (x, y))) <= self.r:
                self.errors.append(t - self.t[j])
                self.ptr = j + 1
                return
            j += 1


# ---------------------------------------------------------------- игра
def run_live(model_path: str, keys=("z", "x"), device="cpu", port=24050, deterministic=True,
             acc=0.97, miss=0.005, dry_run=False, offset_y=0.0, once=False, offset_ms=0.0,
             auto_offset=True, smoke_key="c", curve=0.2):
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    _hires_timer()
    net = M.load(model_path, device).eval()
    tosu = Tosu(port=port)
    tosu.manual_offset = offset_ms
    tosu.offset_ms = offset_ms
    tosu.auto_offset = auto_offset and not dry_run
    scans = [SCANCODES[k] for k in keys]
    smoke = SCANCODES[smoke_key] if smoke_key else None
    inp = DryInput() if dry_run else RealInput()
    cond = M.skill_features(acc, miss)
    print("Alya live: жду начала карты в osu! (F8 — стоп)" + (" [dry-run: без ввода]" if dry_run else ""))
    warned_online = False
    while True:
        if user32.GetAsyncKeyState(VK_F8) & 0x8000:
            return inp
        if tosu.state == 2 and tosu.map_path and tosu.song_time() is not None:
            if tosu.logged_in and not dry_run:
                if not warned_online:
                    print("Ты залогинен в osu! — скор бота уйдёт на сервер (это бан и нечестно к другим игрокам).\n"
                          "Выйди из аккаунта (играй оффлайн), тогда Alya начнёт играть.")
                    warned_online = True
                time.sleep(0.5)
                continue
            auto = _load_auto_offset(tosu.client) if tosu.auto_offset else 0.0
            tosu.offset_ms = tosu.manual_offset + auto
            if auto:
                print(f"автоподстройка тайминга: {auto:+.1f} мс (из прошлых игр)")
            _play_one(net, tosu, scans, device, deterministic, cond, inp, offset_y, smoke, curve)
            if tosu.auto_offset:
                _save_auto_offset(tosu.client, tosu.offset_ms - tosu.manual_offset)
            if once:
                return inp
        time.sleep(0.05)


def _play_one(net, tosu, scans, device, deterministic, cond, inp, offset_y, smoke_scan=None, curve=0.0):
    path, mods = tosu.map_path, tosu.mods
    try:
        bm = parse_osu(path, mods)
    except Exception as e:
        print(f"не смог прочитать карту {path}: {e}")
        time.sleep(1)
        return
    hwnd = find_osu_window()
    if not hwnd and not isinstance(inp, DryInput):
        print("окно osu! не найдено")
        time.sleep(1)
        return
    to_screen = playfield_transform(hwnd, tosu.client, offset_y) if hwnd else (lambda x, y: (x, y))
    print(f"играю: {bm} [{tosu.client}, rate {bm.rate}]")
    sim = Simulator(MapPool([bm]), 1)
    pos = np.array([[256.0, 192.0]], np.float32)
    sim.reset([0], 0, 0, tosu.sim_time(), pos[0])
    tosu.time_log = []
    tosu.hits_log = []
    frames = []  # (время сима, x, y, K1, K2, smoke) — для сохранения реплея живой игры
    used_err = 0  # сколько ошибок попаданий уже учтено автоподстройкой
    own = _OwnErrors(bm)
    used_own = 0
    h = net.init_hidden(1, device)
    held = np.zeros(2, bool)
    keys = KeyAssigner(1)
    doodler = BreakDoodler(bm) if smoke_scan is not None else None
    bend = CurveBend(bm, curve)
    note_t = own.t  # времена голов нот (без спиннеров), по возрастанию
    smoking = False
    lag = []
    esc_down = False
    esc_pause = None  # (perf_counter нажатия Esc, видели ли паузу в tosu) — пока на паузе по Esc

    def release():
        nonlocal smoking
        for c in range(2):
            if held[c]:
                inp.key(scans[c], True)
                held[c] = False
        if smoking:
            inp.key(smoke_scan, True)
            smoking = False

    try:
        while tosu.state == 2 and tosu.map_path == path and tosu.mods == mods:
            if user32.GetAsyncKeyState(VK_F8) & 0x8000:
                print("F8: стоп")
                break
            errs = tosu.hit_errors
            if tosu.auto_offset and len(errs) >= used_err + 8:
                # + = в игре жмём позже, чем собирались по своим часам -> действовать раньше
                game = np.median(errs[used_err:]) / bm.rate
                mine = np.median(own.errors[used_own:]) if len(own.errors) > used_own else 0.0
                shift = float(np.clip(0.5 * (game - mine), -10, 10))
                auto = float(np.clip(tosu.offset_ms - tosu.manual_offset + shift, -50, 50))
                tosu.offset_ms = tosu.manual_offset + auto
                used_err, used_own = len(errs), len(own.errors)
            elif len(errs) < used_err:  # ретрай — массив ошибок обнулился
                used_err, used_own = 0, len(own.errors)
            esc = bool(user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000)
            if esc and not esc_down and esc_pause is None:
                release()  # сразу, tosu сообщит о паузе только через ~0.2–1 с
                esc_pause = [time.perf_counter(), False]
            esc_down = esc
            if esc_pause is not None:
                esc_pause[1] |= tosu.paused
                # продолжаем, когда игра снова идёт: пауза была и кончилась, или Esc не поставил
                # паузу вовсе (за 1.5 с tosu её не показал)
                if tosu.paused or not (esc_pause[1] or time.perf_counter() - esc_pause[0] > 1.5):
                    time.sleep(0.005)
                    continue
                esc_pause = None
            if tosu.paused:
                release()
                time.sleep(0.05)
                continue
            now = tosu.sim_time()
            if now is None:  # время ещё не пошло (после паузы, на старте)
                time.sleep(0.005)
                continue
            if now < sim.t[0] - 500:  # ретрай / перемотка назад
                print("рестарт карты")
                release()
                sim.reset([0], 0, 0, now, pos[0])
                h = net.init_hidden(1, device)
                keys = KeyAssigner(1)
                doodler = BreakDoodler(bm) if smoke_scan is not None else None
                own = _OwnErrors(bm)
                used_own = 0
                frames = []
            if sim.finished()[0]:
                # последнее нажатие не отпускаем сразу: иначе клавиша успевает подняться
                # в ту же миллисекунду, osu! его не видит и ставит мисс на последнюю ноту
                if held.any() and frames and now < frames[-1][0] + 60:
                    time.sleep(0.005)
                    continue
                release()
                time.sleep(0.05)
                continue
            obs = torch.as_tensor(M.with_cond(sim.obs(), cond), device=device)
            move, kp, _, _, h = net.act(obs, h, deterministic)
            target_t = _next_step(sim.t[0], note_t)
            pos = clamp_cursor(pos + np.clip(move.cpu().numpy(), -5, 5) * MOVE_SCALE)
            want = kp.cpu().numpy() > 0.5
            smoke = False
            shown = bend(target_t, pos[0])  # на экран — с изгибом между нотами, модели — без
            if doodler is not None:  # перерыв — рисуем дымом вместо модели
                p, smoke = doodler.step(target_t, pos[0])
                if p is not None:
                    pos[0], want[:] = p, False
                    shown = pos[0]
            press = keys(target_t, want)[0]
            if not _wait_until(tosu, target_t):  # действие относится к моменту t+16 мс, как в обучении
                continue  # проверки рестарта/конца в начале цикла
            inp.move(*to_screen(*shown))
            for c in range(2):
                if press[c] != held[c]:
                    inp.key(scans[c], not press[c])
            if (press & ~held).any():
                own.press(target_t, pos[0, 0], pos[0, 1])
            held = press.copy()
            if smoke != smoking:
                inp.key(smoke_scan, not smoke)
                smoking = smoke
            now = tosu.sim_time()
            if now is None:  # время замёрзло прямо сейчас (пауза) — считаем, что успели вовремя
                now = target_t
            lag.append(now - target_t)
            # отстали больше чем на шаг (лаг) — догоняем реальное время
            t_adv = max(target_t, now - DT)
            sim.advance(t_adv, pos, press[None])
            frames.append((t_adv, shown[0], shown[1], press[0], press[1], smoke))
    finally:
        release()
    _save_live(bm, sim, frames, tosu)
    lag = np.array(lag) if lag else np.zeros(1)
    if own.errors:
        e = np.array(own.errors)
        print(f"тайминг модели по своим часам: среднее {e.mean():+.1f} мс, разброс {e.std():.1f} мс (реальные)")
    print(f"конец карты: 300/100/50/miss по симулятору {sim.counts[0].tolist()} | "
          f"опоздание шага: медиана {np.median(lag):.1f} мс, 99% {np.quantile(lag, .99):.1f} мс")
    return sim


def _save_live(bm, sim, frames, tosu):
    """Сохранить реплей живой игры и сравнить счёт симулятора со счётом игры."""
    from .export import to_replay
    if len(frames) < 50:
        return
    f = np.array(frames, np.float64)
    out = Path("replays/live")
    out.mkdir(parents=True, exist_ok=True)
    stem = time.strftime("%m%d_%H%M%S_") + "".join(c for c in bm.title if c.isalnum() or c in " -_")[:40]
    r = to_replay(bm, f[:, 0], f[:, 1:3].astype(np.float32), f[:, 3:5] > 0.5, sim, player="Alya live",
                  smoke=f[:, 5] > 0.5)
    from .osr import write_osr
    write_osr(r, out / f"{stem}.osr")
    np.savez_compressed(out / f"{stem}_time.npz", tosu=np.array(tosu.time_log or [(0, 0)]), frames=f,
                        rate=bm.rate, hits=np.array(tosu.hits_log or [(0, 0, 0, 0)]),
                        errors=np.array(tosu.hit_errors or [0.0]))
    g = tosu.hits
    if g:
        print(f"счёт игры: 300 {g.get('300')} | 100 {g.get('100')} | 50 {g.get('50')} | miss {g.get('0')} | "
              f"срывов слайдеров {g.get('sliderBreaks')}")
        e = np.array(tosu.hit_errors or [0.0], np.float64)
        print(f"ошибка попаданий в игре: средняя {e.mean():+.1f} мс, разброс {e.std():.1f} мс; "
              f"автоподстройка сейчас {tosu.offset_ms - tosu.manual_offset:+.1f} мс")
        _report_misses(bm, tosu.hits_log)
    print(f"реплей живой игры: {out / (stem + '.osr')}")


def _report_misses(bm, hits_log):
    """По моменту, когда в игре вырос счётчик промахов, найти, на каких нотах они были."""
    if len(hits_log) < 2:
        return
    h = np.array(hits_log, np.float64)
    inc = np.nonzero(np.diff(h[:, 1]) > 0)[0] + 1
    if not len(inc):
        return
    times = np.array([o.time for o in bm.objects]) * bm.rate   # время карты
    w50 = bm.windows[2] * bm.rate
    found = []
    for i in inc:
        t_prev, t_now = h[i - 1, 0], h[i, 0]
        # промах засчитывается через ~w50 после ноты (или раньше — ранним кликом)
        cand = np.nonzero((times + w50 >= t_prev - 200) & (times - 400 <= t_now))[0]
        found.append(int(cand[-1]) if len(cand) else -1)
    n = len(bm.objects)
    desc = ", ".join(f"#{j + 1}" + (" (конец)" if j >= n - 10 else "") for j in found)
    print(f"промахи в игре (из {n} нот): {desc}")


def calibrate(port=24050, offset_y=0.0):
    """Обвести курсором углы игрового поля, чтобы сверить с рамкой поля в игре."""
    try:
        user32.SetProcessDPIAware()
    except Exception:
        pass
    tosu = Tosu(port=port)
    time.sleep(1.5)
    hwnd = find_osu_window()
    if not hwnd:
        print("окно osu! не найдено")
        return
    to_screen = playfield_transform(hwnd, tosu.client, offset_y)
    print(f"клиент: {tosu.client}. Курсор обходит углы поля 0,0 → 512,0 → 512,384 → 0,384 и центр (F8 — стоп)")
    pts = [(0, 0), (512, 0), (512, 384), (0, 384), (256, 192)]
    for _ in range(3):
        for x, y in pts:
            if user32.GetAsyncKeyState(VK_F8) & 0x8000:
                return
            move_mouse(*to_screen(x, y))
            time.sleep(1.0)
