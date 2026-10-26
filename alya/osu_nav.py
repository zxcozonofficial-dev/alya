"""Управление меню osu! stable: фокус окна, нажатия клавиш, переход к карте по ссылке osu://b/<id>.

Состояние игры берём из tosu (/json/v2): 2 — игра, 5 — выбор песни, 7 — результаты, 0 — главное меню.
"""
from __future__ import annotations

import ctypes
import json
import os
import time
import urllib.request

from .live import find_osu_window, key_event

TOSU_JSON = "http://127.0.0.1:24050/json/v2"
SC_ENTER, SC_ESC = 0x1C, 0x01
PLAYING, SONG_SELECT, RESULTS, MENU = 2, 5, 7, 0

user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32


def tosu_state() -> dict:
    """Текущее состояние: номер экрана, id карты и сета, на паузе ли."""
    with urllib.request.urlopen(TOSU_JSON, timeout=3) as r:
        d = json.load(r)
    b = d.get("beatmap") or {}
    t = b.get("time") or {}
    return {"state": (d.get("state") or {}).get("number"), "id": b.get("id"), "set": b.get("set"),
            "title": b.get("title"), "artist": b.get("artist"), "version": b.get("version"),
            "mapper": b.get("mapper"),
            "stars": float(((b.get("stats") or {}).get("stars") or {}).get("total") or 0),  # с выбранными модами
            "length": int(t.get("lastObject") or 0) - int(t.get("firstObject") or 0), "focused": (d.get("game") or {}).get("focused"),
            "paused": bool((d.get("game") or {}).get("paused")),
            "failed": bool((d.get("play") or {}).get("failed")),
            "mods": int(((d.get("play") or {}).get("mods") or {}).get("number") or 0)}  # в выборе песни — выбранные


def focus_osu() -> bool:
    """Вывести окно osu! на передний план. Windows не даёт фоновому процессу просто так забрать фокус,
    поэтому на время цепляемся к потоку текущего активного окна."""
    hwnd = find_osu_window()
    if not hwnd:
        return False
    fg = user32.GetForegroundWindow()
    if fg == hwnd:
        return True
    me = kernel32.GetCurrentThreadId()
    other = user32.GetWindowThreadProcessId(fg, None)
    user32.AttachThreadInput(me, other, True)
    try:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
    finally:
        user32.AttachThreadInput(me, other, False)
    time.sleep(0.15)
    return user32.GetForegroundWindow() == hwnd


def press(scan: int):
    _wait_modifiers_up()
    if not focus_osu():
        raise RuntimeError("не получилось вывести окно osu! на передний план")
    key_event(scan, False)
    time.sleep(0.05)
    key_event(scan, True)


SC_BACKSPACE, SC_DOWN, SC_UP, SC_P = 0x0E, 0x50, 0x48, 0x19
DIGITS = {str(d): 0x02 + (d - 1) % 10 for d in range(1, 11)}  # 1..9 → 0x02..0x0A, 0 → 0x0B
DIGITS["0"] = 0x0B


def _wait_modifiers_up(timeout: float = 3.0):
    """Ждать, пока отпущены Ctrl/Shift/Alt: Ctrl+Enter в osu! запускает карту с автоплеем (мод AT)."""
    end = time.monotonic() + timeout
    while time.monotonic() < end and any(user32.GetAsyncKeyState(vk) & 0x8000 for vk in (0x10, 0x11, 0x12)):
        time.sleep(0.05)


def tap(scan: int, extended: bool = False):
    """Нажать клавишу в osu!, окно уже на переднем плане. Стрелки — extended, иначе это цифры numpad."""
    _wait_modifiers_up()
    from .live import INPUT, INPUT_KEYBOARD, KEYBDINPUT, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, _send
    flags = KEYEVENTF_SCANCODE | (0x0001 if extended else 0)  # KEYEVENTF_EXTENDEDKEY
    for up in (False, True):
        i = INPUT(type=INPUT_KEYBOARD)
        i.u.ki = KEYBDINPUT(0, scan, flags | (KEYEVENTF_KEYUP if up else 0), 0, 0)
        _send(i)
        time.sleep(0.03)


SC_F2 = 0x3C
_typed = 0  # сколько символов мы сами напечатали в поиске (чтобы стереть ровно их)


def hotkey(vk: int, shift: bool = True, only_osu: bool = False) -> bool:
    """Нажато ли Ctrl(+Shift)+<клавиша>. Не F-клавиши: F10 в Windows работает как Alt, F9 в osu! — панель игроков.
    only_osu — считать только когда активно окно osu! (Ctrl+A в других программах — «выделить всё»)."""
    st = user32.GetAsyncKeyState
    down = st(0x11) & 0x8000 and st(vk) & 0x8000 and (not shift or st(0x10) & 0x8000)
    if down and only_osu:
        return user32.GetForegroundWindow() == find_osu_window()
    return bool(down)


def roll_random(lo: float, hi: float, skip_ids=(), max_length_ms: int = 5 * 60 * 1000, rolls: int = 25) -> dict | None:
    """Свой выбор карты: F2 (случайная карта в выборе песни), потом стрелками вниз по сложностям сета
    (звёзды растут) до первой в диапазоне [lo, hi] с выбранными модами. Нет такой в сете — снова F2."""
    if not focus_osu():
        return None
    if _typed:  # F2 выбирает только среди найденного поиском
        reset_search()
    sets = set()
    for n in range(rolls):
        tap(SC_F2)
        time.sleep(0.45)
        s = tosu_state()
        sets.add(s["set"])
        if n == 3 and len(sets) == 1:  # F2 не меняет сет — в поиске что-то осталось (например, после перезапуска)
            reset_search(no_matches=True)  # поиск точно не пуст, Esc его сбросит
            continue
        set_id, seen = s["set"], set()
        while s["set"] == set_id and s["id"] not in seen and s["stars"] <= hi:
            seen.add(s["id"])
            if lo <= s["stars"] <= hi and s["id"] not in skip_ids and 0 < s["length"] <= max_length_ms:
                return s
            tap(SC_DOWN, extended=True)
            time.sleep(0.25)
            s = tosu_state()
    return None


SC_F1 = 0x3B
# клавиши модов в меню модов stable (по физическому положению, раскладка не важна) и их биты
MOD_KEYS = {2: 0x10, 1: 0x11, 256: 0x12, 16: 0x1E, 32: 0x1F, 64: 0x20, 8: 0x21, 1024: 0x22, 4096: 0x2E,
            128: 0x2C, 8192: 0x2D, 2048: 0x2F}  # EZ NF HT HR SD DT HD FL SO RX AP AT
LINKED = {512: 64, 16384: 32}  # NC переключается той же клавишей, что DT, PF — что SD


def set_mods(target: int) -> bool:
    """Выставить в выборе песни ровно эти моды (маска osu!): меню модов (F1) и клавиши модов,
    пока tosu не покажет нужное. True — получилось."""
    managed = sum(MOD_KEYS) | sum(LINKED)
    cur = tosu_state()["mods"] & managed
    if cur == target:
        return True
    if not focus_osu():
        return False
    tap(SC_F1)
    time.sleep(0.7)
    for _ in range(8):
        cur = tosu_state()["mods"] & managed
        if cur == target:
            break
        for bit, key in MOD_KEYS.items():
            have = bool(cur & bit) or any(cur & sub and LINKED[sub] == bit for sub in LINKED)
            want = bool(target & bit)
            exact = all(bool(cur & sub) == bool(target & sub) for sub in LINKED if LINKED[sub] == bit)
            if have != want or (want and not exact):  # NC вместо DT — ещё нажатие по кругу
                tap(key)
        time.sleep(0.4)
    tap(SC_F1)  # закрыть меню модов
    time.sleep(0.5)
    return tosu_state()["mods"] & managed == target


def reset_search(no_matches: bool = False):
    """Стереть то, что мы напечатали в поиске. Обычно — Backspace ровно столько раз, сколько печатали:
    Esc на пустом поиске уводит в главное меню. При «No matches found» Backspace не работает, а Esc
    как раз сбрасывает поиск (он в этот момент точно не пуст)."""
    global _typed
    if no_matches:
        tap(SC_ESC)
    else:
        for _ in range(_typed):
            tap(SC_BACKSPACE)
    _typed = 0
    time.sleep(0.5)


def goto_beatmap(set_id: int, beatmap_id: int, timeout: float = 20) -> bool:
    """В выборе песни дойти до нужной сложности. После импорта .osz osu! сам выбирает новую карту;
    иначе ищем по номеру сета (stable находит по нему не всегда) и идём стрелками.
    True — tosu показывает именно её."""
    if not focus_osu():
        return False
    if tosu_state()["set"] != set_id:
        global _typed
        reset_search()
        for ch in str(set_id):
            tap(DIGITS[ch])
        _typed = len(str(set_id))
        if not wait_for(lambda s: s["set"] == set_id, 6):
            reset_search(no_matches=True)  # «No matches found» не оставляем на экране
            return False
    end = time.monotonic() + timeout
    seen = set()
    while time.monotonic() < end:
        s = tosu_state()
        if s["id"] == beatmap_id:
            return True
        seen.add(s["id"])
        # сложности в сете идут по возрастанию звёзд: идём вниз, пока не упрёмся, потом наверх
        tap(SC_DOWN if len(seen) < 12 else SC_UP, extended=True)
        time.sleep(0.35)
        if tosu_state()["set"] != set_id:  # проскочили сет — назад
            tap(SC_UP, extended=True)
            time.sleep(0.35)
            break
    return wait_for(lambda s: s["id"] == beatmap_id, 1)


def wait_for(cond, timeout: float, step: float = 0.3) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        try:
            if cond(tosu_state()):
                return True
        except Exception:
            pass
        time.sleep(step)
    return False
