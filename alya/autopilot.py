"""Автопилот: Аля сама ходит по меню osu! и играет заказы из чата, без рук создателя.

Цикл: экран результатов → (дать ей договорить) Esc → выбор песни → пауза на чат → следующий заказ:
скачать, найти нужную сложность, объявить, Enter → играет бот → результаты → и так далее.
Если заказов нет — выбирает карту сама: F2 (случайная карта) и стрелками до сложности в OWN_STARS
звёзд с выбранными сейчас модами, без повторов за стрим. Включается и выключается Ctrl+A в окне osu! или /auto.

Пока автопилот включён, он выводит окно osu! на передний план и жмёт клавиши — компьютером
в это время лучше не пользоваться (голосом в Discord — можно).
"""
from __future__ import annotations

import ctypes
import threading
import time

from . import osu_nav as nav
from .requests import MAX_LENGTH_S, mods_name

VK_A = 0x41  # Ctrl+A в окне osu!
RESULTS_HOLD = 4.0     # с на экране результатов (итог она говорит уже по последней ноте)
BETWEEN_MAPS = 4.0     # с в выборе песни между картами
START_TIMEOUT = 15.0   # с после Enter: не началась игра — пропускаем заказ
PAUSE_GIVE_UP = 10.0   # с на паузе (обычно из-за потери фокуса окна), потом карта бросается
RETRY_EVERY = 5.0      # с между попытками что-то нажать в меню
FAIL_HOLD = 3.0        # с на экране фейла, потом Esc
OWN_STARS = (6.5, 9.3) # диапазон звёзд для карт, которые она выбирает сама


class Autopilot:
    def __init__(self, requests, notify, busy=lambda: False, log=print, stars=OWN_STARS):
        """notify(текст, важное) — сказать Але; busy() — она сейчас говорит или сочиняет ответ;
        stars — диапазон звёзд для карт, которые она выбирает сама."""
        self.reqs, self.notify, self.busy, self.log = requests, notify, busy, log
        self.stars = stars
        self.played = requests.played  # id сыгранных в этом стриме — общие с заказами, переживают перезапуск
        self.default_mods = None       # моды для её собственных карт
        self.enabled = False
        self._results_at = None
        self._idle_from = time.monotonic()
        self._said_empty = False
        self._said_menu = False
        self._unfocused_at = None
        self._failed_at = None
        self._last_try = 0.0
        threading.Thread(target=self._hotkey, daemon=True).start()
        threading.Thread(target=self._loop, daemon=True).start()

    def toggle(self) -> str:
        self.enabled = not self.enabled
        self._said_empty = self._said_menu = False
        if self.enabled:  # моды, выбранные при включении, — для её собственных карт (заказ может их поменять)
            try:
                self.default_mods = nav.tosu_state()["mods"] & (sum(nav.MOD_KEYS) | sum(nav.LINKED))
            except Exception:
                self.default_mods = None
        self._idle_from = time.monotonic()
        return "автопилот включён: Аля сама играет заказы (Ctrl+A в osu! — выключить)" if self.enabled else \
            "автопилот выключен: меню osu! снова твоё"

    def _hotkey(self):
        user32 = ctypes.windll.user32
        down = False
        while True:
            pressed = nav.hotkey(VK_A, shift=False, only_osu=True)
            if pressed and not down:
                self.log(self.toggle())
            down = pressed
            time.sleep(0.05)

    def _loop(self):
        while True:
            time.sleep(0.5)
            if not self.enabled:
                continue
            try:
                self._step()
            except Exception as e:
                self.log(f"автопилот: {type(e).__name__}: {e}")
                time.sleep(3)

    def _step(self):
        s = nav.tosu_state()
        now = time.monotonic()
        if s["state"] == nav.RESULTS:
            self._results_at = self._results_at or now
            if now - self._results_at > RESULTS_HOLD:
                nav.press(nav.SC_ESC)
                self._results_at = None
                self._idle_from = time.monotonic()
            return
        self._results_at = None
        if s["state"] == nav.PLAYING:
            self._idle_from = now
            self.reqs.mark_played(s["id"])  # любая сыгранная карта, в том числе заказ или запущенная руками
            # экран фейла: osu! формально всё ещё «играет» — через пару секунд выходим в выбор песни
            if s["failed"]:
                self._failed_at = self._failed_at or now
                if now - self._failed_at > FAIL_HOLD and now - self._last_try > RETRY_EVERY:
                    self._last_try = now
                    nav.press(nav.SC_ESC)
                return
            self._failed_at = None
            # osu! stable сам ставит паузу, когда окно теряет фокус (клик в Discord и т. п.), а продолжить
            # с клавиатуры нельзя: Esc в меню паузы выходит с карты. Долгая пауза — бросаем карту и берём следующую
            if s["paused"]:
                self._unfocused_at = self._unfocused_at or now
                if now - self._unfocused_at > PAUSE_GIVE_UP and now - self._last_try > RETRY_EVERY:
                    self._last_try = now
                    self.log("автопилот: карта давно стоит на паузе — выхожу и беру следующую")
                    nav.press(nav.SC_ESC)
            else:
                self._unfocused_at = None
            return
        if s["state"] == nav.MENU:
            if now - self._last_try > RETRY_EVERY:  # главное меню → «Играть» → «Одиночная»
                self._last_try = now
                nav.focus_osu()
                nav.tap(nav.SC_P)
                time.sleep(0.8)
                nav.tap(nav.SC_P)
            return
        if s["state"] != nav.SONG_SELECT:
            if not self._said_menu:
                self.log("автопилот: открой выбор песен в osu! — дальше я сама")
                self._said_menu = True
            return
        self._said_menu = False
        if now - self._idle_from < BETWEEN_MAPS:
            return
        req = self.reqs.pop()
        if req:
            self._said_empty = False
            self._play(req)
            return
        self._play_own()

    def _play_own(self):
        if self.default_mods is not None:  # после заказа с другими модами — вернуть свои
            nav.set_mods(self.default_mods)
        e = nav.roll_random(*self.stars, skip_ids=self.played, max_length_ms=MAX_LENGTH_S * 1000)
        if not e:
            self.log(f"автопилот: не нашла карту на {self.stars[0]}–{self.stars[1]}★ с этими модами — попробую позже")
            self._idle_from = time.monotonic()
            return
        self.reqs.mark_played(e["id"])
        name = f"{e['artist']} - {e['title']} [{e['version']}]"
        self.log(f"автопилот: заказов нет, сама выбрала {name} ({e['stars']:.2f}★)")
        self.reqs.current = {"title": e["title"], "user": "Аля"}
        self.reqs.announced_at = time.monotonic()
        self.notify(f"[выбор] Заказов нет, и ты сама выбрала карту: {name} (маппер {e['mapper']}), "
                    f"{e['stars']:.2f} звёзд. Скажи коротко, почему именно её.", True)
        if not self._start():
            self.log(f"автопилот: {name} не запустилась")

    def _start(self) -> bool:
        """Дать ей объявить карту (не дольше 5 с), нажать Enter и дождаться начала игры."""
        end = time.monotonic() + 5
        time.sleep(1.0)
        while self.busy() and time.monotonic() < end:
            time.sleep(0.3)
        if not self.enabled:
            return True
        nav.press(nav.SC_ENTER)
        ok = nav.wait_for(lambda st: st["state"] == nav.PLAYING, 4)
        if not ok:  # первое Enter могло уйти в закрывающееся меню модов
            nav.press(nav.SC_ENTER)
            ok = nav.wait_for(lambda st: st["state"] == nav.PLAYING, START_TIMEOUT)
        self._idle_from = time.monotonic()
        return ok

    def _play(self, req):
        name = self.reqs.name(req)
        self.log(f"автопилот: открываю {name} (от {req['user']})")
        if not self.reqs.open_map(req):
            self.notify(f"[заказ] Заказ от {req['user']} ({name}) не получилось открыть — пропускаешь его.", False)
            self._idle_from = time.monotonic()
            return
        mods = ""
        if req.get("mods") is not None:
            if nav.set_mods(req["mods"]):
                mods = f" с модами {mods_name(req['mods'])}"
            else:
                self.log(f"автопилот: не получилось выставить моды {mods_name(req['mods'])}")
        self.notify(f"[заказ] Сейчас играешь заказ от {req['user']}: {name}{mods}, {req['stars']:.2f} звёзд.", True)
        self.reqs.announced_at = time.monotonic()
        if not self._start():
            self.notify(f"[заказ] Заказ от {req['user']} не запустился — пропускаешь.", False)
