"""Заказ карт из чата: зритель пишет «!req <ссылка на карту>», карта встаёт в очередь.

Карту берём с зеркала osu.direct (без аккаунта). Принимаются только osu!standard и не NSFW;
у каждого зрителя одна карта в очереди, всего не больше MAX_QUEUE.
Следующий заказ открывает создатель: Ctrl+Shift+N в любом окне или /next в окне Али. Карта скачивается
и открывается в osu! (osu! сам импортирует .osz и переходит к ней), дальше её играет бот.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.request
from pathlib import Path

API = "https://osu.direct/api"
UA = {"User-Agent": "Alya-osu-vtuber/1.0 (personal stream bot)"}
MAX_QUEUE = 10
MAX_LENGTH_S = 5 * 60  # карты длиннее на стриме не играет

# моды в заказе: «!req <ссылка> +HDDT». Только адекватные наборы; DT — только на картах до DT_MAX_STARS
MOD_BITS = {"NM": 0, "HD": 8, "HR": 16, "DT": 64, "NC": 64}
OTHER_MODS = {"EZ", "NF", "HT", "SD", "PF", "FL", "SO", "RX", "AP", "AT", "TD", "V2", "CN"}
ALLOWED_MODS = {0, 8, 16, 64, 8 | 64, 8 | 16}
DT_MAX_STARS = 7.0


def mods_name(mask: int) -> str:
    return "".join(n for n, b in (("HD", 8), ("HR", 16), ("DT", 64)) if mask & b) or "NM"


def parse_mods(text: str):
    """Моды из текста заказа (после ссылки): маска или None, если модов нет; строка — что не так."""
    text = re.sub(r"\S*osu\.ppy\.sh\S*", " ", text)
    found, mask = False, 0
    for word in re.findall(r"\+?([A-Za-z]{2,12})\b", text):
        w = word.upper()
        pairs = [w[i:i + 2] for i in range(0, len(w), 2)] if len(w) % 2 == 0 else []
        if not pairs or not all(p in MOD_BITS or p in OTHER_MODS for p in pairs):
            continue  # обычное слово, не моды
        found = True
        bad = [p for p in pairs if p in OTHER_MODS]
        if bad:
            return f"моды {'+'.join(bad)} на стриме не играем"
        for p in pairs:
            mask |= MOD_BITS[p]
    if not found:
        return None
    if mask not in ALLOWED_MODS:
        return f"набор {mods_name(mask)} — перебор"
    return mask
MAX_OSZ = 80 * 1024 * 1024
DOWNLOADS = Path("data/requests")
SESSION = Path("data/stream_session.json")  # очередь и сыгранные карты переживают перезапуск Али
SESSION_TTL = 4 * 3600                       # с: старше — это уже прошлый стрим
VK_N = 0x4E  # Ctrl+Shift+N

LINK = [
    (re.compile(r"osu\.ppy\.sh/beatmapsets/(\d+)(?:#\w+/(\d+))?"), lambda m: (int(m[1]), int(m[2]) if m[2] else None)),
    (re.compile(r"osu\.ppy\.sh/(?:b|beatmaps)/(\d+)"), lambda m: (None, int(m[1]))),
    (re.compile(r"osu\.ppy\.sh/s/(\d+)"), lambda m: (int(m[1]), None)),
    (re.compile(r"^\s*(\d{2,9})\s*$"), lambda m: (None, int(m[1]))),
]


def parse_link(text: str):
    """(id сета, id сложности) из ссылки или голого номера сложности; None — не похоже на карту."""
    for rx, conv in LINK:
        m = rx.search(text)
        if m:
            return conv(m)
    return None


def _get_json(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=20) as r:
        return json.load(r)


def download(set_id: int) -> Path:
    """Скачать .osz в data/requests (если ещё нет). Зеркало отвечает 503, пока само тянет карту с osu! —
    поэтому несколько попыток с паузой; запасное зеркало — sayobot."""
    path = DOWNLOADS / f"{set_id}.osz"
    if path.exists() and path.stat().st_size > 1000:
        return path
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    urls = [f"{API}/d/{set_id}"] * 4 + [f"https://dl.sayobot.cn/beatmaps/download/full/{set_id}"]
    last = None
    for i, url in enumerate(urls):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120) as r:
                data = r.read(MAX_OSZ + 1)
            if len(data) > MAX_OSZ or not data.startswith(b"PK"):
                raise ValueError("зеркало прислало не карту")
            path.write_bytes(data)
            return path
        except Exception as e:
            last = e
            time.sleep(min(15, 4 * (i + 1)))
    raise last


def _length(sec) -> str:
    sec = int(sec or 0)
    return f"{sec // 60}:{sec % 60:02d}"


class Requests:
    """notify(текст для Али, важное ли) — что сказать в эфире; on_change(список) — для оверлея."""

    def __init__(self, notify, on_change=lambda items: None):
        self.notify, self.on_change = notify, on_change
        self.queue: list[dict] = []
        self.played: set[int] = set()      # id карт, сыгранных в этом стриме (не повторяем)
        self._load_session()
        self.current: dict | None = None   # открытый сейчас заказ
        self.announced_at = 0.0            # когда автопилот объявил текущий заказ
        self._lock = threading.Lock()
        self._busy = False
        threading.Thread(target=self._hotkey, daemon=True).start()

    # --- заказ из чата
    def add(self, user: str, text: str):
        threading.Thread(target=self._add, args=(user, text), daemon=True).start()

    def _add(self, user: str, text: str):
        ids = parse_link(text)
        if not ids:
            return self.notify(f"[заказ] {user} хотел заказать карту, но прислал не ссылку на неё.", False)
        with self._lock:
            if any(r["user"].lower() == user.lower() for r in self.queue):
                return self.notify(f"[заказ] {user} заказывает вторую карту, а первая ещё в очереди.", False)
            if len(self.queue) >= MAX_QUEUE:
                return self.notify(f"[заказ] {user} заказал карту, но очередь заказов полная.", False)
        set_id, diff_id = ids
        mods = parse_mods(text)
        if isinstance(mods, str):
            return self.notify(f"[заказ] {user} заказал карту с модами, но {mods}. Можно: NM, HD, HR, DT, HDDT, HDHR.",
                               False)
        try:
            if set_id is None:
                set_id = _get_json(f"{API}/v2/b/{diff_id}")["beatmapset_id"]
            s = _get_json(f"{API}/v2/s/{set_id}")
        except Exception as e:
            return self.notify(f"[заказ] {user} заказал карту, но я не смогла её найти ({type(e).__name__}).", False)
        std = [b for b in s.get("beatmaps", []) if b.get("mode") == "osu"]
        if s.get("nsfw"):
            return self.notify(f"[заказ] {user} заказал карту с пометкой NSFW — такие на стриме не играем.", False)
        if not std:
            return self.notify(f"[заказ] {user} заказал карту не для osu!standard — ты играешь только стандарт.", False)
        # конкретная сложность из ссылки, иначе самая сложная
        diff = next((b for b in std if b["id"] == diff_id), None) or max(std, key=lambda b: b["difficulty_rating"])
        if (diff.get("total_length") or 0) > MAX_LENGTH_S:
            return self.notify(f"[заказ] {user} заказал карту длиной {_length(diff.get('total_length'))}, "
                               f"а на стриме ты играешь карты не длиннее {_length(MAX_LENGTH_S)}.", False)
        if mods and mods & 64 and diff.get("difficulty_rating", 0) > DT_MAX_STARS:
            return self.notify(f"[заказ] {user} заказал {diff.get('difficulty_rating', 0):.1f}-звёздочную карту с DT — "
                               f"с DT ты играешь карты только до {DT_MAX_STARS:.0f} звёзд.", False)
        req = {"user": user, "set": set_id, "id": diff["id"], "artist": s.get("artist"), "title": s.get("title"),
               "version": diff.get("version"), "stars": diff.get("difficulty_rating", 0),
               "length": _length(diff.get("total_length")), "creator": s.get("creator"), "mods": mods}
        with self._lock:
            if any(r["id"] == req["id"] for r in self.queue):
                return self.notify(f"[заказ] {user} заказал карту, которая уже есть в очереди.", False)
            self.queue.append(req)
            n = len(self.queue)
        self.on_change(self.items())
        self.save_session()
        threading.Thread(target=self._prefetch, args=(req,), daemon=True).start()
        with_mods = f" с модами {mods_name(mods)}" if mods is not None else ""
        self.notify(f"[заказ] {user} заказал {self.name(req)}{with_mods}, {req['stars']:.2f} звёзд, {req['length']}. "
                    f"В очереди заказов: {n}.", True)

    def _prefetch(self, req: dict):
        """Скачать карту заказа заранее: к его очереди файл уже готов, а недоступную карту видно сразу."""
        songs = _songs_dir()
        if songs and any(p.name.startswith(f"{req['set']} ") for p in songs.iterdir()):
            return
        try:
            download(req["set"])
        except Exception as e:
            with self._lock:
                if req not in self.queue:
                    return
                self.queue.remove(req)
            self.on_change(self.items())
            self.save_session()
            self.notify(f"[заказ] Карту {self.name(req)} от {req['user']} не получилось скачать ни с одного зеркала "
                        f"({type(e).__name__}) — заказ снят, пусть закажет другую.", False)

    @staticmethod
    def name(r: dict) -> str:
        return f"{r['artist']} - {r['title']} [{r['version']}]"

    def items(self) -> list[dict]:
        return [{"user": r["user"], "title": r["title"], "version": r["version"], "stars": round(r["stars"], 2)}
                for r in self.queue]

    # --- следующий заказ: Ctrl+Shift+N или /next
    def _hotkey(self):
        import ctypes
        user32 = ctypes.windll.user32
        down = False
        while True:
            from .osu_nav import hotkey
            pressed = hotkey(VK_N)
            if pressed and not down:
                self.next()
            down = pressed
            time.sleep(0.05)

    def _load_session(self):
        try:
            d = json.loads(SESSION.read_text(encoding="utf-8"))
            if time.time() - d.get("saved", 0) < SESSION_TTL:
                self.queue = d.get("queue", [])
                self.played = set(d.get("played", []))
        except Exception:
            pass

    def save_session(self):
        SESSION.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            data = {"saved": time.time(), "queue": self.queue, "played": sorted(self.played)}
        SESSION.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def mark_played(self, beatmap_id):
        if beatmap_id and beatmap_id not in self.played:
            self.played.add(beatmap_id)
            self.save_session()

    def pop(self) -> dict | None:
        """Взять следующий заказ из очереди (None — пусто)."""
        with self._lock:
            req = self.queue.pop(0) if self.queue else None
        if req:
            self.on_change(self.items())
            self.save_session()
        return req

    def next(self) -> str:
        """Ctrl+Shift+N / /next: открыть следующий заказ в выборе песни (играть — уже Enter, руками)."""
        if self._busy:
            return "заказ уже открывается"
        req = self.pop()
        if not req:
            return "очередь заказов пуста"

        def run():
            if self.open_map(req):
                self.notify(f"[заказ] Открыт заказ от {req['user']}: {self.name(req)}.", False)

        threading.Thread(target=run, daemon=True).start()
        return f"открываю: {self.name(req)} (от {req['user']})"

    def open_map(self, req: dict) -> bool:
        """Скачать карту (если её ещё нет), импортировать в osu! и выбрать нужную сложность в выборе песни."""
        from . import osu_nav
        self._busy = True
        try:
            songs = _songs_dir()
            have = songs and any(p.name.startswith(f"{req['set']} ") for p in songs.iterdir())
            if not have:
                os.startfile(download(req["set"]).resolve())  # osu! импортирует .osz (обычно уже скачан заранее)
            # импорт занимает несколько секунд: ищем карту в выборе песни, пока не найдётся
            for _ in range(8):
                if osu_nav.goto_beatmap(req["set"], req["id"]):
                    self.current = req
                    return True
                time.sleep(4)
            raise TimeoutError("карта не нашлась в выборе песни")
        except Exception as e:
            print(f"\n\033[90m(заказ {self.name(req)} не открылся: {type(e).__name__}: {e})\033[0m")
            return False
        finally:
            self._busy = False

    def tag_map_start(self, text: str) -> str | None:
        """К событию «начала карту» дописать, чей это заказ. None — заказ уже объявлен автопилотом,
        второй раз про ту же карту говорить не нужно."""
        r = self.current
        if r and r["title"] and r["title"].lower() in text.lower():
            if time.monotonic() - self.announced_at < 60:
                return None
            return text.rstrip(".") + f". Это заказ зрителя {r['user']}."
        return text


def _songs_dir() -> Path | None:
    try:
        d = _get_json_local("http://127.0.0.1:24050/json/v2")
        p = Path((d.get("folders") or {}).get("songs") or "")
        if p.is_dir():
            return p
    except Exception:
        pass
    p = Path(os.environ.get("LOCALAPPDATA", "")) / "osu!" / "Songs"
    return p if p.is_dir() else None


def _get_json_local(url: str):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.load(r)
