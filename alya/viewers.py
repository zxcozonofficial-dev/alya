"""Память о зрителях Twitch: кто пишет впервые, кто постоянный, что писал раньше, что Аля о нём запомнила.

data/viewers.json — по нику: первое и последнее появление, на скольких стримах был, сколько написал
и последние сообщения. Факты о зрителе Аля запоминает сама в общей памяти строкой «Зритель ник: ...»
(так сказано в persona.md); здесь они подтягиваются к нику и в общий системный промпт не идут.
Когда зритель впервые за стрим попадает в ответ, к его сообщению приписывается короткая справка.
"""
from __future__ import annotations

import atexit
import datetime
import json
import re
import time
from pathlib import Path

PATH = Path("data/viewers.json")
KEEP_RECENT = 4          # сколько последних сообщений зрителя хранить
FACT_PREFIX = "зритель "


def is_viewer_fact(fact: str) -> bool:
    return fact.lower().startswith(FACT_PREFIX)


class Viewers:
    def __init__(self):
        try:
            self.data: dict = json.loads(PATH.read_text(encoding="utf-8"))
        except Exception:
            self.data = {}
        self.session = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        self.introduced: set[str] = set()   # о ком в этом стриме уже рассказали Але
        self._last_save = 0.0
        atexit.register(self.save)  # /выход, Ctrl+C — счётчики не теряются

    def seen(self, nick: str, text: str):
        """Каждое сообщение чата (до фильтров): обновить счётчики."""
        key = nick.lower()
        v = self.data.setdefault(key, {"name": nick, "first": self.session, "streams": 0, "msgs": 0, "recent": []})
        v["name"] = nick
        # один стрим — пока между появлениями меньше 4 часов (перезапуск Али посреди стрима — не новый стрим)
        if v.get("last") != self.session and time.time() - v.get("last_ts", 0) > 4 * 3600:
            v["streams"] += 1
            v["prev_recent"] = v.get("recent", [])  # что писал на прошлых стримах
            v["recent"] = []
            v["last"] = self.session
        v["last"], v["last_ts"] = self.session, time.time()
        v["msgs"] += 1
        v["recent"] = (v["recent"] + [text[:120]])[-KEEP_RECENT:]
        if time.monotonic() - self._last_save > 10:
            self.save()

    def save(self):
        PATH.parent.mkdir(parents=True, exist_ok=True)
        PATH.write_text(json.dumps(self.data, ensure_ascii=False, indent=1), encoding="utf-8")
        self._last_save = time.monotonic()

    def note(self, nick: str, facts: list[str]) -> str:
        """Справка о зрителе для Али — только при первом появлении за стрим, иначе пусто."""
        key = nick.lower()
        if key in self.introduced:
            return ""
        self.introduced.add(key)
        v = self.data.get(key)
        mine = [re.sub(r"^зритель\s+\S+\s*:\s*", "", f, flags=re.I) for f in facts
                if f.lower().startswith(f"{FACT_PREFIX}{key}:") or f.lower().startswith(f"{FACT_PREFIX}{key} ")]
        if not v or v["streams"] <= 1:
            parts = ["впервые в чате"]
        else:
            parts = [f"постоянный зритель, {v['streams']}-й стрим (с {v['first'][:10]})"]
            if v.get("prev_recent"):
                parts.append("в прошлый раз писал: " + " / ".join(f"«{m}»" for m in v["prev_recent"][-2:]))
        if mine:
            parts.append("ты помнишь: " + "; ".join(mine[-3:]))
        return " [о зрителе: " + "; ".join(parts) + "]"
