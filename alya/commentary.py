"""Комментарии к игре: следим за osu! через tosu и превращаем происходящее в сообщения «[игра] ...» для мозга.

Чат читает tosu сам, отдельно от бота: бот не ждёт языковую модель и его тайминги не страдают.
События (не на каждую мелочь, чтобы она не болтала без остановки и не тратила лимиты):
  мисс или слайдербрейк на большом комбо (не чаще раза в BREAK_COOLDOWN с), фейл, конец карты с итогом.
  О начале карты молчим: свою карту или заказ объявляет автопилот.
who="alya" — играет бот (Аля говорит о себе), "creator" — играет создатель, она комментирует со стороны.
"""
from __future__ import annotations

import json
import threading
import time

BREAK_MIN_COMBO = 40      # комбо меньше — слом не комментируем
BREAK_COOLDOWN = 12.0     # с между комментариями к промахам
LATE_CHOKE = 85           # % карты: слом большого комбо после этого — «чоук под конец»


class GameWatcher:
    def __init__(self, emit, who: str = "alya", url: str = "ws://127.0.0.1:24050/websocket/v2"):
        self.emit, self.who, self.url = emit, who, url
        self.playing = False
        self.key = None           # (путь карты, моды) текущей игры
        self.prev = None          # прошлый снимок play
        self.last_break = 0.0
        self.breaks_pending = 0
        self.finished = False     # итог текущей попытки уже сказан
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        from websockets.sync.client import connect
        while True:
            try:
                with connect(self.url, open_timeout=3) as ws:
                    for msg in ws:
                        self.update(json.loads(msg))
            except Exception:
                time.sleep(3)  # tosu не запущен — тихо ждём

    # --- текст событий
    def _subj(self, me: str, them: str) -> str:
        return me if self.who == "alya" else them

    def _map_name(self, d) -> str:
        b = d.get("beatmap") or {}
        stars = ((b.get("stats") or {}).get("stars") or {}).get("total") or 0
        mods = ((d.get("play") or {}).get("mods") or {}).get("name") or "NM"
        return (f"{b.get('artist')} - {b.get('title')} [{b.get('version')}] (маппер {b.get('mapper')}), "
                f"{stars:.2f} звёзд, моды {mods}")

    @staticmethod
    def _progress(d) -> int:
        t = (d.get("beatmap") or {}).get("time") or {}
        first, last, live = t.get("firstObject") or 0, t.get("lastObject") or 0, t.get("live") or 0
        if last <= first:
            return 0
        return int(max(0, min(100, 100 * (live - first) / (last - first))))

    @staticmethod
    def _snap(d) -> dict:
        p = d.get("play") or {}
        h = p.get("hits") or {}
        return dict(miss=h.get("0", 0), sb=h.get("sliderBreaks", 0), n100=h.get("100", 0), n50=h.get("50", 0),
                    n300=h.get("300", 0), combo=(p.get("combo") or {}).get("current", 0),
                    max_combo=(p.get("combo") or {}).get("max", 0), acc=p.get("accuracy") or 0,
                    rank=(p.get("rank") or {}).get("current") or "", pp=(p.get("pp") or {}).get("current") or 0,
                    failed=bool(p.get("failed")))

    def _end_text(self, d, s, r=None) -> str:
        """Итог карты: по снимку игры, а если уже открыт экран результатов — по нему."""
        r = r or {}
        b = (d.get("beatmap") or {}).get("stats") or {}
        acc = r.get("accuracy") or s["acc"]
        miss = (r.get("hits") or {}).get("0", s["miss"]) if r.get("accuracy") else s["miss"]
        combo = r.get("maxCombo") or s["max_combo"]
        fc = miss == 0 and s["sb"] == 0
        return (f"[игра] Карта пройдена{' ФУЛЛ КОМБО' if fc else ''}: точность {acc:.2f}%, "
                f"промахов {miss}, слайдербрейков {s['sb']}, макс комбо {combo}"
                f"/{b.get('maxCombo', '?')}, ранг {r.get('rank') or s['rank']}, ~{s['pp']:.0f}pp"
                f"{' (оффлайн, не засчитано)' if self.who == 'alya' else ''}.")

    def update(self, d: dict):
        state = (d.get("state") or {}).get("number")
        path = (d.get("directPath") or {}).get("beatmapFile")
        mods = ((d.get("play") or {}).get("mods") or {}).get("number")
        now = time.monotonic()
        if state == 2:
            s = self._snap(d)
            s["prog"] = self._progress(d)
            key = (path, mods)
            restarted = self.prev is not None and s["n300"] + s["n100"] + s["n50"] + s["miss"] < \
                self.prev["n300"] + self.prev["n100"] + self.prev["n50"] + self.prev["miss"]
            if not self.playing or key != self.key:
                self.playing, self.key, self.prev = True, key, s
                self.finished = False
                self.breaks_pending = 0
                return  # о начале карты не говорим: свою карту или заказ объявляет автопилот
            if restarted:
                self.prev = s
                self.finished = False
                return
            p = self.prev
            # итог — как только засчитана последняя нота: экран результатов osu! показывает на несколько
            # секунд позже (а при сториборде — ещё позже), и Аля молчала бы всё это время
            total = (((d.get("beatmap") or {}).get("stats") or {}).get("objects") or {}).get("total") or 0
            judged = s["n300"] + s["n100"] + s["n50"] + s["miss"]
            if total and judged >= total and not self.finished and not s["failed"]:
                self.finished = True
                self.prev = s
                self.emit(self._end_text(d, s), True)
                return
            if s["failed"] and not p["failed"]:
                self.emit(f"[игра] {self._subj('Ты зафейлила', 'Создатель зафейлил')} карту на "
                          f"{self._progress(d)}%. Точность {s['acc']:.2f}%, промахов {s['miss']}.", True)
            elif s["miss"] > p["miss"] or s["sb"] > p["sb"]:
                if p["combo"] >= BREAK_MIN_COMBO:
                    self.breaks_pending += 1
                    if now - self.last_break > BREAK_COOLDOWN:
                        what = "мисс" if s["miss"] > p["miss"] else "слайдербрейк"
                        more = f" (уже {self.breaks_pending}-й слом с прошлого раза)" if self.breaks_pending > 1 else ""
                        if self._progress(d) >= LATE_CHOKE and p["combo"] >= 100:
                            more += ". Это чоук под самый конец карты"
                        self.emit(f"[игра] Комбо {p['combo']} сломалось: {what} на {self._progress(d)}% карты{more}. "
                                  f"Точность {s['acc']:.2f}%, всего промахов {s['miss']}.", False)
                        self.last_break, self.breaks_pending = now, 0
            self.prev = s
            return
        if self.playing:
            self.playing = False
            s = self.prev or self._snap(d)
            if self.finished:
                pass  # итог уже сказан по последней ноте
            elif state == 7:
                self.emit(self._end_text(d, s, d.get("resultsScreen") or {}), True)
            self.prev = None
