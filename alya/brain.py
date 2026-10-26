"""Мозг Али: характер (persona.md) + память + языковая модель.

Этап 1 — чат в терминале. Ответ начинается с тега эмоции ([smug] ...): позже по нему
будет меняться выражение лица модельки в VTube Studio, а текст пойдёт в озвучку.

Бэкенды:
  gemini — бесплатный тариф Google AI Studio (ключ GEMINI_API_KEY; запасной ключ — gemini2, GEMINI_API_KEY_2;
           -lite — более быстрая и лёгкая модель с отдельным лимитом)
  groq   — бесплатный тариф Groq (ключ GROQ_API_KEY)
  ollama — локальная модель на своей видеокарте, без ключа и интернета
  claude — Claude API (ключ ANTHROPIC_API_KEY, платный)
gemini/groq/ollama ходят через OpenAI-совместимый протокол и могут подменять друг друга:
если основной упёрся в лимит или недоступен, ответ берётся у следующего из цепочки.

Между запусками хранится текст последних реплик (data/chat_history.json), а факты,
которые Аля решила запомнить, — в data/memory.json; они попадают в системный промпт при старте.
"""
from __future__ import annotations

import datetime
import difflib
import json
import os
import queue
import re
import threading
import time
from pathlib import Path

PERSONA = Path("persona.md")
MEMORY = Path("data/memory.json")
HISTORY = Path("data/chat_history.json")
KEEP_MESSAGES = 40  # сколько прошлых реплик подгружать при старте
EMOTIONS = ("neutral", "happy", "smug", "angry", "embarrassed", "sad", "surprised")

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
# лимиты бесплатного тарифа считаются отдельно на ключ и на модель — цепочка тратит их по очереди
# lite первой: отвечает за ~1 с, и лимит у неё больше, чем у flash
DEFAULT_CHAIN = "gemini-lite,gemini,gemini2-lite,gemini2,ollama"
# flash по умолчанию «думает» ~5 с перед ответом — для живого разговора выключаем
NO_THINKING = {"gemini-3.5-flash": {"reasoning_effort": "none"}, "qwen/qwen3-32b": {"reasoning_effort": "none"}}
GAME_SETTLE = 0.7  # с: ждём, не придут ли следом ещё события игры, чтобы ответить на всё разом
# чат Twitch: отвечает не чаще раза в CHAT_GAP с, из накопившегося видит последние CHAT_PICK сообщений
CHAT_GAP = 8.0
CHAT_PICK = 6
CHAT_FRESH = 30.0  # с: сообщения старше уже неактуальны
CHAT_MAX_LEN = 200
CHAT_BOTS = {"nightbot", "streamelements", "streamlabs", "moobot", "fossabot", "wizebot", "sery_bot"}
LIMIT_COOLDOWN = 60.0  # с: упёршийся в лимит бэкенд пропускаем, чтобы не тратить время на отказ
# name -> (base_url, переменная с ключом, модель по умолчанию, где взять ключ)
OPENAI_BACKENDS = {
    "gemini": (GEMINI_URL, "GEMINI_API_KEY", "gemini-3.5-flash", "aistudio.google.com → Get API key"),
    "gemini-lite": (GEMINI_URL, "GEMINI_API_KEY", "gemini-3.5-flash-lite", "aistudio.google.com → Get API key"),
    "gemini2": (GEMINI_URL, "GEMINI_API_KEY_2", "gemini-3.5-flash", "второй ключ из aistudio.google.com"),
    "gemini2-lite": (GEMINI_URL, "GEMINI_API_KEY_2", "gemini-3.5-flash-lite", "второй ключ из aistudio.google.com"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "qwen/qwen3-32b", "console.groq.com → API Keys"),
    "ollama": ("http://localhost:11434/v1", None, "qwen3:8b", "ключ не нужен, нужна запущенная Ollama"),
}
CLAUDE_MODEL = "claude-opus-5-5"
# модели Claude, у которых есть серверный запасной вариант на случай отказа классификатора
CLAUDE_FALLBACK_MODELS = ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-opus-5")

# Claude запоминает через инструмент remember (так написано в persona.md); остальным —
# пометка в тексте ответа: инструменты у маленьких моделей работают ненадёжно
MARKER_NOTE = ("Вместо инструмента remember: если узнала что-то новое, чего ещё нет в памяти и что ты ещё "
               "не записывала в этом разговоре, допиши в самом конце ответа отдельной строкой "
               "<запомни>факт одной фразой</запомни>. Эта строка не озвучивается. Не повторяй уже записанное.")
# --lang en: говорит по-английски, как Нейро, на каком бы языке ей ни писали
LANG_NOTE = {
    "en": ("LANGUAGE: always answer in English, even when the person writes in Russian or another language "
           "(you understand them perfectly, you just talk in English). Natural casual spoken English, "
           "osu! slang is fine: FC, acc, miss, sliderbreak, pp, farm map. The emotion tag stays at the start."),
    "ru": "",
}
MARKER_RE = re.compile(r"<запомни>(.*?)</запомни>", re.S)

REMEMBER_TOOL = {
    "name": "remember",
    "description": "Сохранить в долговременную память короткий факт о собеседнике или договорённость. "
                   "Память сохраняется между разговорами.",
    "input_schema": {
        "type": "object",
        "properties": {"fact": {"type": "string", "description": "Факт одной фразой"}},
        "required": ["fact"],
        "additionalProperties": False,
    },
    "strict": True,
    "eager_input_streaming": True,
}


class NoKey(Exception):
    pass


def _load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def split_emotion(text: str) -> tuple[str, str]:
    """'[smug] текст' -> ('smug', 'текст'). Без тега — neutral."""
    m = re.match(r"\s*\[(\w+)\]\s*", text)
    if m and m.group(1).lower() in EMOTIONS:
        return m.group(1).lower(), text[m.end():]
    return "neutral", text.strip()


class _HideTags:
    """Вырезает из потока текста <think>...</think> (рассуждения Qwen) и <запомни>...</запомни>."""

    TAGS = (("<think>", "</think>"), ("<запомни>", "</запомни>"))

    def __init__(self, out):
        self.out, self.buf, self.close = out, "", None

    def feed(self, s: str):
        self.buf += s
        while self.buf:
            if self.close:
                i = self.buf.find(self.close)
                if i < 0:
                    self.buf = self.buf[-len(self.close):]
                    return
                self.buf, self.close = self.buf[i + len(self.close):], None
                continue
            hits = [(self.buf.find(o), o, c) for o, c in self.TAGS if o in self.buf]
            if hits:
                i, o, c = min(hits)
                self.out(self.buf[:i])
                self.buf, self.close = self.buf[i + len(o):], c
                continue
            # хвост может оказаться началом тега — придерживаем его
            k = self.buf.rfind("<")
            if k >= 0 and any(o.startswith(self.buf[k:]) for o, _ in self.TAGS):
                self.out(self.buf[:k])
                self.buf = self.buf[k:]
                return
            self.out(self.buf)
            self.buf = ""

    def flush(self):
        if not self.close:
            self.out(self.buf)
        self.buf = ""


class Brain:
    """Общее: характер, память, история между запусками."""

    def __init__(self, fresh: bool = False, lang: str = "ru"):
        self.facts: list[str] = _load_json(MEMORY, {}).get("facts", [])
        self.history: list[dict] = [] if fresh else _load_json(HISTORY, [])
        self.persona = PERSONA.read_text(encoding="utf-8")
        if LANG_NOTE.get(lang):
            self.persona += "\n\n## Язык\n" + LANG_NOTE[lang]
        from .viewers import is_viewer_fact
        # факты о зрителях подтягиваются к их сообщениям в чате, а не висят в промпте все сразу
        mem = "\n".join(f"- {f}" for f in self.facts if not is_viewer_fact(f)) or "(пока пусто)"
        # не меняется до конца запуска, чтобы работал кэш промпта
        self.context = f"Сегодня {datetime.date.today().isoformat()}.\n\nЧто ты помнишь из прошлых разговоров:\n{mem}"
        self.past: list[dict] = []
        for h in self.history[-KEEP_MESSAGES:]:
            if not self.past and h["role"] != "user":
                continue
            self.past.append({"role": h["role"], "content": h["text"]})

    def remember(self, fact) -> bool:
        if not isinstance(fact, str) or not fact.strip():
            return False
        fact = fact.strip()
        key = re.sub(r"[^\w ]", "", fact.lower())
        # тот же факт другими словами — не дублируем
        if any(difflib.SequenceMatcher(None, key, re.sub(r"[^\w ]", "", f.lower())).ratio() > 0.75
               for f in self.facts):
            return False
        self.facts.append(fact)
        _save_json(MEMORY, {"facts": self.facts})
        return True

    def _log(self, user_text: str, answer: str):
        self.history += [{"role": "user", "text": user_text}, {"role": "assistant", "text": answer}]
        _save_json(HISTORY, self.history[-200:])


class ClaudeBrain(Brain):
    name = "claude"

    def __init__(self, model: str = CLAUDE_MODEL, effort: str = "low", fresh: bool = False, lang: str = "ru"):
        super().__init__(fresh, lang)
        import anthropic
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise NoKey("ANTHROPIC_API_KEY (platform.claude.com → API keys)")
        self.client = anthropic.Anthropic()
        self.model, self.effort = model, effort
        self.system = [
            {"type": "text", "text": self.persona, "cache_control": {"type": "ephemeral"}},
            {"type": "text", "text": self.context},
        ]
        # история в одном запуске только дописывается — thinking-блоки модели остаются валидными
        self.messages: list[dict] = list(self.past)

    def reply(self, user_text: str, on_text=lambda s: None, on_note=lambda s: None) -> str:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": user_text})
        extra = {}
        if self.model in CLAUDE_FALLBACK_MODELS:
            extra = {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        parts = []
        while True:
            with self.client.beta.messages.stream(
                model=self.model, max_tokens=8000, system=self.system, messages=self.messages,
                tools=[REMEMBER_TOOL], output_config={"effort": self.effort}, **extra,
            ) as stream:
                for text in stream.text_stream:
                    parts.append(text)
                    on_text(text)
                msg = stream.get_final_message()
            if msg.stop_reason == "refusal":
                # ход целиком откатываем: после него ничего не отправляли, история остаётся валидной
                del self.messages[start:]
                on_note("(Аля отказалась отвечать на это)")
                return ""
            self.messages.append({"role": "assistant", "content": msg.content})
            if msg.stop_reason != "tool_use":
                break
            results = []
            for b in msg.content:
                if b.type == "tool_use":
                    fact = b.input.get("fact") if isinstance(b.input, dict) else None
                    ok = b.name == "remember" and self.remember(fact)
                    if ok:
                        on_note(f"(запомнила: {fact})")
                    results.append({"type": "tool_result", "tool_use_id": b.id,
                                    "content": "сохранено" if ok else "ошибка: нужен непустой fact"})
            self.messages.append({"role": "user", "content": results})
        answer = "".join(parts).strip()
        self._log(user_text, answer)
        return answer


class OpenAIBrain(Brain):
    """gemini / groq / ollama. chain — бэкенды по порядку: при лимите или сбое берётся следующий."""

    def __init__(self, chain: list[str], model: str | None = None, fresh: bool = False, lang: str = "ru"):
        super().__init__(fresh, lang)
        from openai import OpenAI
        self.clients = []
        missing = []
        for i, name in enumerate(chain):
            url, env, default_model, how = OPENAI_BACKENDS[name]
            key = os.environ.get(env) if env else "ollama"
            if not key:
                missing.append(f"{env} ({how})")
                continue
            # пользовательская модель относится к первому бэкенду цепочки
            m = model if (model and i == 0) else default_model
            self.clients.append((name, m, OpenAI(base_url=url, api_key=key, max_retries=0, timeout=30)))
        if not self.clients:
            raise NoKey(", ".join(missing))
        self.missing = missing
        self.cooldown: dict[str, float] = {}
        self.system = f"{self.persona}\n\n{MARKER_NOTE}\n\n{self.context}"
        self.messages: list[dict] = list(self.past)

    @property
    def name(self):
        return " → ".join(f"{n} ({m})" for n, m, _ in self.clients)

    def reply(self, user_text: str, on_text=lambda s: None, on_note=lambda s: None) -> str:
        import openai
        msgs = [{"role": "system", "content": self.system}] + self.messages + [{"role": "user", "content": user_text}]
        last_err = None
        now = time.monotonic()
        ready = [c for c in self.clients if self.cooldown.get(c[0], 0) <= now] or self.clients
        for name, model, client in ready:
            raw = []
            hide = _HideTags(on_text)
            try:
                stream = client.chat.completions.create(model=model, messages=msgs, stream=True,
                                                        max_tokens=1500, **NO_THINKING.get(model, {}))
                for chunk in stream:
                    if not chunk.choices:
                        continue
                    s = chunk.choices[0].delta.content or ""
                    raw.append(s)
                    hide.feed(s)
                hide.flush()
            except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError,
                    openai.InternalServerError, openai.NotFoundError) as e:
                last_err = e
                if isinstance(e, openai.RateLimitError):
                    self.cooldown[name] = time.monotonic() + LIMIT_COOLDOWN
                if raw:  # уже начала говорить — не переспрашиваем другой бэкенд
                    break
                on_note(f"({name} недоступен: {type(e).__name__}, пробую следующий)")
                continue
            text = "".join(raw)
            for fact in MARKER_RE.findall(text):
                if self.remember(fact):
                    on_note(f"(запомнила: {fact.strip()})")
            answer = re.sub(r"<think>.*?</think>", "", MARKER_RE.sub("", text), flags=re.S).strip()
            self.messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": answer}]
            self._log(user_text, answer)
            return answer
        raise last_err


def make_brain(backend: str = DEFAULT_CHAIN, model: str | None = None, effort: str = "low", fresh: bool = False,
               lang: str = "ru"):
    """backend: claude | gemini | groq | ollama | цепочка через запятую, напр. gemini,ollama."""
    if backend == "claude":
        return ClaudeBrain(model or CLAUDE_MODEL, effort, fresh, lang)
    chain = [b.strip() for b in backend.split(",") if b.strip()]
    bad = [b for b in chain if b not in OPENAI_BACKENDS]
    if bad:
        raise ValueError(f"неизвестный бэкенд {bad}; есть: claude, {', '.join(OPENAI_BACKENDS)}")
    return OpenAIBrain(chain, model, fresh, lang)


from .safety import clean  # noqa: E402  (субтитры без голоса проходят тот же фильтр)


def _goodbye(auto, viewers, note) -> str:
    """/конец: новую карту не начинать, в OBS — сцена конца, и ей — попрощаться со зрителями."""
    if auto and auto.enabled:
        note(auto.toggle())
    try:
        from .obs_setup import OBS
        OBS()("SetCurrentProgramScene", sceneName="Alya — конец")
        note("OBS: сцена «Alya — конец»")
    except Exception as e:
        note(f"сцену в OBS не переключила ({type(e).__name__}) — переключи сам")
    names = [viewers.data[k]["name"] for k in viewers.introduced if k in viewers.data] if viewers else []
    who = f" Сегодня в чате были: {', '.join(names[:8])} — можешь кого-то упомянуть." if names else ""
    return ("[стрим] Стрим заканчивается. Попрощайся со зрителями в своём стиле: поблагодари за стрим, "
            f"позови на следующий, 2–3 короткие фразы.{who}")


def _chat_ok(user: str, text: str) -> bool:
    """Что из чата Twitch вообще показывать Але: без команд, ссылок, ботов и пустых сообщений."""
    from .safety import bad
    if user.lower() in CHAT_BOTS or not text or text.startswith("!"):
        return False
    if bad(text) or bad(user):  # она это не увидит и не процитирует
        return False
    return not re.search(r"https?://|www\.|\.(com|ru|gg|tv|net)/", text, re.I)


def chat(backend: str = DEFAULT_CHAIN, model: str | None = None, effort: str = "low", fresh: bool = False,
         lang: str = "ru", voice: str | None = None, vts: bool = False, pitch: str = "+0Hz", rate: str = "+0%",
         game: str | None = None, twitch: str | None = None, overlay: int | None = None, autopilot: bool = False):
    """Чат с Алей в терминале. /память — что она помнит, /выход — выйти.
    voice — голос edge-tts (озвучка ответов), vts — вести модельку в VTube Studio,
    game — комментировать игру из tosu: "alya" (играет бот) или "creator" (играет создатель),
    twitch — канал, чат которого она читает, overlay — порт оверлея для OBS."""
    try:
        brain = make_brain(backend, model, effort, fresh, lang)
    except NoKey as e:
        var = str(e).split()[0]
        print(f"Нет ключа: {e}.\nСохрани его в переменную окружения в PowerShell:\n"
              f"  setx {var} \"твой-ключ\"\nи открой терминал заново.")
        return
    colors = {"happy": "92", "smug": "95", "angry": "91", "embarrassed": "93", "sad": "94",
              "surprised": "96", "neutral": "37"}
    print(f"Аля: {brain.name}. Помню фактов: {len(brain.facts)}, реплик из прошлых разговоров: "
          f"{len(brain.past)}. /память, /конец (попрощаться в конце стрима), /выход")
    for m in getattr(brain, "missing", []):
        print(f"\033[90m(без запасного: нет {m})\033[0m")
    note = lambda s: print(f"\r\033[90m({s})\033[0m")
    screen = None
    if overlay:
        from .overlay import Overlay
        try:
            screen = Overlay(overlay)
            note(f"оверлей для OBS: http://localhost:{overlay}/")
        except OSError as e:
            note(f"оверлей не запустился: порт {overlay} занят ({e})")
    show = screen.send if screen else (lambda kind, **data: None)
    mouth = face = None
    if voice:
        import base64
        import numpy as np
        from .voice import SR, Voice
        mouth = Voice(voice, rate, pitch)
        # субтитры идут вместе с голосом: фраза появляется, когда начинает звучать
        mouth.on_sentence = lambda s: show("say", text=s)
        mouth.on_idle = lambda: show("say_end")
        # голос дублируется в оверлей: в OBS его берёт браузерный источник, а не «Звук рабочего стола»
        # (иначе в стрим попадёт всё, что играет на компьютере, включая Discord)
        mouth.on_audio = lambda a: show("audio", sr=SR, pcm=base64.b64encode(
            (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()).decode())
        note(f"голос: {voice} {pitch} {rate}")
    if vts:
        from .vts import VTS
        try:
            face = VTS(voice=mouth)
            note(f"VTube Studio подключён; горячие клавиши эмоций: {list(face.hotkeys) or 'нет'}")
        except Exception as e:
            note(str(e))
    # реплики из терминала, события игры и чат Twitch идут в одну очередь: (вид, текст, важное ли, когда пришло)
    inbox: queue.Queue = queue.Queue()

    def read_input():
        while True:
            try:
                inbox.put(("user", input().strip(), True, time.monotonic()))
            except (EOFError, KeyboardInterrupt):
                inbox.put(("quit", "", True, time.monotonic()))
                return

    threading.Thread(target=read_input, daemon=True).start()

    def read_command_file():
        """Команды извне (как из терминала): строки в data/alya_command.txt, файл удаляется после чтения."""
        path = Path("data/alya_command.txt")
        while True:
            time.sleep(1)
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
                path.unlink()
            except OSError:
                continue
            for line in filter(None, (s.strip() for s in lines)):
                if line in ("/конец", "/end") and auto and auto.enabled:
                    note(auto.toggle())  # сразу, чтобы не успела начать новую карту
                inbox.put(("user", line, True, time.monotonic()))

    threading.Thread(target=read_command_file, daemon=True).start()
    reqs = auto = None  # заказы карт из чата и автопилот (только вместе с --twitch)
    replying = [False]  # сейчас сочиняет ответ
    if game:
        from .commentary import GameWatcher

        def on_game(text, important):
            if reqs and ("начала карту" in text or "начал карту" in text):
                text = reqs.tag_map_start(text)
                if text is None:
                    return
            inbox.put(("game", text, important, time.monotonic()))
            show("game", text=text.removeprefix("[игра] "))

        GameWatcher(on_game, game)
        note(f"комментирует игру из tosu: {'играет она сама' if game == 'alya' else 'играешь ты'}")
    viewers = None
    if twitch:
        from .twitch import TwitchChat
        from .requests import Requests
        from .viewers import Viewers
        viewers = Viewers()
        reqs = Requests(lambda text, important: inbox.put(("game", text, important, time.monotonic())),
                        lambda items: show("requests", items=items))
        if reqs.queue or reqs.played:
            note(f"продолжаю стрим: в очереди заказов {len(reqs.queue)}, уже сыграно карт {len(reqs.played)}")
            show("requests", items=reqs.items())

        def on_twitch(user, text, color):
            show("chat", user=user, text=text, color=color)
            low = text.lower()
            if user.lower() == twitch.lower().lstrip("#") and not low.startswith("!"):
                # хозяин канала — это создатель: его сообщение как реплика из терминала, а не от зрителя
                inbox.put(("user", text, True, time.monotonic()))
                return
            viewers.seen(user, text)
            if low.startswith(("!req ", "!request ", "!r ")):
                reqs.add(user, text.split(None, 1)[1])
            if _chat_ok(user, text):
                inbox.put(("twitch", f"{user}: {text[:CHAT_MAX_LEN]}", False, time.monotonic()))

        TwitchChat(twitch, on_twitch, note)
        from .autopilot import Autopilot
        # «занята» — говорит или сочиняет ответ; очередь чата не считается, иначе в живом чате автопилот встанет
        auto = Autopilot(reqs, lambda text, important: inbox.put(("game", text, important, time.monotonic())),
                         busy=lambda: replying[0] or bool(mouth and mouth.speaking), log=note)
        if autopilot:
            note(auto.toggle())
        else:
            note("автопилот выключен: Ctrl+A в окне osu! или /auto — Аля сама играет заказы и свои карты")
        show("channel", name=twitch.lower().lstrip("#"))
    print()
    prompt = lambda: print("\033[1mты >\033[0m ", end="", flush=True)
    prompt()
    last_chat_reply = 0.0
    while True:
        try:
            batch = [inbox.get()]
        except KeyboardInterrupt:
            print()
            return
        if mouth:
            mouth.wait()  # не перебиваем её, пока договаривает
        first = batch[0][0]
        if first == "game":
            # события часто идут пачкой (вышел с карты → начал новую → ретрай): даём им собраться
            time.sleep(max(0.0, batch[0][3] + GAME_SETTLE - time.monotonic()))
        elif first == "twitch":
            # на чат отвечает не чаще раза в CHAT_GAP с, сообщения пока копятся — потом выберет интересное;
            # событие игры или реплика создателя ждать не заставляют
            deadline = max(last_chat_reply + CHAT_GAP, batch[0][3] + 1.5)
            while time.monotonic() < deadline:
                try:
                    batch.append(inbox.get(timeout=0.1))
                except queue.Empty:
                    continue
                if batch[-1][0] != "twitch":
                    break
        while not inbox.empty():  # всё, что накопилось, пока она говорила, — в один ответ
            batch.append(inbox.get_nowait())
        if any(k == "quit" for k, *_ in batch):
            print()
            if viewers:
                viewers.save()
            return
        users = []
        for _, t, _, _ in (b for b in batch if b[0] == "user"):
            if t in ("/выход", "/exit", "/quit"):
                return
            if t in ("/память", "/memory"):
                print("\n".join(f"  - {f}" for f in brain.facts) or "  (пусто)")
            elif t in ("/auto", "/авто"):
                print(f"\033[90m({auto.toggle() if auto else 'автопилот работает только с --twitch'})\033[0m")
            elif t in ("/конец", "/end"):
                users.append(_goodbye(auto, viewers, note))
            elif t in ("/next", "/след"):
                print(f"\033[90m({reqs.next() if reqs else 'заказы работают только с --twitch'})\033[0m")
            elif t in ("/заказы", "/queue"):
                q = reqs.queue if reqs else []
                print("\n".join(f"  {i}. {reqs.name(r)} — {r['user']}" for i, r in enumerate(q, 1)) or "  (заказов нет)")
            elif t:
                users.append(t)
        now = time.monotonic()
        games = [b for b in batch if b[0] == "game"]
        last_big = max((i for i, b in enumerate(games) if b[2]), default=-1)
        # мелочь до важного события (слом комбо до конца карты) и устаревшая мелочь уже не интересна
        games = [b[1] for i, b in enumerate(games) if i >= last_big and (b[2] or now - b[3] < 6)][-3:]
        chats = [b[1] for b in batch if b[0] == "twitch" and now - b[3] < CHAT_FRESH][-CHAT_PICK:]
        if viewers:  # справка о зрителе при первом его сообщении за стрим
            chats = [c + viewers.note(c.split(": ", 1)[0], brain.facts) for c in chats]
        if not users and not games and not chats:
            prompt()
            continue
        for g in games + [f"[чат] {c}" for c in chats]:
            print(f"\r\033[90m{g}\033[0m")
        parts = users + games
        if chats:
            parts.append("[чат]\n" + "\n".join(chats))
            last_chat_reply = now
        text = "\n".join(parts)
        if games or chats:
            text += " (Отвечай вслух одной-двумя короткими фразами."
            if chats:
                text += (" В [чат] — сообщения зрителей Twitch: выбери одно-два интересных и ответь, обращаясь"
                         " по нику; остальное пропусти.")
            if len(games) > 1:
                text += " Событий игры несколько — скажи про главное."
            text += ")"
        buf, shown = [], False

        def flush():
            nonlocal shown
            emo, rest = split_emotion("".join(buf))
            print(f"\033[{colors[emo]}mАля [{emo}] >\033[0m {rest}", end="", flush=True)
            shown = True
            show("emotion", emotion=emo)
            if face:
                face.set_emotion(emo)
            if mouth:
                mouth.say(rest)
            else:
                show("say", text=clean(rest))

        def on_text(s):
            if shown:
                print(s, end="", flush=True)
                if mouth:
                    mouth.say(s)
                else:
                    show("say", text=clean(s))
                return
            buf.append(s)
            joined = "".join(buf).lstrip()
            # ждём, пока придёт тег эмоции целиком, чтобы показать его цветом
            if "]" in joined or (joined and not joined.startswith("[")) or len(joined) > 20:
                flush()

        show("thinking")
        replying[0] = True
        try:
            brain.reply(text, on_text, on_note=lambda s: print(f"\n\033[90m{s}\033[0m", end=""))
            if not shown and buf:
                flush()
            if mouth:
                mouth.flush()
            else:
                show("say_end")
        except Exception as e:  # сеть, лимиты, неверный ключ — чат не падает
            print(f"\033[90m(ошибка: {type(e).__name__}: {str(e)[:200]})\033[0m")
        replying[0] = False
        print("\n")
        prompt()
