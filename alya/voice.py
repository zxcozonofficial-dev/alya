"""Голос Али: бесплатный Microsoft Edge TTS (edge-tts, без ключа) + воспроизведение.

Текст подаётся кусками по мере того, как мозг его пишет: каждое законченное предложение сразу
уходит в синтез, пока играет предыдущее, — так Аля начинает говорить, не дожидаясь конца ответа.
Во время воспроизведения level (0..1, громкость последних ~20 мс) — по нему VTube Studio открывает рот.
"""
from __future__ import annotations

import asyncio
import queue
import re
import threading

import numpy as np

VOICE = "en-US-AnaNeural"
SR = 24000
SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
CLAUSE_END = re.compile(r"(?<=[,;:—])\s+")
FIRST_CLAUSE = 12  # символов: совсем короткое «Oh,» отдельно не озвучиваем


def clean_for_speech(text: str) -> str:
    """Убирает то, что не надо читать вслух: теги эмоций, markdown, смайлики в скобках."""
    text = re.sub(r"\[\w+\]", "", text)
    text = re.sub(r"[*_`#~]+", "", text)
    text = re.sub(r"[:;]-?[()DPp]|\^\^|\(\s*[^\w\s]{1,4}\s*\)", "", text)
    return re.sub(r"\s+", " ", text).strip()


async def _synth(text: str, voice: str, rate: str, pitch: str) -> bytes:
    import edge_tts
    out = bytearray()
    async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
        if chunk["type"] == "audio":
            out += chunk["data"]
    return bytes(out)


def _decode(mp3: bytes) -> np.ndarray:
    import miniaudio
    d = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=SR)
    return np.frombuffer(d.samples, np.float32)


HOLD = 2304  # отсчётов: хвост декодированного куска может измениться с приходом следующего mp3-кадра


async def _synth_stream(text: str, voice: str, rate: str, pitch: str, on_samples):
    """Отдаёт звук по мере прихода от сервера: первый кусок слышно через ~0.8 с, а не после всей фразы."""
    import edge_tts
    mp3, sent, n = bytearray(), 0, 0
    async for chunk in edge_tts.Communicate(text, voice, rate=rate, pitch=pitch).stream():
        if chunk["type"] != "audio":
            continue
        mp3 += chunk["data"]
        n += 1
        if n % 4 == 0:  # пересобираем с начала: фразы короткие, это дёшево
            a = _decode(bytes(mp3))
            if len(a) - HOLD > sent:
                on_samples(a[sent:len(a) - HOLD].copy())
                sent = len(a) - HOLD
    if mp3:
        a = _decode(bytes(mp3))
        if len(a) > sent:
            on_samples(a[sent:].copy())


def synth(text: str, voice: str = VOICE, rate: str = "+0%", pitch: str = "+0Hz") -> np.ndarray:
    """Текст -> моно float32 при SR. Пустой массив, если синтез не удался."""
    import miniaudio
    mp3 = asyncio.run(_synth(text, voice, rate, pitch))
    if not mp3:
        return np.zeros(0, np.float32)
    d = miniaudio.decode(mp3, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=SR)
    return np.frombuffer(d.samples, np.float32).copy()


class Voice:
    """say(кусок текста) — можно звать много раз подряд по мере стриминга; flush() — договорить хвост."""

    def __init__(self, voice: str = VOICE, rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice, self.rate, self.pitch = voice, rate, pitch
        self.level = 0.0
        self.buf = ""
        self._started = False  # первая фраза ответа уже ушла в синтез
        self._texts: queue.Queue = queue.Queue()
        self._audio: queue.Queue = queue.Queue()
        self._busy = 0  # предложений в работе (синтез или воспроизведение)
        self._lock = threading.Lock()
        # зовутся из потока воспроизведения: фраза начала звучать / всё сказанное доиграло (для субтитров)
        self.on_sentence = lambda text: None
        self.on_idle = lambda: None
        # кусок звука (float32, SR) в момент, когда он начинает играть, — оверлей проигрывает его внутри OBS
        self.on_audio = lambda samples: None
        threading.Thread(target=self._synth_loop, daemon=True).start()
        threading.Thread(target=self._play_loop, daemon=True).start()

    def say(self, chunk: str):
        self.buf += chunk
        parts = SENTENCE_END.split(self.buf)
        for s in parts[:-1]:
            self._push(s)
            self._started = True
        self.buf = parts[-1]
        if not self._started:
            # начало ответа: озвучиваем первую фразу до запятой, не дожидаясь конца предложения
            m = CLAUSE_END.search(self.buf, FIRST_CLAUSE)
            if m:
                self._push(self.buf[:m.start()])
                self.buf = self.buf[m.end():]
                self._started = True

    def flush(self):
        self._push(self.buf)
        self.buf = ""
        self._started = False

    def _push(self, text: str):
        from .safety import bad, clean
        text = clean_for_speech(text)
        if not re.search(r"\w", text):
            return
        if bad(text):  # фильтр стрима: фраза не звучит и не попадает в субтитры
            print(f"\n\033[90m(фильтр: не озвучила «{text[:80]}»)\033[0m")
            text = clean(text)
        with self._lock:
            self._busy += 1
        self._texts.put(text)

    @property
    def speaking(self) -> bool:
        return self._busy > 0

    def wait(self):
        """Ждать, пока всё сказанное доиграет."""
        import time
        while self.speaking:
            time.sleep(0.02)

    def _done(self):
        with self._lock:
            self._busy -= 1
            idle = self._busy == 0
        if idle:
            self.on_idle()

    def _synth_loop(self):
        while True:
            text = self._texts.get()
            self._audio.put(text)  # метка начала фразы для субтитров
            try:
                asyncio.run(_synth_stream(text, self.voice, self.rate, self.pitch, self._audio.put))
            except Exception as e:  # нет сети и т. п. — молчим, но чат не падает
                print(f"\n\033[90m(голос: {type(e).__name__}: {str(e)[:120]})\033[0m")
            self._audio.put(None)  # конец фразы

    def _play_loop(self):
        import sounddevice as sd
        block = SR // 50  # 20 мс
        with sd.OutputStream(samplerate=SR, channels=1, dtype="float32", blocksize=block) as out:
            while True:
                a = self._audio.get()
                if a is None:
                    self.level = 0.0
                    self._done()
                    continue
                if isinstance(a, str):
                    self.on_sentence(a)
                    continue
                self.on_audio(a)
                for i in range(0, len(a), block):
                    b = a[i:i + block]
                    self.level = min(1.0, float(np.sqrt(np.mean(b ** 2))) * 6)
                    out.write(b.reshape(-1, 1))
