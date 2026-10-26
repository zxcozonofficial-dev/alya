"""Связь с VTube Studio через его API (WebSocket, localhost:8001).

Камеры нет, поэтому лицо модельки ведём сами, 30 раз в секунду подменяя входные параметры трекинга:
  MouthOpen          — по громкости голоса (Voice.level)
  MouthSmile, Brows, EyeOpen*, FaceAngle* — по текущей эмоции из ответа мозга
  плюс лёгкое покачивание головы, моргание и взгляд по сторонам, чтобы она не стояла статуей.
Если в модели есть горячие клавиши, в названии которых встречается эмоция (например «happy»),
они тоже нажимаются при смене эмоции.

В VTube Studio нужно включить API: Настройки → «Start API (allow plugins)», порт 8001.
При первом подключении VTube Studio спросит разрешение — нажми «Allow». Токен сохраняется в data/vts_token.txt.
"""
from __future__ import annotations

import json
import math
import random
import threading
import time
from pathlib import Path

PLUGIN = "Alya"
DEV = "Ruri_AI"
TOKEN = Path("data/vts_token.txt")
FPS = 30

# эмоция -> целевые значения параметров (переход плавный)
FACES = {
    "neutral":     dict(MouthSmile=0.3, Brows=0.5, EyeOpen=0.9, Tilt=0.0, LookX=0.0),
    "happy":       dict(MouthSmile=1.0, Brows=0.7, EyeOpen=0.8, Tilt=4.0, LookX=0.0),
    "smug":        dict(MouthSmile=0.8, Brows=0.35, EyeOpen=0.55, Tilt=-6.0, LookX=0.3),
    "angry":       dict(MouthSmile=0.0, Brows=0.0, EyeOpen=0.75, Tilt=0.0, LookX=0.0),
    "embarrassed": dict(MouthSmile=0.5, Brows=0.75, EyeOpen=0.5, Tilt=8.0, LookX=-0.7),
    "sad":         dict(MouthSmile=0.0, Brows=0.9, EyeOpen=0.6, Tilt=-3.0, LookX=0.0),
    "surprised":   dict(MouthSmile=0.3, Brows=1.0, EyeOpen=1.0, Tilt=0.0, LookX=0.0),
}


class VTS:
    def __init__(self, url: str = "ws://localhost:8001", voice=None):
        from websockets.sync.client import connect
        try:
            self.ws = connect(url, open_timeout=3)
        except Exception as e:
            raise ConnectionError(
                "не получилось подключиться к VTube Studio. Запусти его и включи "
                "Настройки → Start API (allow plugins), порт 8001") from e
        self.voice = voice
        self.emotion = "neutral"
        self.cur = dict(FACES["neutral"])
        self.mouth = 0.0
        self._lock = threading.Lock()
        self._n = 0
        self._auth()
        self.hotkeys = self._hotkeys()
        threading.Thread(target=self._loop, daemon=True).start()

    # --- протокол
    def _req(self, kind: str, data: dict | None = None) -> dict:
        with self._lock:
            self._n += 1
            msg = {"apiName": "VTubeStudioPublicAPI", "apiVersion": "1.0", "requestID": str(self._n),
                   "messageType": kind, "data": data or {}}
            self.ws.send(json.dumps(msg))
            r = json.loads(self.ws.recv())
        if r.get("messageType") == "APIError":
            raise RuntimeError(f"VTube Studio: {r['data'].get('message')}")
        return r["data"]

    def _auth(self):
        token = TOKEN.read_text().strip() if TOKEN.exists() else None
        for attempt in range(2):
            if not token:
                print("VTube Studio: нажми «Allow» в окне VTube Studio, чтобы пустить Алю…")
                token = self._req("AuthenticationTokenRequest",
                                  {"pluginName": PLUGIN, "pluginDeveloper": DEV})["authenticationToken"]
                TOKEN.parent.mkdir(parents=True, exist_ok=True)
                TOKEN.write_text(token)
            ok = self._req("AuthenticationRequest", {"pluginName": PLUGIN, "pluginDeveloper": DEV,
                                                     "authenticationToken": token})["authenticated"]
            if ok:
                return
            token = None  # токен отозвали в VTube Studio — просим новый
        raise RuntimeError("VTube Studio не пустил плагин")

    def _hotkeys(self) -> dict[str, str]:
        """эмоция -> id горячей клавиши модели, если название подходит."""
        try:
            keys = self._req("HotkeysInCurrentModelRequest").get("availableHotkeys", [])
        except Exception:
            return {}
        found = {}
        for emo in FACES:
            for k in keys:
                if emo in (k.get("name") or "").lower():
                    found[emo] = k["hotkeyID"]
                    break
        return found

    # --- управление
    def set_emotion(self, emo: str):
        if emo not in FACES or emo == self.emotion:
            return
        old, self.emotion = self.emotion, emo
        for e in (old, emo):  # горячие клавиши выражений обычно переключаются повторным нажатием
            if e in self.hotkeys and e != "neutral":
                try:
                    self._req("HotkeyTriggerRequest", {"hotkeyID": self.hotkeys[e]})
                except Exception:
                    pass

    def _loop(self):
        t0 = time.perf_counter()
        next_blink = t0 + random.uniform(2, 5)
        look = 0.0
        while True:
            t = time.perf_counter() - t0
            goal = FACES[self.emotion]
            for k, v in goal.items():  # плавный переход за ~0.2 с
                self.cur[k] += (v - self.cur[k]) * 0.2
            level = self.voice.level if self.voice else 0.0
            # рот открывается быстро, закрывается чуть медленнее — меньше дрожит
            self.mouth += (level - self.mouth) * (0.6 if level > self.mouth else 0.3)
            eye = self.cur["EyeOpen"]
            if time.perf_counter() > next_blink:
                if time.perf_counter() > next_blink + 0.12:
                    next_blink = time.perf_counter() + random.uniform(2, 6)
                else:
                    eye = 0.0
            if random.random() < 0.01:  # иногда переводит взгляд
                look = random.uniform(-0.4, 0.4)
            talking = 1.0 + 1.5 * self.mouth  # говорит — качает головой живее
            params = {
                "MouthOpen": min(1.0, self.mouth * 1.3),
                "MouthSmile": self.cur["MouthSmile"],
                "Brows": self.cur["Brows"],
                "EyeOpenLeft": eye, "EyeOpenRight": eye,
                "EyeLeftX": look + self.cur["LookX"], "EyeRightX": look + self.cur["LookX"],
                "FaceAngleX": 6 * math.sin(t * 0.7) * talking,
                "FaceAngleY": 3 * math.sin(t * 1.1 + 1) * talking,
                "FaceAngleZ": self.cur["Tilt"] + 2 * math.sin(t * 0.5),
            }
            try:
                self._req("InjectParameterDataRequest", {
                    "faceFound": True, "mode": "set",
                    "parameterValues": [{"id": k, "value": float(v)} for k, v in params.items()]})
            except Exception as e:
                print(f"\n\033[90m(VTube Studio отключился: {e})\033[0m")
                return
            time.sleep(1 / FPS)
