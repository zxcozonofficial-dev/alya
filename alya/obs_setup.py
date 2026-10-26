"""Настройка OBS под стрим Али через obs-websocket (встроен в OBS 28+, порт 4455).

Создаёт отдельную коллекцию сцен «Alya» (твои существующие сцены не трогаются) и в ней:
  Alya — игра:   оверлей (index.html) → захват osu! в окно оверлея → VTube Studio поверх
  Alya — скоро / отошла / конец: заставка (scene.html?mode=...) → VTube Studio поверх
VTube Studio захватывается на весь холст с прозрачностью: модельку двигаешь прямо в окне VTube Studio
(правый нижний угол — место под неё на оверлее). Холст 1920×1080, 60 кадров.
В OBS: Сервис → Настройки сервера WebSocket → «Включить сервер WebSocket».
"""
from __future__ import annotations

import base64
import ctypes
import ctypes.wintypes as wt
import hashlib
import json
import os
import time
import uuid

COLLECTION = "Alya"
OVERLAY = "http://localhost:24080"
GAME_RECT = (32, 32, 1408, 792)   # окно игры на оверлее: x, y, ширина, высота
VTS_CROP_TOP = 0.43               # доля окна VTube Studio сверху, которая не попадает в кадр


class OBS:
    def __init__(self, url="ws://127.0.0.1:4455", password: str | None = None):
        from websockets.sync.client import connect
        self.ws = connect(url, open_timeout=3, max_size=None)
        hello = json.loads(self.ws.recv())["d"]
        ident = {"rpcVersion": 1}
        auth = hello.get("authentication")
        if auth:
            if not password:
                raise PermissionError("в OBS включена аутентификация WebSocket")
            secret = base64.b64encode(hashlib.sha256((password + auth["salt"]).encode()).digest()).decode()
            ident["authentication"] = base64.b64encode(
                hashlib.sha256((secret + auth["challenge"]).encode()).digest()).decode()
        self.ws.send(json.dumps({"op": 1, "d": ident}))
        msg = json.loads(self.ws.recv())
        if msg["op"] != 2:
            raise ConnectionError(f"OBS не принял подключение: {msg}")

    def __call__(self, kind: str, ok_codes=(), **data):
        rid = uuid.uuid4().hex
        self.ws.send(json.dumps({"op": 6, "d": {"requestType": kind, "requestId": rid, "requestData": data}}))
        while True:
            msg = json.loads(self.ws.recv())
            if msg["op"] == 7 and msg["d"]["requestId"] == rid:
                st = msg["d"]["requestStatus"]
                if not st["result"] and st["code"] not in ok_codes:
                    raise RuntimeError(f"OBS {kind}: {st.get('comment') or st['code']}")
                return msg["d"].get("responseData") or {}


def _window_of(exe_name: str) -> str | None:
    """Строка окна для захвата игры в формате OBS «заголовок:класс:exe» — первое видимое окно процесса."""
    user32, kernel32 = ctypes.windll.user32, ctypes.windll.kernel32
    found = []

    def cb(hwnd, _):
        if not user32.IsWindowVisible(hwnd):
            return True
        pid = wt.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        h = kernel32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return True
        buf = ctypes.create_unicode_buffer(1024)
        size = wt.DWORD(1024)
        kernel32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(size))
        kernel32.CloseHandle(h)
        if os.path.basename(buf.value).lower() == exe_name.lower():
            title = ctypes.create_unicode_buffer(512)
            cls = ctypes.create_unicode_buffer(256)
            user32.GetWindowTextW(hwnd, title, 512)
            user32.GetClassNameW(hwnd, cls, 256)
            if title.value:
                # OBS экранирует двоеточия в частях строки как «#3A»
                esc = lambda s: s.replace("#", "#22").replace(":", "#3A")
                found.append(f"{esc(title.value)}:{esc(cls.value)}:{esc(os.path.basename(buf.value))}")
                return False
        return True

    proto = ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    user32.EnumWindows(proto(cb), 0)
    return found[0] if found else None


def setup(password: str | None = None):
    obs = OBS(password=password)
    names = obs("GetSceneCollectionList")["sceneCollections"]
    fresh = COLLECTION not in names
    if not fresh:
        obs("SetCurrentSceneCollection", sceneCollectionName=COLLECTION)
    else:
        obs("CreateSceneCollection", sceneCollectionName=COLLECTION)  # сразу становится текущей
    print(f"коллекция сцен «{COLLECTION}»")

    obs("SetVideoSettings", baseWidth=1920, baseHeight=1080, outputWidth=1920, outputHeight=1080,
        fpsNumerator=60, fpsDenominator=1)

    osu_win = _window_of("osu!.exe") or "osu!:WindowsForms10.Window.2b.app.0.1ca0192_r6_ad1:osu!.exe"
    vts_win = _window_of("VTube Studio.exe") or "VTube Studio:UnityWndClass:VTube Studio.exe"

    existing_scenes = {s["sceneName"] for s in obs("GetSceneList")["scenes"]}

    def scene(name):
        if name not in existing_scenes:
            obs("CreateScene", sceneName=name)
            existing_scenes.add(name)
        # пересобираем сцену с нуля, чтобы повторный запуск давал ту же раскладку
        for it in obs("GetSceneItemList", sceneName=name)["sceneItems"]:
            obs("RemoveSceneItem", sceneName=name, sceneItemId=it["sceneItemId"])

    def place(scene_name, source, kind, settings, rect=(0, 0, 1920, 1080)):
        # список берём заново: источник, убранный из всех сцен, OBS удаляет сам
        if source in {i["inputName"] for i in obs("GetInputList")["inputs"]}:
            obs("SetInputSettings", inputName=source, inputSettings=settings, overlay=True)
            item = obs("CreateSceneItem", sceneName=scene_name, sourceName=source)["sceneItemId"]
        else:
            item = obs("CreateInput", sceneName=scene_name, inputName=source, inputKind=kind,
                       inputSettings=settings, sceneItemEnabled=True)["sceneItemId"]
        x, y, w, h = rect
        obs("SetSceneItemTransform", sceneName=scene_name, sceneItemId=item, sceneItemTransform={
            "positionX": x, "positionY": y, "boundsType": "OBS_BOUNDS_SCALE_INNER",
            "boundsWidth": w, "boundsHeight": h, "boundsAlignment": 0})
        return item

    # reroute_audio — «Управлять звуком через OBS»: голос Али, который играет страница, идёт в стрим
    browser = lambda url: {"url": url, "width": 1920, "height": 1080, "fps": 60, "fps_custom": True,
                           "css": "", "shutdown": False, "restart_when_active": False, "reroute_audio": True}
    vts = {"capture_mode": "window", "window": vts_win, "priority": 2, "allow_transparency": True,
           "capture_cursor": False}
    osu = {"capture_mode": "window", "window": osu_win, "priority": 2, "capture_cursor": False}

    def vts_item(scene_name):
        # на весь холст по высоте, верх окна (там кнопки интерфейса VTube Studio) обрезан:
        # моделька стоит внизу справа, в кадр попадает только она
        item = place(scene_name, "VTube Studio", "game_capture", vts)
        for _ in range(40):  # размер окна известен, только когда захват подцепился к VTube Studio
            t = obs("GetSceneItemTransform", sceneName=scene_name, sceneItemId=item)["sceneItemTransform"]
            if t["sourceWidth"] > 0:
                break
            time.sleep(0.25)
        w, h = t["sourceWidth"] or 1920, t["sourceHeight"] or 1080
        k = 1080 / h
        top = int(h * VTS_CROP_TOP)
        obs("SetSceneItemTransform", sceneName=scene_name, sceneItemId=item, sceneItemTransform={
            "boundsType": "OBS_BOUNDS_NONE", "alignment": 5, "scaleX": k, "scaleY": k, "cropTop": top,
            "positionX": 1920 - w * k, "positionY": top * k})  # прижато к правому краю

    game = "Alya — игра"
    scene(game)
    place(game, "Оверлей Али", "browser_source", browser(f"{OVERLAY}/?voice=1"))
    place(game, "osu!", "game_capture", osu, GAME_RECT)
    vts_item(game)

    for name, mode in (("Alya — скоро", "starting"), ("Alya — отошла", "brb"), ("Alya — конец", "end")):
        scene(name)
        place(name, f"Заставка: {mode}", "browser_source", browser(f"{OVERLAY}/scene.html?mode={mode}&voice=1"))
        vts_item(name)

    # пустая сцена, с которой OBS создаёт новую коллекцию, не нужна (в старой коллекции чужое не трогаем)
    for leftover in (existing_scenes if fresh else set()) - {game, "Alya — скоро", "Alya — отошла", "Alya — конец"}:
        try:
            obs("RemoveScene", sceneName=leftover)
        except RuntimeError:
            pass
    # звук без «Звука рабочего стола», чтобы Discord и прочее не попадали в стрим:
    # осу — захват звука только этого приложения, голос Али — через страницу оверлея (выше),
    # микрофон добавлен выключенным (включишь, когда будешь говорить сам)
    for name in (game, "Alya — скоро", "Alya — отошла", "Alya — конец"):
        place(name, "Звук osu!", "wasapi_process_output_capture", {"window": osu_win, "priority": 2})
        place(name, "Микрофон", "wasapi_input_capture", {"device_id": "default"})
    obs("SetInputMute", inputName="Микрофон", inputMuted=True)
    obs("SetCurrentProgramScene", sceneName="Alya — скоро")
    print("сцены: Alya — скоро, Alya — игра, Alya — отошла, Alya — конец")
    print(f"захват osu!: {osu_win.split(':')[0]} | VTube Studio: {vts_win.split(':')[0]}")
    return obs


def screenshot(obs: OBS, scene: str, path: str):
    """Снимок сцены в PNG — проверить раскладку."""
    d = obs("GetSourceScreenshot", sourceName=scene, imageFormat="png", imageWidth=1280, imageHeight=720)
    data = d["imageData"].split(",", 1)[1]
    with open(path, "wb") as f:
        f.write(base64.b64decode(data))
