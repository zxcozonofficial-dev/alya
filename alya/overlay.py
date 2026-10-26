"""Оверлей для OBS: маленький HTTP-сервер на localhost, страницы из папки overlay/ и поток событий /events (SSE).

В OBS добавляется «Браузер» с адресом http://localhost:24080/ — страница сама получает
реплики Али (субтитры, эмоция), сообщения чата Twitch и события игры. Данные о карте
(название, точность, комбо, pp) страница берёт прямо из tosu.
"""
from __future__ import annotations

import json
from collections import deque
import queue
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "overlay"
STICKY = ("requests", "emotion", "channel")  # последнее такое событие получает и только что открытая страница
HISTORY = 20  # столько последних сообщений чата видит только что открытая страница


class Overlay:
    def __init__(self, port: int = 24080):
        self.port = port
        self._clients: list[queue.Queue] = []
        self._sticky: dict[str, str] = {}
        self._chat: deque[str] = deque(maxlen=HISTORY)
        self._lock = threading.Lock()
        hub = self

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path.split("?")[0] != "/events":
                    return super().do_GET()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                q = hub._subscribe()
                try:
                    while True:
                        try:
                            msg = q.get(timeout=15)
                        except queue.Empty:
                            msg = None
                        # пустой комментарий раз в 15 с держит соединение живым
                        self.wfile.write(b": ping\n\n" if msg is None else f"data: {msg}\n\n".encode())
                        self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError, OSError):
                    pass
                finally:
                    hub._unsubscribe(q)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), partial(Handler, directory=str(ROOT)))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def _subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=500)
        with self._lock:
            self._clients.append(q)
            # новая страница (смена сцены в OBS, перезагрузка) сразу видит очередь заказов и недавний чат
            for msg in [*self._sticky.values(), *self._chat]:
                q.put_nowait(msg)
        return q

    def _unsubscribe(self, q):
        with self._lock:
            if q in self._clients:
                self._clients.remove(q)

    def send(self, kind: str, **data):
        msg = json.dumps({"type": kind, **data}, ensure_ascii=False)
        with self._lock:
            if kind in STICKY:
                self._sticky[kind] = msg
            elif kind == "chat":
                self._chat.append(msg)
            for q in self._clients:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass
