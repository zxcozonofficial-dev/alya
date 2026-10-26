"""Чтение чата Twitch без аккаунта и токена: анонимный вход в IRC (ник justinfanNNNN, только чтение).

Писать в чат так нельзя — и не нужно: Аля отвечает голосом на стриме.
Каждое сообщение уходит в on_message(ник, текст, цвет ника).
"""
from __future__ import annotations

import random
import re
import socket
import threading
import time

HOST, PORT = "irc.chat.twitch.tv", 6667
PRIVMSG = re.compile(r"^(?:@(?P<tags>\S+) )?:(?P<nick>[^!]+)!\S+ PRIVMSG #\S+ :(?P<text>.*)$")


class TwitchChat:
    def __init__(self, channel: str, on_message, on_status=lambda s: None):
        self.channel = channel.lower().lstrip("#")
        self.on_message, self.on_status = on_message, on_status
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        delay = 2
        while True:
            try:
                with socket.create_connection((HOST, PORT), timeout=360) as s:
                    nick = f"justinfan{random.randint(10000, 99999)}"
                    s.sendall(f"CAP REQ :twitch.tv/tags\r\nPASS SCHMOOPIIE\r\nNICK {nick}\r\n"
                              f"JOIN #{self.channel}\r\n".encode())
                    self.on_status(f"читаю чат twitch.tv/{self.channel}")
                    delay = 2
                    buf = b""
                    while True:
                        data = s.recv(4096)
                        if not data:
                            raise ConnectionError("Twitch закрыл соединение")
                        buf += data
                        *lines, buf = buf.split(b"\r\n")
                        for raw in lines:
                            line = raw.decode("utf-8", "replace")
                            if line.startswith("PING"):
                                s.sendall(b"PONG :tmi.twitch.tv\r\n")
                                continue
                            m = PRIVMSG.match(line)
                            if m:
                                tags = dict(t.split("=", 1) for t in (m["tags"] or "").split(";") if "=" in t)
                                name = tags.get("display-name") or m["nick"]
                                self.on_message(name, m["text"].strip(), tags.get("color") or "")
            except Exception as e:
                self.on_status(f"чат Twitch отвалился ({type(e).__name__}), переподключаюсь через {delay} с")
                time.sleep(delay)
                delay = min(delay * 2, 60)
