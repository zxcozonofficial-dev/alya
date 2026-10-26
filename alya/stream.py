"""Запуск стрима одной командой: tosu → бот (руки) в своём окне → Аля (мозг, голос, оверлей) в этом окне.

Настройки — в stream.json рядом с main.py (создаётся при первом запуске). Если tosu или бот уже
запущены, второй экземпляр не поднимается: два бота жали бы клавиши одновременно.
Когда закрываешь Алю (/выход или Ctrl+C), бот, запущенный отсюда, тоже останавливается.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
from pathlib import Path

CONFIG = Path("stream.json")
DEFAULT = {
    "tosu": "tosu.exe",      # путь к tosu.exe — поправь под себя
    "twitch": "",            # канал, чат которого она читает; пусто — без чата
    "model": "ckpt/alya.pt",
    "keys": "zx",
    "acc": 0.96,
    "vts": True,
    "voice": "en-US-AnaNeural",
    "lang": "en",
    "autopilot": True,       # автопилот включён сразу при запуске
}


def load_config() -> dict:
    if not CONFIG.exists():
        CONFIG.write_text(json.dumps(DEFAULT, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"создал {CONFIG} с настройками по умолчанию — канал Twitch впиши туда в \"twitch\"")
    return {**DEFAULT, **json.loads(CONFIG.read_text(encoding="utf-8"))}


def _processes(subcommand: str) -> list[str]:
    """Командные строки запущенных процессов python … main.py <subcommand>."""
    ps = ("Get-CimInstance Win32_Process | Where-Object { $_.Name -like 'python*' -and "
          "$_.CommandLine -match 'main\\.py\"?\\s+" + subcommand + "(\\s|$)' } | ForEach-Object { $_.CommandLine }")
    out = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True,
                         encoding="utf-8", errors="replace")
    return [line for line in out.stdout.splitlines() if line.strip()]


def _port_open(port: int) -> bool:
    with socket.socket() as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def run():
    cfg = load_config()
    new_console = subprocess.CREATE_NEW_CONSOLE
    if _processes("chat"):
        print("Аля уже запущена в другом окне (main.py chat) — закрой её там (/выход) и запусти заново")
        return

    if _port_open(24050):
        print("tosu уже запущен")
    else:
        tosu = Path(cfg["tosu"])
        if not tosu.exists():
            print(f"не нашёл tosu: {tosu} — поправь путь в {CONFIG}")
            return
        subprocess.Popen([str(tosu)], cwd=tosu.parent, creationflags=new_console)
        print("запускаю tosu…", end="", flush=True)
        for _ in range(60):
            if _port_open(24050):
                break
            time.sleep(0.5)
        print(" готов" if _port_open(24050) else " не отвечает, иду дальше")

    bot = None
    if _processes("live"):
        print("бот уже запущен — второй не запускаю")
    else:
        cmd = [sys.executable, "main.py", "live", "--model", cfg["model"], "--keys", cfg["keys"],
               "--acc", str(cfg["acc"])]
        # строкой, а не списком: cmd не понимает кавычки, экранированные обратной косой чертой;
        # внешние кавычки после /k cmd снимает сам. Окно остаётся открытым, если бот упадёт
        bot = subprocess.Popen(f'cmd /k "title Alya live & {subprocess.list2cmdline(cmd)}"',
                               creationflags=new_console)
        print("бот запущен в отдельном окне (F8 в игре — стоп)")

    from .brain import chat
    try:
        chat(lang=cfg["lang"], voice=cfg["voice"] or None, vts=bool(cfg["vts"]), game="alya",
             twitch=cfg["twitch"] or None, overlay=24080, autopilot=bool(cfg["autopilot"]))
    finally:
        # бот закрывается вместе с Алей, даже если его запускал прошлый запуск (тогда bot=None):
        # гасим окно «Alya live» (cmd /k держит его открытым) вместе с ботом внутри
        ps = ("Get-CimInstance Win32_Process | Where-Object { ($_.Name -eq 'cmd.exe' -and $_.CommandLine -like "
              "'*title Alya live*') -or ($_.Name -like 'python*' -and $_.CommandLine -match 'main\\.py\"?\\s+live') } "
              "| ForEach-Object { $_.ProcessId }")
        pids = subprocess.run(["powershell", "-NoProfile", "-Command", ps], capture_output=True, text=True).stdout.split()
        if bot:
            pids.append(str(bot.pid))
        for pid in set(pids):
            subprocess.run(["taskkill", "/PID", pid, "/T", "/F"], capture_output=True)
        if pids:
            print("бот остановлен")
