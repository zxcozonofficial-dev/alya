@echo off
rem One-click stream: tosu, bot and Alya. Settings are in stream.json.
cd /d "%~dp0"
title Alya
.venv\Scripts\python.exe -X utf8 main.py stream
pause
