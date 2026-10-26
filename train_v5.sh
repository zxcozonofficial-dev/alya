#!/usr/bin/env bash
# v5: дообучение alya.pt со штрафом за резкие углы аима и выбором клавиш правилом, затем сравнение
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
date > logs/v5_start
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/alya.pt --ref ckpt/v4_bc.pt --out ckpt/v5_ppo.pt --envs 512 --updates 400 --smooth 0.05 --mods NM DT HR HRDT EZ EZDT > logs/v5_ppo.log 2>&1
PYTHONIOENCODING=utf-8 $PY tools_eval.py v4b=ckpt/alya.pt v5=ckpt/v5_ppo.pt > logs/v5_eval.log 2>&1
date > logs/v5_done
