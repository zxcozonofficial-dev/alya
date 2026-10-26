#!/usr/bin/env bash
# v6: v5 с штрафом за углы в 6 раз сильнее, затем сравнение
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
date > logs/v6_start
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/v5_ppo.pt --ref ckpt/v4_bc.pt --out ckpt/v6_ppo.pt --envs 512 --updates 300 --smooth 0.3 --mods NM DT HR HRDT EZ EZDT > logs/v6_ppo.log 2>&1
PYTHONIOENCODING=utf-8 $PY tools_eval.py v5=ckpt/v5_ppo.pt v6=ckpt/v6_ppo.pt > logs/v6_eval.log 2>&1
date > logs/v6_done
