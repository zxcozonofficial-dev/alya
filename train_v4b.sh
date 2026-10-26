#!/usr/bin/env bash
# дообучение v4: ещё 600 итераций PPO с v4_ppo (якорь стиля — v4_bc), затем сравнение
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
date > logs/v4b_start
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/v4_ppo.pt --ref ckpt/v4_bc.pt --out ckpt/v4b_ppo.pt --envs 512 --updates 600 --mods NM DT HR HRDT EZ EZDT > logs/v4b_ppo.log 2>&1
PYTHONIOENCODING=utf-8 $PY tools_eval.py v3=ckpt/ppo.pt v4=ckpt/v4_ppo.pt v4b=ckpt/v4b_ppo.pt > logs/v4b_eval.log 2>&1
date > logs/v4b_done
