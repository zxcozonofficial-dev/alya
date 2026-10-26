#!/usr/bin/env bash
# v4: путь слайдера в наблюдении, строгие слайдеры, ограничение скорости пальца
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
date > logs/v4_start
$PY main.py bc-data --songs "$O/Songs" --replays "$O/Replays" "$O/Data/r" --autoplay-maps data/synth --out data/bc.npz > logs/v4_data.log 2>&1
$PY main.py bc --data data/bc.npz --out ckpt/v4_bc.pt --steps 12000 > logs/v4_bc.log 2>&1
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/v4_bc.pt --ref ckpt/v4_bc.pt --out ckpt/v4_ppo.pt --envs 512 --updates 600 --mods NM DT HR HRDT EZ EZDT > logs/v4_ppo.log 2>&1
date > logs/v4_done
