#!/usr/bin/env bash
# v7: дообучение v6 на трудных местах (стримы, двойки 1/8) со штрафом за сорванный слайдер, затем сравнение
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
MYTIME="$O/Songs/2337338 bo en - My Time/bo en - My Time (gwb) [regret].osu"
date > logs/v7_start
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/v6_ppo.pt --ref ckpt/v4_bc.pt --out ckpt/v7_ppo.pt --envs 512 --updates 400 --smooth 0.3 --hard 0.5 --cp-cost 0.3 --mods NM DT DT HR HRDT EZDT > logs/v7_ppo.log 2>&1
PYTHONIOENCODING=utf-8 $PY tools_eval.py v6=ckpt/v6_ppo.pt v7=ckpt/v7_ppo.pt > logs/v7_eval.log 2>&1
for m in v6 v7; do
  echo "== $m My Time HDDT" >> logs/v7_eval.log
  PYTHONIOENCODING=utf-8 $PY main.py export --model ckpt/${m}_ppo.pt --map "$MYTIME" --mods HDDT --out replays/mytime_${m}.osr 2>&1 | tail -1 >> logs/v7_eval.log
done
date > logs/v7_done
