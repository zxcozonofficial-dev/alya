#!/usr/bin/env bash
# v9: дообучение v8 жать вовремя: шаги по нотам + штраф за ошибку по времени, затем сравнение
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
MYTIME="$O/Songs/2337338 bo en - My Time/bo en - My Time (gwb) [regret].osu"
date > logs/v9_start
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/v8_ppo.pt --ref ckpt/v4_bc.pt --out ckpt/v9_ppo.pt --envs 512 --updates 300 --smooth 0.3 --hard 0.5 --cp-cost 0.5 --timing 0.3 --mods NM DT DT HR HRDT EZ EZDT HT EZHT > logs/v9_ppo.log 2>&1
PYTHONIOENCODING=utf-8 $PY tools_eval.py v8=ckpt/v8_ppo.pt v9=ckpt/v9_ppo.pt > logs/v9_eval.log 2>&1
for m in v8 v9; do
  echo "== $m My Time HDDT" >> logs/v9_eval.log
  PYTHONIOENCODING=utf-8 $PY main.py export --model ckpt/${m}_ppo.pt --map "$MYTIME" --mods HDDT --out replays/mytime_${m}.osr 2>&1 | tail -1 >> logs/v9_eval.log
done
date > logs/v9_done
