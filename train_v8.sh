#!/usr/bin/env bash
# v8: дообучение v7: трудные места + слайдеры с повторами, штраф за срыв 0.5, снова EZ и HT, затем сравнение
set -e
O="$LOCALAPPDATA/osu!"
PY=./.venv/Scripts/python
MYTIME="$O/Songs/2337338 bo en - My Time/bo en - My Time (gwb) [regret].osu"
date > logs/v8_start
$PY main.py ppo --maps "$O/Songs" --limit 600 --init ckpt/v7_ppo.pt --ref ckpt/v4_bc.pt --out ckpt/v8_ppo.pt --envs 512 --updates 400 --smooth 0.3 --hard 0.5 --cp-cost 0.5 --mods NM DT DT HR HRDT EZ EZDT HT EZHT > logs/v8_ppo.log 2>&1
PYTHONIOENCODING=utf-8 $PY tools_eval.py v6=ckpt/v6_ppo.pt v8=ckpt/v8_ppo.pt > logs/v8_eval.log 2>&1
for m in v6 v8; do
  echo "== $m My Time HDDT" >> logs/v8_eval.log
  PYTHONIOENCODING=utf-8 $PY main.py export --model ckpt/${m}_ppo.pt --map "$MYTIME" --mods HDDT --out replays/mytime_${m}.osr 2>&1 | tail -1 >> logs/v8_eval.log
done
date > logs/v8_done
