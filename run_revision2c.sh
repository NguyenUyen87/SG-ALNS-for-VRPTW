#!/usr/bin/env bash
# Reward sensitivity (second review 4.3): lambda = 1 and lambda = 25 (seed 0), trained in parallel with
# run_revision2b.sh, evaluated on Solomon (T=5000, 5 seeds) into e1_lam.csv, probed, and merged into e1_solomon.csv
# once run_revision2b.sh has finished writing it. Resumable.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-4}
RP=results_paper/results
LP=results_paper/logs
step() { echo "[$(date '+%F %T')] $*"; }
train() { name=$1; shift
  if [ ! -f results/$name/DONE ]; then
    step "train $name"
    /usr/bin/time -p $PY -u scripts/train.py --name $name --set "$@" > logs/train_$name.log 2>&1 && touch results/$name/DONE
  fi
  ln -sfn "$PWD/results/$name" $RP/$name; cp logs/train_$name.log $LP/; }
train lam1  env.lam_nv=1 &
train lam25 env.lam_nv=25 &
wait
[ -f results/lam1/DONE ] && [ -f results/lam25/DONE ]
sg() { echo "sg:$1=$RP/$1/last.pt"; }
step "Solomon T=5000 (lambda variants)"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 5000 --seeds 5 --out $RP/e1_lam.csv --methods $(sg lam1) $(sg lam25)
step "probe lambda variants"
$PY scripts/probe_state.py --models lam1=$RP/lam1/last.pt lam25=$RP/lam25/last.pt --out $RP/probe_state_lam.csv > logs/probe_state_lam.log 2>&1
until grep -q REVISION2_DONE logs/run_revision2b.log 2>/dev/null; do sleep 60; done
if ! grep -q "^sg:lam1," $RP/e1_solomon.csv; then
  step "merge e1_lam.csv into e1_solomon.csv"; tail -n +2 $RP/e1_lam.csv >> $RP/e1_solomon.csv
fi
step REVISION2C_DONE
