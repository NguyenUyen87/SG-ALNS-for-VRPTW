#!/usr/bin/env bash
# Second-review experiments without the reward-sensitivity variants (lambda = 1, 25 dropped by the authors).
# Continues run_revision2.sh: waits for the running seed-3 training (pid 44571), then seed 4, the 285-action grid,
# the Solomon evaluations and the T=200 study. Resumable.
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-6}
RP=results_paper/results
LP=results_paper/logs
step() { echo "[$(date '+%F %T')] $*"; }
while kill -0 44571 2>/dev/null; do sleep 30; done
grep -q "\[ 300/300\]" logs/train_main_s3.log && touch results/main_s3/DONE
train() { name=$1; shift
  if [ ! -f results/$name/DONE ]; then
    step "train $name"
    /usr/bin/time -p $PY -u scripts/train.py --name $name --set "$@" > logs/train_$name.log 2>&1 && touch results/$name/DONE
  fi
  ln -sfn "$PWD/results/$name" $RP/$name; cp logs/train_$name.log $LP/; }
train main_s3 seed=3
train main_s4 seed=4
if [ ! -f results/.rev_tune_full.done ]; then
  step "tune full (285 actions)"; $PY scripts/tune.py full > logs/rev_tune_full.log 2>&1 && touch results/.rev_tune_full.done
fi
cp results/tune_full.csv results/tune_full_best.txt $RP/; cp logs/rev_tune_full.log $LP/
FULL=$(cat $RP/tune_full_best.txt)
sg() { echo "sg:$1=$RP/$1/last.pt"; }
step "Solomon T=5000"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 5000 --seeds 5 --out $RP/e1_solomon.csv     --methods $(sg main_s3) $(sg main_s4) $FULL
step "time-matched T=10000 ($FULL)"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 10000 --seeds 5 --out $RP/e5_timematched.csv --methods $FULL
step "short horizon T=200"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 200 --seeds 3 --out $RP/e8_short.csv     --methods rw static:10-3-0 $(sg main) $(sg iter_gnn) $(sg iter_mlp)
step REVISION2_DONE
