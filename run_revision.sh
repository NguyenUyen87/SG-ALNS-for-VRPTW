#!/usr/bin/env bash
# Additional experiments requested by the review (Phan_bien_SG-ALNS.pdf). Runs after run_all.sh; resumable.
#   R1  CPU cost of tuning (re-run of the two grids with per-run CPU timing; results are identical)   [review 2.5]
#   R2  tuned static over all 285 constant actions (leakage check of the 30-configuration grid)      [review 3.1]
#   R3  training seeds 3 and 4 of SG-ALNS (5 seeds in total), with measured CPU time                 [review 3.3]
#   R4  segment-level Branching Dueling Double DQN (2 seeds): is policy collapse specific to PPO?    [lit.: PPO collapse in DAC]
#   R5  evaluation of R2-R4 on Solomon (T=5000, 5 seeds) and time-matched (T=10000)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-8}
step() { echo "[$(date '+%F %T')] $*"; }
until grep -q ALL_DONE logs/run_all.log 2>/dev/null; do sleep 120; done

for m in static rw full; do
  f=results/.rev_tune_$m.done
  [ -f $f ] || { step "R1/R2 tune $m (timed)"; $PY scripts/tune.py $m > logs/rev_tune_$m.log 2>&1 && touch $f; }
done
FULL=$(cat results/tune_full_best.txt)

train() { name=$1; shift; [ -f results/$name/DONE ] && return 0; step "train $name";
          /usr/bin/time -p $PY -u scripts/train.py --name $name --set "$@" > logs/train_$name.log 2>&1 && touch results/$name/DONE; }
train main_s3 seed=3
train main_s4 seed=4
train bdq     algo=bdq seed=0
train bdq_s1  algo=bdq seed=1

sg() { echo "sg:$1=results/$1/last.pt"; }
step "R5 Solomon T=5000 (5 seeds)"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 5000 --seeds 5 --out results/e1_solomon.csv \
    --methods $(sg main_s3) $(sg main_s4) $(sg bdq) $(sg bdq_s1) $FULL
step "R5 time-matched (T=10000, 5 seeds)"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 10000 --seeds 5 --out results/e5_timematched.csv --methods $FULL
step REVISION_DONE
