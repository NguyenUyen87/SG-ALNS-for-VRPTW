#!/usr/bin/env bash
# Experiments requested by the second review (paper/Nhan_xet_chi_tiet_bai_bao_SG-ALNS.docx). Resumable; runs next
# to run_all.sh. Models are trained into results/ (v2 code, bit-identical to the paper runs) and evaluated into
# results_paper/results/, the result set the paper is built from.
#   Q1  training seeds 3 and 4 of SG-ALNS (5 seeds in total)                         [review: priority 1]
#   Q2  tuned static over all 285 constant actions (timed)                          [review 4.1]
#   Q3  reward sensitivity: lambda = 1 and lambda = 25 (seed 0)                       [review 4.3]
#   Q4  Solomon evaluation of Q1-Q3 (T=5000, 5 seeds)
#   Q5  iteration-level agents near their training horizon (T=200, 3 seeds)         [review 4.5]
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-6}
RP=results_paper/results
LP=results_paper/logs
step() { echo "[$(date '+%F %T')] $*"; }

train() { name=$1; shift
  if [ ! -f results/$name/DONE ]; then
    step "train $name"
    /usr/bin/time -p $PY -u scripts/train.py --name $name --set "$@" > logs/train_$name.log 2>&1 && touch results/$name/DONE
  fi
  ln -sfn "$PWD/results/$name" $RP/$name; cp logs/train_$name.log $LP/; }
train main_s3 seed=3
train main_s4 seed=4

if [ ! -f results/.rev_tune_full.done ]; then
  step "Q2 tune full (285 actions)"; $PY scripts/tune.py full > logs/rev_tune_full.log 2>&1 && touch results/.rev_tune_full.done
fi
cp results/tune_full.csv results/tune_full_best.txt $RP/; cp logs/rev_tune_full.log $LP/
FULL=$(cat $RP/tune_full_best.txt)

train lam1  env.lam_nv=1
train lam25 env.lam_nv=25

sg() { echo "sg:$1=$RP/$1/last.pt"; }
step "Q4 Solomon T=5000"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 5000 --seeds 5 --out $RP/e1_solomon.csv \
    --methods $(sg main_s3) $(sg main_s4) $(sg lam1) $(sg lam25) $FULL
step "Q4 time-matched T=10000 ($FULL)"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 10000 --seeds 5 --out $RP/e5_timematched.csv --methods $FULL
step "Q5 short horizon T=200"
$PY scripts/evaluate.py --workers $W --data data/solomon --T 200 --seeds 3 --out $RP/e8_short.csv \
    --methods rw static:10-3-0 $(sg main) $(sg iter_gnn) $(sg iter_mlp)
step REVISION2_DONE
