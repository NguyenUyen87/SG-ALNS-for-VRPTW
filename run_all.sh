#!/usr/bin/env bash
# Full experimental scenario of the paper (resumable: finished steps are skipped).
# Usage: ./run_all.sh  (8-core machine; log in logs/run_all.log)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
W=${WORKERS:-8}
mkdir -p logs results
step() { echo "[$(date '+%F %T')] $*"; }

# ------------------------------------------------------------------ 0) tests
step tests
$PY tests/test_core.py > logs/test_core.log 2>&1
$PY tests/test_env.py > logs/test_env.log 2>&1

# ------------------------------------------------------------------ 1) offline tuning / headroom studies
for m in static rw headroom dynamic; do
  f=results/.tune_$m.done
  [ -f $f ] || { step "tune $m"; $PY scripts/tune.py $m > logs/tune_$m.log 2>&1 && touch $f; }
done
ST=$(cat results/tune_static_best.txt)        # tuned static (paper: static:10-3-0)
RWT=$(cat results/tune_rw_best.txt)           # tuned RW     (paper: rwt:100-0.3-0.4-1.0)
CONST=$(sed -n 1p results/dyn_headroom_best.txt)
SCHED=$(sed -n 2p results/dyn_headroom_best.txt)

# ------------------------------------------------------------------ 2) training (2400 episodes each)
train() { name=$1; shift; [ -f results/$name/DONE ] && return 0; step "train $name";
          $PY -u scripts/train.py --name $name --set "$@" > logs/train_$name.log 2>&1 && touch results/$name/DONE; }
ITER="env.K=1 env.T=100 env.p_main=1.0 env.use_a2=false env.use_a3=false env.expert=false"
train main      seed=0
train main_s1   seed=1
train main_s2   seed=2
train no_expert env.expert=false
train no_a1     env.use_a1=false
train no_a2     env.use_a2=false
train no_a3     env.use_a3=false
train no_dyn    env.dyn=false
train mlp       model.encoder=mlp
train iter_mlp  $ITER model.encoder=mlp
train iter_gnn  $ITER ppo.max_minutes=57          # paper: stopped after 57 min (2.6x the SG-ALNS budget)
train ctx       env.ctx=true "env.sizes=[50,100,200]" "env.T_choices=[1000,2500,5000]"
train main_ent  ppo.ent_end=0.01 env.prior=false  # collapse sensitivity (review M5)

sg() { echo "sg:$1=results/$1/last.pt"; }
EV="$PY scripts/evaluate.py --workers $W"

# ------------------------------------------------------------------ 3) evaluation
step "E1 Solomon T=5000 (5 seeds): baselines, SG-ALNS (3 training seeds, sampled), SG-ALNS-C, review"
$EV --data data/solomon --T 5000 --seeds 5 --out results/e1_solomon.csv --methods \
    uni rw $RWT $ST $(sg main) sgs:main=results/main/last.pt $(sg main_s1) $(sg main_s2) $(sg ctx) \
    $(sg main_ent) $SCHED $CONST
step "E1b ablations (Solomon T=5000, 5 seeds)"
$EV --data data/solomon --T 5000 --seeds 5 --out results/e1_solomon.csv --methods \
    $(sg no_expert) $(sg no_a1) $(sg no_a2) $(sg no_a3) $(sg no_dyn) $(sg mlp)
step "E4 iteration-level RL baselines (3 seeds)"
$EV --data data/solomon --T 5000 --seeds 3 --out results/e4_iter.csv --methods $(sg iter_gnn) $(sg iter_mlp)
step "E5 time-matched baselines (T=10000, 5 seeds)"
$EV --data data/solomon --T 10000 --seeds 5 --out results/e5_timematched.csv --methods rw $ST $RWT
step "E3 budgets (T=1000, 20000; 3 seeds)"
$EV --data data/solomon --T 1000 20000 --seeds 3 --out results/e3_budget.csv --methods uni rw $RWT $ST $(sg main) $(sg ctx)
step "E2 Gehring-Homberger 200 (3 seeds)"
$EV --data data/homberger_200 --T 5000 --seeds 3 --out results/e2_hg200.csv --methods uni rw $RWT $ST $(sg main) $(sg ctx)
step "E6 Gehring-Homberger 400 (3 seeds)"
$EV --data data/homberger_400 --T 5000 --seeds 3 --out results/e6_hg400.csv --methods rw $RWT $ST $(sg main) $(sg ctx)

# ------------------------------------------------------------------ 4) BKS check and report
step "verify BKS + report"
$PY scripts/verify_bks.py > logs/verify_bks.log 2>&1
$PY scripts/report.py > logs/report.log 2>&1
step ALL_DONE
