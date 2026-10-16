#!/usr/bin/env bash
# Third review (release/.../Phan_bien_SG-ALNS.docx): instance-class oracle, neutral-prior seeds, long training.
#   C1  the 15 best constant actions of the 285-action grid on all Solomon instances (T=5000, 5 seeds) -> e9_class.csv
#   C2  neutral prior + constant entropy (as main_ent) with training seeds 1 and 2                    [review 3.2]
#   C3  seed 0 trained 5x longer (12,000 episodes = 1,500 PPO updates)                                [review 3.1]
#   C4  Solomon evaluation and state probe of C2-C3 (written to e1_rev3.csv, merged into e1 at the end)
set -e
cd "$(dirname "$0")"
PY=.venv/bin/python
RP=results_paper/results
LP=results_paper/logs
step() { echo "[$(date '+%F %T')] $*"; }
train() { name=$1; shift
  if [ ! -f results/$name/DONE ]; then
    step "train $name"
    /usr/bin/time -p $PY -u scripts/train.py --name $name --set "$@" > logs/train_$name.log 2>&1 && touch results/$name/DONE
  fi
  ln -sfn "$PWD/results/$name" $RP/$name; cp logs/train_$name.log $LP/; }
train main_long seed=0 ppo.episodes=12000 &
train ent_s1 seed=1 ppo.ent_end=0.01 env.prior=false &
train ent_s2 seed=2 ppo.ent_end=0.01 env.prior=false &
step "C1 class oracle candidates on Solomon"
$PY scripts/evaluate.py --workers 3 --data data/solomon --T 5000 --seeds 5 --out $RP/e9_class.csv --methods $(cat $RP/class_candidates.txt)
wait
for n in main_long ent_s1 ent_s2; do [ -f results/$n/DONE ]; done
sg() { echo "sg:$1=$RP/$1/last.pt"; }
step "C4 Solomon T=5000"
$PY scripts/evaluate.py --workers 8 --data data/solomon --T 5000 --seeds 5 --out $RP/e1_rev3.csv --methods $(sg main_long) $(sg ent_s1) $(sg ent_s2)
step "C4 probe"
$PY scripts/probe_state.py --models main_long=$RP/main_long/last.pt ent_s1=$RP/ent_s1/last.pt ent_s2=$RP/ent_s2/last.pt main_ent=$RP/main_ent/last.pt \
    --out $RP/probe_state_rev3.csv > logs/probe_state_rev3.log 2>&1
step REVISION3_DONE
