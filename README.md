# SG-ALNS: Segment-Level Graph RL for ALNS Configuration in the VRPTW

Code, data, trained models and raw results of the paper
*SG-ALNS: Segment-Level Graph RL for ALNS Configuration in the VRPTW*
(D. Q. Vo, S. X. Tran, U. T. Nguyen, Vinh University). Vietnamese notes: [README_vi.md](README_vi.md).

SG-ALNS configures Adaptive Large Neighbourhood Search once per segment of iterations: a graph policy (GINE,
trained with PPO) jointly selects the destroy/repair operators, the destroy scale and the annealing temperature, and
can fall back to the classical roulette wheel. The paper evaluates whether the learned policies exploit the search
state, with six controls (policy inspection with counterfactual states, training-seed replication, tuned baselines,
a dynamic-headroom search, equal run time, and transfer across budgets and instance sizes).

## Installation
```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # Python 3.14, pinned versions
```

## Layout
```
sgalns/            core ALNS (Numba), segment-level MDP, graph policy, PPO
scripts/           train.py, evaluate.py, tune.py, probe_state.py, verify_bks.py
scripts/paper/     generate every number, table and figure of the paper from the raw results
configs/           training configuration (default.yaml)
data/              Solomon and Gehring-Homberger instances, BKS table
results_paper/     raw results (CSV), logs and trained models (*/last.pt) used in the paper
paper/             LaTeX sources (IEEE: main.tex, main_anonymous.tex, supplementary.tex; LNCS: *_lncs.tex)
tests/             kernel and environment tests
run_*.sh           experiment pipelines (run_all.sh, then run_revision*.sh)
make_paper.sh      regenerate paper/generated/ from the results and compile the PDFs
```

## Reproducing
```bash
.venv/bin/python scripts/train.py --name main                       # train one policy (seed 0)
.venv/bin/python scripts/evaluate.py --data data/solomon --T 5000 --seeds 5 \
    --methods rw sg:main=results/main/last.pt --out results/e1_solomon.csv
RESULTS=$PWD/results_paper/results LOGS=$PWD/results_paper/logs ./make_paper.sh   # rebuild the paper from the shipped results
nohup ./run_all.sh > logs/run_all.log 2>&1 &                         # full pipeline (~8 h on an 8-core Apple M1)
```
All numbers in `paper/` are generated from `results_paper/`; none is typed by hand.
