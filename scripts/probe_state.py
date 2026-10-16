"""Does a trained policy react to its state?  (second review, 4.2 / priority 4)

For each policy, run greedy SG-ALNS on 12 Solomon instances (two per class, seed 0, T=5000). At every decision the
policy is also queried on counterfactual states:
  natural   the states actually visited during the run (distance to the first decision of the run)
  global    the six search-progress features (t/T, tau/tau0, stagnation, two gaps, fleet ratio) and the operator
            success rates redrawn uniformly over their observed range, 16 draws per decision
  dynamic   the six dynamic node features permuted across nodes (destroys their link to the graph), 4 draws
  nodyn     the six dynamic node features set to zero
  other     the graph of another instance (same decision index), 4 draws
Reported per policy and probe: share of queries whose greedy action differs from the actual greedy action, mean
total-variation distance between the joint action distributions, and the mean probability of the actual greedy action.

python scripts/probe_state.py --models main=results_paper/results/main/last.pt ... --out results_paper/results/probe_state.csv
"""
import argparse, glob, os, sys
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgalns.instance import read_solomon
from sgalns.alns import Engine
from sgalns.env import build_obs, decode_action
from sgalns.policy import load_policy, collate

G_PROG = 6                         # first six global features: progress, temperature, stagnation, gaps, fleet


def joint(logits):
    p1, p2, p3 = (torch.softmax(l, -1) for l in logits)
    return (p1[:, :, None, None] * p2[:, None, :, None] * p3[:, None, None, :]).reshape(len(p1), -1)


def query(net, obs):
    with torch.no_grad():
        logits, _ = net(collate(obs))
    return joint(logits).numpy()


def run(net, cfg, inst, others, rng, T=5000, K=50):
    eng = Engine(inst, 0, T, None)
    first, recs, visited = None, [], []
    gl_lo, gl_hi = None, None
    while eng.iter < T:
        o = build_obs(eng, cfg.get("dyn", True), cfg.get("ctx", False))
        P = query(net, [o])[0]
        a = int(P.argmax())
        if first is None:
            first = P
        visited.append(o["g"].copy())
        lo = np.min(visited, 0); hi = np.max(visited, 0)
        probes = {"natural": [o], "global": [], "dynamic": [], "nodyn": [], "other": []}
        for _ in range(16):
            q = dict(o); g = o["g"].copy()
            nrate = len(g) - G_PROG - (2 if cfg.get("ctx", False) else 0)
            g[0] = rng.uniform(0, 1); g[1] = rng.uniform(0.003, 3.0); g[2] = rng.uniform(0, 1)
            g[3] = rng.uniform(0, 0.1); g[4] = rng.uniform(0, 0.2); g[5] = rng.uniform(0.5, 1.0)
            g[G_PROG:G_PROG + nrate] = rng.uniform(0, 1, nrate)
            q["g"] = g.astype(np.float32); probes["global"].append(q)
        for _ in range(4):
            q = dict(o); x = o["x"].copy(); x[:, 6:] = x[rng.permutation(len(x)), 6:]; q["x"] = x
            probes["dynamic"].append(q)
        q = dict(o); x = o["x"].copy(); x[:, 6:] = 0.0; q["x"] = x; probes["nodyn"].append(q)
        for oi in rng.choice(len(others), 4, replace=False):
            q = dict(others[oi]); q["g"] = o["g"]; probes["other"].append(q)
        for name, obs in probes.items():
            if name == "natural":
                Q = first[None]
                ref = P[None]
                # distance of the visited state from the first decision of the run
                recs.append(dict(probe=name, changed=float(first.argmax() != a), tv=0.5 * np.abs(P - first).sum(), p_top=P[a]))
                continue
            Q = query(net, obs)
            for row in Q:
                recs.append(dict(probe=name, changed=float(row.argmax() != a), tv=0.5 * np.abs(row - P).sum(), p_top=row[a]))
        pD, pR, qmin, qmax, tm = decode_action(np.unravel_index(a, (19, 5, 3)), eng, cfg)
        eng.segment(min(K, T - eng.iter), pD, pR, qmin, qmax, tm)
    return recs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--data", default="data/solomon")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(2)
    rng = np.random.default_rng(0)
    paths = sorted(glob.glob(os.path.join(a.data, "*.txt")))
    pick = []
    for cls in ("c1", "c2", "r1", "r2", "rc1", "rc2"):
        cand = [p for p in paths if os.path.basename(p).lower().startswith(cls) and
                (cls.startswith("rc") or not os.path.basename(p).lower().startswith("rc"))]
        pick += cand[:2]
    insts = [read_solomon(p) for p in pick]
    rows = []
    for spec in a.models:
        name, path = spec.split("=", 1)
        net, cfg = load_policy(path)
        # graphs of the other instances at their initial solution, used by the "other" probe
        others = {}
        for inst in insts:
            e = Engine(inst, 0, 5000, None)
            others[inst.name] = build_obs(e, cfg.get("dyn", True), cfg.get("ctx", False))
        for inst in insts:
            oth = [o for k, o in others.items() if k != inst.name]
            for r in run(net, cfg, inst, oth, rng):
                rows.append(dict(model=name, instance=inst.name, **r))
        print(name, "done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(a.out, index=False)
    print(df.groupby(["model", "probe"])[["changed", "tv", "p_top"]].mean().round(4).to_string())
