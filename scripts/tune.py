"""Offline studies on 24 generated n=100 tuning instances (3 seeds, T=5000; never on benchmarks).
Score of a configuration vs ALNS-RW on the same instance and seed: 5 * dNV + 100 * dTD / TD_rw.

python scripts/tune.py static      # 30 static configurations of the SG-ALNS action space -> tune_static.csv
python scripts/tune.py rw          # 36 roulette-wheel configurations                      -> tune_rw.csv, tune_rw_best.txt
python scripts/tune.py dynamic     # 100 two-phase schedules over the 10 best statics      -> dyn_headroom.csv, dyn_headroom_best.txt
python scripts/tune.py full        # all 285 constant actions (leakage check for the 30-configuration grid) -> tune_full.csv
python scripts/tune.py headroom    # fixed destroy sizes / lead operator pairs vs RW (T=2500)  -> headroom.csv
"""
import itertools, os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
OUT = "results"


def job(args):
    spec, inst_seed, seed, T = args
    from sgalns.instance import random_instance
    from sgalns.methods import run_method
    t0 = time.process_time()
    r = run_method(spec, random_instance(np.random.default_rng(inst_seed), (100,)), T, seed, {})
    return spec, inst_seed, seed, r["NV"], r["TD"], time.process_time() - t0


def run_grid(specs, inst_base=90000, T=5000, n_inst=24, n_seeds=3):
    tasks = [(s, inst_base + i, sd, T) for s in ["rw"] + specs for i in range(n_inst) for sd in range(n_seeds)]
    t0 = time.time()
    with mp.get_context("spawn").Pool(8) as p:
        res = p.map(job, tasks, chunksize=2)
    df = pd.DataFrame(res, columns=["spec", "inst", "seed", "NV", "TD", "cpu_s"])
    print(f"{len(tasks)} runs: {df.cpu_s.sum() / 60:.1f} CPU-minutes, {(time.time() - t0) / 60:.1f} wall-minutes on 8 workers", flush=True)
    ref = df[df.spec == "rw"].set_index(["inst", "seed"])
    df = df.join(ref[["NV", "TD"]], on=["inst", "seed"], rsuffix="_rw")
    df["score"] = 5 * (df.NV_rw - df.NV) + 100 * (df.TD_rw - df.TD) / df.TD_rw
    return df


def summary(df):
    return df.groupby("spec").agg(score=("score", "mean"), NV=("NV", "sum")).sort_values("score", ascending=False)


if __name__ == "__main__":
    mode = sys.argv[1]
    os.makedirs(OUT, exist_ok=True)
    if mode == "static":
        df = run_grid([f"static:{a1}-{a2}-{a3}" for a1 in (18, 10) for a2 in range(5) for a3 in range(3)])
        df.to_csv(f"{OUT}/tune_static.csv", index=False)
        g = summary(df[df.spec != "rw"])
        open(f"{OUT}/tune_static_best.txt", "w").write(g.index[0])
    elif mode == "full":
        df = run_grid([f"static:{a1}-{a2}-{a3}" for a1 in range(19) for a2 in range(5) for a3 in range(3)])
        df.to_csv(f"{OUT}/tune_full.csv", index=False)
        g = summary(df[df.spec != "rw"])
        open(f"{OUT}/tune_full_best.txt", "w").write(g.index[0])
    elif mode == "rw":
        df = run_grid([f"rwt:{K}-{rho}-{qhi}-{m}" for K, rho, qhi, m in
                       itertools.product((50, 100), (0.1, 0.3), (0.2, 0.3, 0.4), (0.3, 1.0, 3.0))])
        df.to_csv(f"{OUT}/tune_rw.csv", index=False)
        g = summary(df[df.spec != "rw"])
        open(f"{OUT}/tune_rw_best.txt", "w").write(g.index[0])
    elif mode == "dynamic":
        ts = pd.read_csv(f"{OUT}/tune_static.csv")
        top = [s.split(":")[1] for s in ts[ts.spec != "rw"].groupby("spec").score.mean().sort_values(ascending=False).index[:10]]
        df = run_grid([f"sched:{a}/{b}" for a in top for b in top])
        df.to_csv(f"{OUT}/dyn_headroom.csv", index=False)
        s = df[df.spec != "rw"].groupby("spec").score.mean()
        A = s.index.str.split(":").str[1].str.split("/")
        const = s[[x[0] == x[1] for x in A]].sort_values(ascending=False)
        dyn = s[[x[0] != x[1] for x in A]].sort_values(ascending=False)
        print("best constant:", const.index[0], round(const.iloc[0], 3), "| best two-phase:", dyn.index[0], round(dyn.iloc[0], 3))
        open(f"{OUT}/dyn_headroom_best.txt", "w").write(f"{const.index[0]}\n{dyn.index[0]}\n")
        g = dyn.head(10).to_frame("score")
    elif mode == "headroom":                     # same runs and file format as the paper's headroom table
        specs = ["uni"] + [f"rwq:{q}" for q in (0.05, 0.1, 0.2, 0.3)] + [f"static:{a}-4-1" for a in range(18)]
        df = run_grid(specs, inst_base=5000, T=2500)
        df["cfg"] = [s if s in ("rw", "uni") else (str(("q", float(s[4:]))) if s.startswith("rwq")
                     else str(("op", int(s.split(":")[1].split("-")[0])))) for s in df.spec]
        df["dTD"] = 100 * (df.TD - df.TD_rw) / df.TD_rw
        df[["cfg", "inst", "seed", "NV", "TD", "NV_rw", "TD_rw", "dTD"]].to_csv(f"{OUT}/headroom.csv", index=False)
        g = df.groupby("cfg").agg(NV=("NV", "sum"), dTD=("dTD", "mean")).sort_values("dTD")
    else:
        sys.exit(f"unknown mode {mode}")
    print(g.round(3).to_string())
