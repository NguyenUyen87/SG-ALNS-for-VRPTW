"""Evaluate methods on benchmark files in parallel; the CSV output is appended to and resumable.

python scripts/evaluate.py --data data/solomon --methods rw uni sg:main=results/main/last.pt \
       --T 5000 --seeds 5 --workers 8 --out results/e1_solomon.csv
"""
import argparse, csv, glob, json, os, re, sys
import multiprocessing as mp
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FIELDS = ["method", "instance", "class", "seed", "T", "NV", "TD", "f", "time_s", "nv_init",
          "trace", "actions", "t_red", "t_alns", "t_pol"]
_POL, _INST = {}, {}


def cls_of(name):
    m = re.match(r"(RC|R|C)(\d)", name.upper())
    return m.group(1) + m.group(2) if m else "NA"


def _init(policy_specs):
    import numpy as np                       # warm-up so that timings exclude JIT compilation
    from sgalns.instance import generate
    from sgalns.alns import run_baseline
    run_baseline(generate(30, "R", False, np.random.default_rng(0)), "rw", 200, 0)
    if policy_specs:
        import torch
        torch.set_num_threads(1)
        from sgalns.policy import load_policy
        for spec in policy_specs:
            name, path = spec.split("=", 1)
            _POL[name.split(":", 1)[1]] = load_policy(path)


def _task(t):
    from sgalns.instance import read_solomon
    from sgalns.methods import run_method
    spec, path, T, seed = t
    if path not in _INST:
        _INST[path] = read_solomon(path)
    inst = _INST[path]
    rec = run_method(spec, inst, T, seed, _POL)
    rec["class"] = cls_of(inst.name)
    rec["trace"] = json.dumps(rec["trace"])
    rec["actions"] = json.dumps(rec["actions"])
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--T", nargs="+", type=int, default=[5000])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.data, "*.txt")) + glob.glob(os.path.join(a.data, "*.TXT")))
    pol_specs = [m for m in a.methods if m.startswith(("sg:", "sgs:"))]
    done = set()
    if os.path.exists(a.out):
        done = {(r["method"], r["instance"], int(r["seed"]), int(r["T"])) for r in csv.DictReader(open(a.out, encoding="utf-8"))}
    tasks = [(m.split("=")[0], p, T, seed) for T in a.T for p in paths for seed in range(a.seeds) for m in a.methods
             if (m.split("=")[0], os.path.splitext(os.path.basename(p))[0], seed, T) not in done]
    tasks.sort(key=lambda t: -t[2])          # longest budgets first for load balance
    print(f"{len(tasks)} runs on {a.workers} workers", flush=True)
    if not tasks:
        sys.exit(0)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="", encoding="utf-8") as fh, mp.get_context("spawn").Pool(a.workers, _init, (pol_specs,)) as pool:
        w = csv.DictWriter(fh, FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        for i, rec in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            w.writerow(rec)
            fh.flush()
            if i % 100 == 0:
                print(f"  {i + 1}/{len(tasks)}", flush=True)
