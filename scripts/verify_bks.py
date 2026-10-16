"""Cross-check data/bks.csv (hierarchical NV-then-TD BKS) against (i) the CVRPLIB Solomon solutions
re-evaluated with unrounded distances and (ii) every Solomon run in results/e*.csv (none may dominate a BKS)."""
import glob, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pandas as pd
from sgalns.instance import read_solomon, check_solution

b = pd.read_csv("data/bks.csv").set_index("instance")
rows = []
for name in b.index:
    inst = read_solomon(f"data/solomon/{name.lower()}.txt")
    txt = open(f"data/cvrplib_solomon_sol/{name}.sol").read()
    routes = [[int(x) for x in ln.split(":")[1].split()] for ln in txt.splitlines() if ln.startswith("Route")]
    ok, nv, td, _ = check_solution(inst, routes)
    rows.append(dict(instance=name, feasible_unrounded=ok, cv_nv=nv if ok else None, cv_td=round(td, 2) if ok else None))
d = pd.DataFrame(rows).set_index("instance").join(b)
feas = d[d.feasible_unrounded]
same = feas[feas.cv_nv == feas.NV_bks]
print(f"CVRPLIB solutions feasible with unrounded distances: {len(feas)}/{len(d)}")
print(f"  same NV as BKS: {len(same)}; TD identical (<0.01%): {((same.cv_td - same.TD_bks).abs() / same.TD_bks < 1e-4).sum()}")
print(f"  BKS NV <= CVRPLIB NV everywhere: {(feas.NV_bks <= feas.cv_nv).all()}")

r = pd.concat([pd.read_csv(f, usecols=["instance", "NV", "TD"]) for f in glob.glob("results/e*.csv")])
r["I"] = r.instance.str.upper()
r = r[r.I.isin(b.index)].join(b, on="I")
dom = r[(r.NV < r.NV_bks) | ((r.NV == r.NV_bks) & (r.TD < r.TD_bks * (1 - 1e-4)))]
best = r.sort_values(["NV", "TD"]).groupby("I").first()
hit = ((best.NV == best.NV_bks) & ((best.TD - best.TD_bks).abs() / best.TD_bks < 1e-4)).sum()
print(f"own Solomon runs: {len(r)}; runs dominating a BKS: {len(dom)}; BKS reproduced exactly: {hit}")
d.to_csv("results/bks_check.csv")
