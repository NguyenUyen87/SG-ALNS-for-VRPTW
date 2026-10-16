"""Build results/REPORT.md (tables of every experiment, side by side with the values reported in the paper)
and the figures in results/figs/. Missing experiment files are skipped, so it can be run on partial results.

python scripts/report.py
"""
import json, os, sys
from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"
FIG = os.path.join(R, "figs")
CLASSES = ["C1", "C2", "R1", "R2", "RC1", "RC2"]
D_NAMES = ["Random", "Worst", "Shaw", "String", "Route", "Wait"]
R_NAMES = ["Greedy", "Regret-3", "Noisy"]
QS = ["5%", "10%", "20%", "30%", "U[5,30]%"]
TS = ["0.3", "1", "3"]


def read(name):
    return open(os.path.join(R, name)).read().strip() if os.path.exists(os.path.join(R, name)) else None


ST, RWT = read("tune_static_best.txt") or "static:10-3-0", read("tune_rw_best.txt") or "rwt:100-0.3-0.4-1.0"
_dh = (read("dyn_headroom_best.txt") or "\n").split("\n")
CONST, SCHED = _dh[0], _dh[-1]
LAB = {"uni": "ALNS-Uni", "rw": "ALNS-RW", RWT: "Tuned ALNS-RW", ST: "Tuned static", "sg:main": "SG-ALNS",
       "sgs:main": "SG-ALNS (sampled)", "sg:main_s1": "SG-ALNS seed 1", "sg:main_s2": "SG-ALNS seed 2",
       "sg:ctx": "SG-ALNS-C", "sg:main_ent": "SG-ALNS, const. entropy + neutral prior",
       "sg:no_expert": "w/o expert actions & prior", "sg:no_a1": "w/o a1 (RW operators)",
       "sg:no_a2": "w/o a2 (default q range)", "sg:no_a3": "w/o a3 (scheduled tau)",
       "sg:no_dyn": "w/o dynamic TW features", "sg:mlp": "MLP encoder", "sg:iter_gnn": "Iteration-level GNN (GRLOS-like)",
       "sg:iter_mlp": "Iteration-level MLP (PPO-ALNS-like)", SCHED: "Best two-phase schedule"}
LAB.setdefault(CONST, "Best constant (dyn. study)")

# values reported in the paper (SG_ALNS/results_*.tex, tables/*.tex) for side-by-side comparison
PAPER_E1 = {"uni": (421.0, 57352, 2.09, 5.44, 1.10, "35/16/5", "<0.001"), "rw": (421.4, 57115, 1.86, 5.58, 1.09, "25/16/15", "0.011"),
            RWT: (419.8, 57144, 1.61, 5.30, 1.44, "25/17/14", "0.065"), ST: (418.6, 57190, 1.69, 5.07, 0.97, "25/16/15", "0.253"),
            "sg:main": (418.0, 57188, 1.48, 5.05, 2.15, "--", "--"), "sg:ctx": (422.0, 57512, 2.44, 5.56, 1.71, "34/16/6", "<0.001"),
            "sgs:main": (418.4, 57190, 1.53, 4.95, 1.44, "21/16/19", "0.353"),
            "sg:iter_gnn": (425.3, 57424, 2.63, 5.83, 18.80, "34/15/7", "<0.001"),
            "sg:iter_mlp": (425.3, 57424, 2.63, 5.83, 10.66, "34/15/7", "<0.001")}
PAPER_ABL = {"sg:no_expert": (419.0, 1.40, "18/16/22", "0.657"), "sg:no_a1": (421.6, 1.85, "27/16/13", "0.002"),
             "sg:no_a2": (420.6, 1.95, "26/16/14", "0.002"), "sg:no_a3": (420.2, 1.64, "23/16/17", "0.062"),
             "sg:no_dyn": (420.2, 2.05, "28/16/12", "0.002"), "sg:mlp": (420.2, 2.05, "28/16/12", "0.002")}


# ============================================================================================ helpers
def primal_integral(trace, f_ref, T):
    tr = json.loads(trace)
    its = np.array([t[0] for t in tr], float)
    f = np.array([t[2] for t in tr], float)
    return float(np.sum(100 * (f - f_ref) / f_ref * np.diff(np.r_[its, T])) / T)


def load(name, bks=True):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, keep_default_na=False)
    df["instance"] = df["instance"].str.upper()
    if bks:
        b = pd.read_csv("data/bks.csv").set_index("instance")
        df["NV_ref"], df["TD_ref"] = df.instance.map(b.NV_bks), df.instance.map(b.TD_bks)
    else:                                         # best solution found in the experiment
        g = df.sort_values(["NV", "TD"]).groupby("instance").first()
        df["NV_ref"], df["TD_ref"] = df.instance.map(g.NV), df.instance.map(g.TD)
    df["gap"] = np.where(df.NV == df.NV_ref, 100 * (df.TD - df.TD_ref) / df.TD_ref, np.nan)
    df["PI"] = [primal_integral(t, 1e4 * nv + td, T) for t, nv, td, T in zip(df.trace, df.NV_ref, df.TD_ref, df["T"])]
    return df


def per_instance(df):
    return df.groupby(["method", "class", "instance"]).agg(NV=("NV", "mean"), TD=("TD", "mean"), f=("f", "mean"),
                                                           gap=("gap", "mean"), PI=("PI", "mean"),
                                                           time=("time_s", "mean")).reset_index()


def summary(df, methods):
    pi = per_instance(df[df.method.isin(methods)])
    g = pi.groupby("method").agg(CNV=("NV", "sum"), CTD=("TD", "sum"), gap=("gap", "mean"), PI=("PI", "mean"), time=("time", "mean"))
    return g.reindex([m for m in methods if m in g.index])


def wtl(df, a, b):
    """W/T/L of a vs b on instance means of f over common seeds; one-sided Wilcoxon p (H1: a better)."""
    seeds = set(df[df.method == a].seed) & set(df[df.method == b].seed)
    base = df[df.seed.isin(seeds) & df.method.isin([a, b])].groupby(["method", "instance"]).f.mean().unstack(0).dropna()
    d = (base[a] - base[b]).values
    w, l = int((d < -1e-6).sum()), int((d > 1e-6).sum())
    p = wilcoxon(base[a], base[b], alternative="less").pvalue if np.any(np.abs(d) > 1e-9) else 1.0
    return f"{w}/{len(d) - w - l}/{l}", p


def holm(p):
    p = np.asarray(p, float)
    adj, run = np.empty_like(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(run, 1.0)
    return adj


def fp(p):
    return "<0.001" if p < 1e-3 else f"{p:.3f}"


def md(rows, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    return "\n".join(out + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows])


def act_name(code):
    a1, a2, a3 = (int(x) for x in code.split("-"))
    op = "RW" if a1 == 18 else f"{D_NAMES[a1 // 3]}/{R_NAMES[a1 % 3]}"
    return f"{op}, q={QS[a2]}, m={TS[a3]}"


def action_stats(df, method):
    """Greedy-action behaviour: most frequent action, its share, mean number of switches per run."""
    cnt, sw = Counter(), []
    for a in df[df.method == method].actions:
        acts = ["-".join(map(str, x[1:])) for x in json.loads(a)]
        cnt.update(acts)
        sw.append(sum(1 for i in range(1, len(acts)) if acts[i] != acts[i - 1]))
    if not cnt:
        return None
    top, n = cnt.most_common(1)[0]
    return act_name(top), n / sum(cnt.values()), float(np.mean(sw)), len(cnt)


# ============================================================================================ sections
def sec_tuning(out):
    out.append("## 1. Offline tuning and headroom studies (24 generated instances, n=100, 3 seeds)\n")
    out.append("Score vs ALNS-RW = 5·ΔNV + 100·ΔTD/TD (positive = better than RW).\n")
    for f, title in (("tune_static.csv", "Tuned static (30 configurations)"), ("tune_rw.csv", "Tuned ALNS-RW (36 configurations)")):
        df = pd.read_csv(os.path.join(R, f)) if os.path.exists(os.path.join(R, f)) else None
        if df is None:
            continue
        g = df[df.spec != "rw"].groupby("spec").agg(score=("score", "mean"), NV=("NV", "sum")).sort_values("score", ascending=False)
        out.append(f"**{title}** — top 5:\n")
        out.append(md([(s, f"{r.score:+.3f}", int(r.NV)) for s, r in g.head(5).iterrows()], ["config", "score", "ΣNV"]) + "\n")
    out.append(f"Selected: tuned static = `{ST}` ({act_name(ST.split(':')[1])}); paper: `static:10-3-0` (String/Regret-3, q=30%, m=0.3). "
               f"Tuned RW = `{RWT}`; paper: `rwt:100-0.3-0.4-1.0`.\n")
    p = os.path.join(R, "dyn_headroom.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d = d[d.spec != "rw"].copy()
        d["A"] = d.spec.str.split(":").str[1].str.split("/").str[0]
        d["B"] = d.spec.str.split("/").str[1]
        s = d.groupby("spec").agg(score=("score", "mean"), A=("A", "first"), B=("B", "first"))
        con = s[s.A == s.B].sort_values("score", ascending=False)
        dyn = s[s.A != s.B].sort_values("score", ascending=False)
        bc, bd = con.index[0], dyn.index[0]
        pv = d.pivot_table(index=["inst", "seed"], columns="spec", values="score")
        pval = wilcoxon(pv[bd].groupby("inst").mean(), pv[bc].groupby("inst").mean()).pvalue
        m01, m2 = d[d.seed < 2].groupby("spec").score.mean(), d[d.seed == 2].groupby("spec").score.mean()
        sd, sc = max(dyn.index, key=lambda x: m01[x]), max(con.index, key=lambda x: m01[x])
        out.append("**Dynamic headroom (two-phase schedules over the 10 best statics)**\n")
        out.append(md([
            ("best constant score", f"{con.score.iloc[0]:.2f}", "0.70"),
            ("best two-phase score", f"{dyn.score.iloc[0]:.2f}", "0.81"),
            ("Wilcoxon p (best schedule vs best constant)", f"{pval:.2f}", "0.51"),
            ("schedules beating the best constant", f"{int((dyn.score > con.score.iloc[0]).sum())}/{len(dyn)}", "11/90"),
            ("hold-out seed: selected schedule / constant", f"{m2[sd]:.2f} / {m2[sc]:.2f}", "0.20 / 0.30"),
            ("best schedule", f"{act_name(dyn.loc[bd, 'A'])} → {act_name(dyn.loc[bd, 'B'])}",
             "String/Regret-3 20% m=0.3 → String/Regret-3 30% m=3")], ["quantity", "this run", "paper"]) + "\n")
    p = os.path.join(R, "headroom.csv")
    if os.path.exists(p):
        d = pd.read_csv(p)
        d["dTD"] = 100 * (d.TD - d.TD_rw) / d.TD_rw
        g = d[d.spec != "rw"].groupby("spec").agg(dNV=("NV", "sum"), dTD=("dTD", "mean"), score=("score", "mean"))
        g["dNV"] -= d[d.spec == "rw"].NV.sum()
        g = g.sort_values("score", ascending=False)
        out.append("**Static headroom (T=2500): fixed destroy sizes / lead operator pairs vs RW** — top 8 and bottom 3:\n")
        rows = [(s if not s.startswith("static") else act_name(s.split(":")[1]), int(r.dNV), f"{r.dTD:+.2f}", f"{r.score:+.3f}")
                for s, r in pd.concat([g.head(8), g.tail(3)]).iterrows()]
        out.append(md(rows, ["config", "ΔNV vs RW", "ΔTD % vs RW", "score"]) + "\n")


def sec_e1(out, e1, e4):
    out.append("## 2. Main comparison — Solomon (56 instances, n=100, T=5000)\n")
    out.append("5 seeds (iteration-level baselines: 3 seeds, compared on common seeds). Gap to SINTEF BKS over runs with BKS fleet size; "
               "W/T/L of SG-ALNS against each row on instance means of f; p = one-sided Wilcoxon, Holm-corrected over rows.\n")
    df = pd.concat([x for x in (e1, e4) if x is not None])
    methods = [m for m in ["sg:main", "rw", RWT, "uni", ST, "sg:ctx", "sgs:main", "sg:iter_gnn", "sg:iter_mlp"] if m in set(df.method)]
    g = summary(df, methods)
    tests = {m: wtl(df, "sg:main", m) for m in methods if m != "sg:main"}
    ph = dict(zip(tests, holm([t[1] for t in tests.values()]))) if tests else {}
    rows = [("BKS", 405, "57,187", "0.00", "--", "--", "--", "--", "")]
    for m in methods:
        r = g.loc[m]
        w, p = tests.get(m, ("--", None))
        pp = PAPER_E1.get(m)
        rows.append((LAB.get(m, m), f"{r.CNV:.1f}", f"{r.CTD:,.0f}", f"{r.gap:.2f}", f"{r.PI:.2f}", f"{r.time:.2f}", w,
                     fp(ph[m]) if m in ph else "--",
                     f"{pp[0]:.1f} / {pp[2]:.2f} / {pp[5]} ({pp[6]})" if pp else ""))
    seeds = [m for m in ("sg:main", "sg:main_s1", "sg:main_s2") if m in set(df.method)]
    if len(seeds) == 3:
        gs = summary(df, seeds)
        rows.insert(2, ("SG-ALNS, 3 training seeds (mean±sd)", f"{gs.CNV.mean():.1f}±{gs.CNV.std():.1f}", f"{gs.CTD.mean():,.0f}",
                        f"{gs.gap.mean():.2f}±{gs.gap.std():.2f}", f"{gs.PI.mean():.2f}", f"{gs.time.mean():.2f}", "--", "--",
                        "419.0±1.1 / 1.88±0.35"))
    out.append(md(rows, ["Method", "CNV", "CTD", "Gap %", "PI %", "Time s", "SG W/T/L", "p_Holm", "paper: CNV / gap / W/T/L (p)"]) + "\n")

    # per-class table
    pi = per_instance(df[df.method.isin(["rw", RWT, ST, "sg:main", "sg:ctx"])])
    b = pd.read_csv("data/bks.csv")
    b["class"] = b.instance.str.extract(r"^(RC|R|C)(\d)").agg("".join, axis=1)
    rows = []
    for c in CLASSES + ["All"]:
        sel = pi if c == "All" else pi[pi["class"] == c]
        bs = b if c == "All" else b[b["class"] == c]
        row = [c, f"{bs.NV_bks.sum()} / {bs.TD_bks.sum():,.0f}"]
        for m in ["rw", RWT, ST, "sg:main", "sg:ctx"]:
            s = sel[sel.method == m]
            row.append(f"{s.NV.sum():.1f} / {s.TD.sum():,.0f}" if len(s) else "")
        rows.append(row)
    out.append("**Per class (CNV / CTD)**\n")
    out.append(md(rows, ["Class", "BKS", "ALNS-RW", "Tuned RW", "Tuned static", "SG-ALNS", "SG-ALNS-C"]) + "\n")

    # SG-ALNS-C vs others (paper Table main2)
    if "sg:ctx" in set(df.method):
        rows = []
        for m in ["uni", "rw", RWT, ST, "sg:main"]:
            w, p = wtl(df, "sg:ctx", m)
            rows.append((LAB[m], w, fp(p)))
        out.append("**SG-ALNS-C against each method** (unadjusted p; paper: 18/16/22, 8/16/32, 10/16/30, 6/16/34, 6/16/34 — always worse)\n")
        out.append(md(rows, ["vs", "W/T/L", "p"]) + "\n")

    # Friedman / Nemenyi over the six methods run with 5 seeds
    fm = [m for m in ["uni", "rw", RWT, ST, "sg:main", "sg:ctx"] if m in set(e1.method)]
    if len(fm) == 6:
        f = per_instance(e1[e1.method.isin(fm)]).pivot(index="instance", columns="method", values="f")[fm].round(6)
        stat, p = friedmanchisquare(*[f[m] for m in fm])
        ranks = f.rank(axis=1, method="average").mean()
        k, N = len(fm), len(f)
        cd = studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2) * np.sqrt(k * (k + 1) / (6 * N))
        paper_r = {"uni": 4.29, "rw": 3.32, RWT: 3.29, ST: 2.96, "sg:main": 2.71, "sg:ctx": 4.43}
        out.append(f"**Friedman test** over 6 methods: χ²={stat:.1f}, p={p:.2e} (paper 1.1e-10); Nemenyi CD = {cd:.2f} (paper 1.01)\n")
        out.append(md([(LAB[m], f"{ranks[m]:.2f}", paper_r[m]) for m in ranks.sort_values().index], ["Method", "mean rank", "paper"]) + "\n")
        fig, ax = plt.subplots(figsize=(6, 1.8))
        rs = ranks.sort_values()
        ax.set_xlim(1, k); ax.set_ylim(0, 1); ax.get_yaxis().set_visible(False)
        for i, (m, v) in enumerate(rs.items()):
            ax.plot([v, v], [0.5, 0.15 + 0.1 * i], color="k", lw=0.8)
            ax.text(v, 0.1 + 0.1 * i, LAB[m], ha="center", va="top", fontsize=7)
            ax.plot(v, 0.5, "o", color="C0")
        ax.plot([rs.iloc[0], rs.iloc[0] + cd], [0.8, 0.8], color="C3", lw=2)
        ax.text(rs.iloc[0] + cd / 2, 0.85, f"CD = {cd:.2f}", ha="center", fontsize=7)
        ax.set_title("Mean Friedman ranks (lower is better)", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(FIG, "cd.png"), dpi=160); plt.close(fig)


def sec_ablation(out, e1):
    out.append("## 3. Ablations (Solomon, T=5000, 5 seeds; each a separately trained policy)\n")
    abl = [m for m in PAPER_ABL if m in set(e1.method)]
    if not abl:
        return
    g = summary(e1, ["sg:main"] + abl)
    tests = {m: wtl(e1, "sg:main", m) for m in abl}
    ph = dict(zip(tests, holm([t[1] for t in tests.values()])))
    rows = [("SG-ALNS", f"{g.loc['sg:main'].CNV:.1f}", f"{g.loc['sg:main'].gap:.2f}", f"{g.loc['sg:main'].PI:.2f}",
             f"{g.loc['sg:main'].time:.2f}", "--", "--", "418.0 / 1.48")]
    for m in abl:
        r, pp = g.loc[m], PAPER_ABL[m]
        rows.append((LAB[m], f"{r.CNV:.1f}", f"{r.gap:.2f}", f"{r.PI:.2f}", f"{r.time:.2f}", tests[m][0], fp(ph[m]),
                     f"{pp[0]:.1f} / {pp[1]:.2f} / {pp[2]} ({pp[3]})"))
    out.append(md(rows, ["Variant", "CNV", "Gap %", "PI %", "Time s", "SG W/T/L", "p_Holm", "paper: CNV / gap / W/T/L (p)"]) + "\n")


def sec_collapse(out, dfs):
    out.append("## 4. Policy behaviour (greedy actions on Solomon, T=5000)\n")
    out.append("Paper: 11 of 12 policies collapse to a single constant action; the exception switches once. "
               "Collapse-sensitivity run (constant entropy 0.01, neutral prior): main action 78 % of decisions, 1.7 switches/run, CNV 422.6.\n")
    rows = []
    df = pd.concat([d for d in dfs if d is not None])
    for m in ["sg:main", "sg:main_s1", "sg:main_s2", "sg:ctx", "sg:main_ent", "sg:no_expert", "sg:no_a1", "sg:no_a2",
              "sg:no_a3", "sg:no_dyn", "sg:mlp"]:
        s = action_stats(df, m)
        if s:
            rows.append((LAB[m], s[0], f"{100 * s[1]:.1f}%", f"{s[2]:.2f}", s[3]))
    out.append(md(rows, ["Policy", "most frequent action", "share", "switches/run", "#distinct"]) + "\n")
    if "sg:main_ent" in set(df.method):
        e = df[df.method.isin(["sg:main_ent", "rw"])]
        g = summary(e, ["sg:main_ent", "rw"])
        w, p = wtl(e, "sg:main_ent", "rw")
        out.append(f"Collapse-sensitivity policy: CNV {g.loc['sg:main_ent'].CNV:.1f}, gap {g.loc['sg:main_ent'].gap:.2f}%, "
                   f"vs ALNS-RW {w} (p={fp(p)}); paper: 422.6, 2.32 %, 10/16/30 (0.999).\n")
    if SCHED in set(df.method) and CONST in set(df.method):
        g = summary(df, [SCHED, CONST])
        w, p = wtl(df, SCHED, CONST)
        out.append(f"Best two-phase schedule on Solomon: CNV {g.loc[SCHED].CNV:.1f}, gap {g.loc[SCHED].gap:.2f}% vs best constant "
                   f"CNV {g.loc[CONST].CNV:.1f}, gap {g.loc[CONST].gap:.2f}%; W/T/L {w} (p={fp(p)}). "
                   "Paper: 419.0 / 1.47 % vs 418.6 / 1.69 %, 26/15/15 (0.013).\n")


def sec_budget(out, e1, e3, e5):
    out.append("## 5. Iteration budget and time-matched comparison (Solomon)\n")
    if e3 is not None:
        e = pd.concat([e3, e1[e1.method.isin(set(e3.method)) & (e1.seed < 3)]])
        rows = []
        for T in sorted(set(e["T"])):
            d = e[e["T"] == T]
            g = summary(d, ["rw", RWT, ST, "sg:main", "sg:ctx"])
            row = [T] + [f"{g.loc[m].CNV:.1f} / {g.loc[m].gap:.2f}" if m in g.index else "" for m in ["rw", RWT, ST, "sg:main", "sg:ctx"]]
            for b in ("rw", RWT, ST):
                w, p = wtl(d, "sg:main", b)
                row.append(f"{w} ({fp(p)})")
            rows.append(row)
        out.append("3 seeds; cells = CNV / gap %. Paper (SG vs RW, vs tuned RW, vs tuned static): T=1000: 32/14/10, 25/14/17, 27/14/15; "
                   "T=5000: 23/16/17, 24/17/15, 21/16/19; T=20000: 27/16/13, 21/17/18, 16/16/24.\n")
        out.append(md(rows, ["T", "ALNS-RW", "Tuned RW", "Tuned static", "SG-ALNS", "SG-ALNS-C", "SG vs RW", "SG vs tuned RW", "SG vs static"]) + "\n")
        fig, ax = plt.subplots(figsize=(4.5, 3))
        for m in ["rw", RWT, ST, "sg:main", "sg:ctx"]:
            g = e[e.method == m].groupby("T").agg(gap=("gap", "mean"), time=("time_s", "mean"))
            ax.plot(g.time, g.gap, "o-", label=LAB[m])
        ax.set_xscale("log"); ax.set_xlabel("mean time per run (s)"); ax.set_ylabel("gap to BKS (%)")
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(os.path.join(FIG, "budget.png"), dpi=160); plt.close(fig)
    if e5 is not None:
        sg = e1[e1.method == "sg:main"]
        e = pd.concat([e5, sg])
        rows = []
        g = summary(e5, ["rw", RWT, ST])
        paper = {"rw": "419.8 / 1.40 / 2.04 s; 19/16/21 (0.425)", RWT: "418.8 / 1.25 / 2.71 s; 14/17/25 (0.967)",
                 ST: "417.8 / 1.28 / 2.57 s; 13/16/27 (0.984)"}
        for m in ["rw", RWT, ST]:
            w, p = wtl(e, "sg:main", m)
            rows.append((LAB[m] + " (T=10000)", f"{g.loc[m].CNV:.1f}", f"{g.loc[m].gap:.2f}", f"{g.loc[m].time:.2f}", f"{w} ({fp(p)})", paper[m]))
        out.append("**Time-matched**: baselines with T=10000 vs SG-ALNS with T=5000 (5 seeds).\n")
        out.append(md(rows, ["Baseline", "CNV", "Gap %", "Time s", "SG W/T/L (p)", "paper: CNV / gap / time; W/T/L (p)"]) + "\n")


def sec_scale(out, e2, e6):
    out.append("## 6. Larger instances — Gehring–Homberger n=200 and n=400 (T=5000, 3 seeds)\n")
    out.append("Gap and PI relative to the best solution found in each experiment; W/T/L vs ALNS-RW (one-sided Wilcoxon).\n")
    paper = {200: {"rw": "697.7 / 1.93", RWT: "697.7 / 1.66; 34/6/20", ST: "697.7 / 1.93; 32/4/24", "sg:main": "697.3 / 2.41; 18/6/36",
                   "sg:ctx": "698.0 / 1.66; 34/6/20", "uni": "699.3"},
             400: {"rw": "1397.3 / 2.79", RWT: "1397.0 / 2.75; 35/0/25", ST: "1395.7 / 3.79; 27/1/32", "sg:main": "1396.0 / 5.01; 17/1/42",
                   "sg:ctx": "1396.7 / 1.53; 43/0/17"}}
    for n, e in ((200, e2), (400, e6)):
        if e is None:
            continue
        methods = [m for m in ["uni", "rw", RWT, ST, "sg:main", "sg:ctx"] if m in set(e.method)]
        g = summary(e, methods)
        rows = []
        for m in methods:
            w, p = wtl(e, m, "rw") if m != "rw" else ("--", None)
            rows.append((LAB[m], f"{g.loc[m].CNV:.1f}", f"{g.loc[m].CTD:,.0f}", f"{g.loc[m].gap:.2f}", f"{g.loc[m].PI:.2f}",
                         f"{g.loc[m].time:.2f}", w if p is None else f"{w} ({fp(p)})", paper[n].get(m, "")))
        out.append(f"**n = {n}**\n")
        out.append(md(rows, ["Method", "CNV", "CTD", "Gap %", "PI %", "Time s", "vs RW", "paper: CNV / gap; vs RW"]) + "\n")
        if "t_pol" not in e.columns:
            continue
        t = e[e.method.isin(["rw", "sg:main"])].groupby("method")[["t_red", "t_alns", "t_pol", "time_s"]].apply(
            lambda x: x.apply(pd.to_numeric, errors="coerce").mean())
        out.append("Run-time breakdown (s): " + "; ".join(
            f"{LAB[m]}: reduction {r.t_red:.2f}, ALNS {r.t_alns:.2f}, policy {r.t_pol:.2f}, total {r.time_s:.2f}" for m, r in t.iterrows()) + "\n")


def sec_training(out):
    out.append("## 7. Training\n")
    rows = []
    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    for name in ["main", "main_s1", "main_s2", "ctx", "main_ent", "no_expert", "no_a1", "no_a2", "no_a3", "no_dyn", "mlp", "iter_mlp", "iter_gnn"]:
        p = os.path.join(R, name, "train_log.csv")
        if not os.path.exists(p):
            continue
        t = pd.read_csv(p)
        v = t.val_impr_vs_rw_pct.dropna()
        rows.append((name, int(t.episodes.iloc[-1]), f"{t.time_s.iloc[-1] / 60:.1f}", f"{t.entropy.iloc[-1]:.2f}",
                     f"{v.iloc[-1]:+.3f}" if len(v) else "", f"{v.max():+.3f}" if len(v) else ""))
        if name in ("main", "main_s1", "main_s2", "ctx", "main_ent"):
            ax[0].plot(t.episodes, t.mean_return.rolling(10, min_periods=1).mean(), label=name)
            ax[1].plot(t.episodes, t.entropy, label=name)
    out.append(md(rows, ["model", "episodes", "minutes", "final entropy", "final val score", "best val score"]) + "\n")
    out.append("Paper: 21–23 min per SG-ALNS seed, 33 min for SG-ALNS-C, iteration-level GNN stopped after 1,368 episodes (57 min).\n")
    ax[0].set_title("episode return (rolling 10)", fontsize=8); ax[1].set_title("policy entropy", fontsize=8)
    for a in ax:
        a.set_xlabel("episodes"); a.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "training.png"), dpi=160); plt.close(fig)


def fig_convergence(e1):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for m in ["uni", "rw", RWT, ST, "sg:main"]:
        d = e1[(e1.method == m) & e1["class"].isin(["R1", "RC1"])]
        if d.empty:
            continue
        grid = np.linspace(0, 5000, 101)
        curves = []
        for tr, fr in zip(d.trace, 1e4 * d.NV_ref + d.TD_ref):
            t = np.array(json.loads(tr))
            idx = np.searchsorted(t[:, 0], grid, side="right") - 1
            curves.append(100 * (t[idx, 2] - fr) / fr)
        ax.plot(grid, np.mean(curves, 0), label=LAB[m])
    ax.set_yscale("log"); ax.set_xlabel("iteration of main phase"); ax.set_ylabel("gap of f to BKS (%)")
    ax.set_title("Convergence on R1 + RC1", fontsize=8); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "convergence.png"), dpi=160); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    e1, e4, e3, e5 = load("e1_solomon.csv"), load("e4_iter.csv"), load("e3_budget.csv"), load("e5_timematched.csv")
    e2, e6 = load("e2_hg200.csv", bks=False), load("e6_hg400.csv", bks=False)
    out = ["# SG-ALNS — re-implementation and re-run of the experimental scenario\n",
           "Generated by `scripts/report.py`. 'paper' columns quote the values of the submitted paper "
           "(SG_ALNS/results_numbers.tex, results_extra.tex, results_review.tex).\n"]
    sec_tuning(out)
    if e1 is not None:
        sec_e1(out, e1, e4)
        sec_ablation(out, e1)
        sec_collapse(out, [e1, e4])
        sec_budget(out, e1, e3, e5)
        fig_convergence(e1)
    sec_scale(out, e2, e6)
    sec_training(out)
    if os.path.exists("logs/verify_bks.log"):
        out.append("## 8. BKS cross-check\n```\n" + open("logs/verify_bks.log").read().strip() + "\n```\n"
                   "Paper: 19 CVRPLIB solutions with BKS fleet reproduce the BKS distance; none of 4,760 runs dominates a BKS; 22 BKS reproduced.\n")
    out.append("Figures: `results/figs/` (cd.png, budget.png, convergence.png, training.png).\n")
    open(os.path.join(R, "REPORT.md"), "w", encoding="utf-8").write("\n".join(out))
    print("\n".join(out))
