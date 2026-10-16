"""Run one method on one instance and return a checked result record.

Method specs:
    uni | rw                     non-learned baselines
    rw_fixedq                    RW with a fixed 10 % destroy size
    rwq:<ratio>                  RW with a fixed destroy fraction (headroom study)
    rwt:K-rho-qhi-m              tuned RW (update period, reaction factor, max destroy fraction, temp. deviation)
    static:a1-a2-a3              constant action of the SG-ALNS action space
    sched:a1-a2-a3/b1-b2-b3      action A in the first half of the main phase, B in the second half
    sg:<name> | sgs:<name>       trained policy, greedy | sampled actions
"""
import time
import numpy as np
from .alns import run_baseline, Engine
from .env import build_obs, decode_action
from .instance import check_solution


class ConstPolicy:
    def __init__(self, a):
        self.a = np.array(a)

    def action(self, eng, obs_fn, greedy):
        return self.a


class SchedPolicy:
    def __init__(self, a, b):
        self.a, self.b = np.array(a), np.array(b)

    def action(self, eng, obs_fn, greedy):
        return self.a if eng.iter < eng.T // 2 else self.b


class NetPolicy:
    def __init__(self, net):
        self.net = net

    def action(self, eng, obs_fn, greedy):
        return self.net.act([obs_fn()], greedy=greedy)[0][0]


def run_policy(inst, policy, cfg, T, seed, greedy=True, red_iters=None):
    """Segment-level control loop shared by SG-ALNS, static configurations and schedules (K from cfg)."""
    t0 = time.perf_counter()
    eng = Engine(inst, seed, T, red_iters)
    K = int(cfg.get("K", 50))
    obs_fn = lambda: build_obs(eng, cfg.get("dyn", True), cfg.get("ctx", False))
    eng.actions, eng.t_pol = [], 0.0
    while eng.iter < T:
        tp = time.perf_counter()
        a = policy.action(eng, obs_fn, greedy)
        eng.t_pol += time.perf_counter() - tp          # features + network inference
        if K > 1:
            eng.actions.append([int(eng.iter), int(a[0]), int(a[1]), int(a[2])])
        pD, pR, qmin, qmax, tm = decode_action(a, eng, cfg)
        eng.segment(min(K, T - eng.iter), pD, pR, qmin, qmax, tm)
        eng.trace[-1] = (eng.iter, time.perf_counter() - t0, eng.f_best)   # wall clock incl. inference
    eng.wall = time.perf_counter() - t0
    return eng


def record(method, inst, eng, seed, T):
    ok, nv, td, msg = check_solution(inst, eng.best_routes())
    assert ok, f"infeasible solution from {method} on {inst.name}: {msg}"
    tr = eng.trace
    return dict(method=method, instance=inst.name, seed=seed, T=T, NV=nv, TD=td, f=eng.f_best,
                time_s=getattr(eng, "wall", eng.time_total()), nv_init=eng.nv_init,
                trace=[(int(i), round(float(t), 4), float(f)) for i, t, f in tr[::max(1, len(tr) // 200)]] + [tuple(tr[-1])],
                actions=getattr(eng, "actions", []), t_red=eng.t_red, t_alns=eng.t_run, t_pol=getattr(eng, "t_pol", 0.0))


def run_method(spec, inst, T, seed, policies):
    """policies: name -> (torch policy, env cfg), needed for sg:/sgs: specs."""
    kind, _, arg = spec.partition(":")
    if spec in ("uni", "rw"):
        eng = run_baseline(inst, spec, T, seed)
    elif spec == "rw_fixedq":
        eng = run_baseline(inst, "rw", T, seed, fixed_q=0.10)
    elif kind == "rwq":
        eng = run_baseline(inst, "rw", T, seed, fixed_q=float(arg))
    elif kind == "rwt":
        K, rho, qhi, m = arg.split("-")
        eng = run_baseline(inst, "rw", T, seed, K=int(K), rho=float(rho), qhi=float(qhi), tau_mult=float(m))
    elif kind == "static":
        eng = run_policy(inst, ConstPolicy([int(x) for x in arg.split("-")]), {"K": 50}, T, seed)
    elif kind == "sched":
        A, B = ([int(x) for x in part.split("-")] for part in arg.split("/"))
        eng = run_policy(inst, SchedPolicy(A, B), {"K": 50}, T, seed)
    elif kind in ("sg", "sgs"):
        net, cfg = policies[arg]
        if kind == "sgs":                                  # sampled actions: reproducible per (instance, seed)
            import torch
            torch.manual_seed(seed)
        eng =run_policy(inst, NetPolicy(net), cfg, T, seed, greedy=(kind == "sg"))
    else:
        raise ValueError(f"unknown method {spec}")
    return record(spec, inst, eng, seed, T)
