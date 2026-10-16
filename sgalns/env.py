"""Segment-level MDP: action space (a1, a2, a3), graph + global state, reward, and CPU-parallel environments.
This module never imports torch, so spawned workers stay light."""
import multiprocessing as mp
import numpy as np
from . import core as C
from .alns import Engine, N_OPS
from .instance import random_instance

# ---------------------------------------------------------------------------------------------- actions
Q_RATIOS = (0.05, 0.10, 0.20, 0.30, None)   # a2: fixed destroy fraction, or None = default U[5 %, 30 %]
TEMP_MULT = (0.3, 1.0, 3.0)                 # a3: multiplier on the scheduled temperature
N_A1, N_A2, N_A3 = C.N_D * C.N_R + 1, len(Q_RATIOS), len(TEMP_MULT)
RW_A1 = N_A1 - 1                            # a1 = 18: keep the roulette-wheel distribution (expert action)
FIXED_A2, FIXED_A3 = 4, 1                   # default destroy range / scheduled temperature (expert values)

NODE_DIM, EDGE_DIM, G_DIM = 12, 2, 6 + N_OPS
CTX_DIM = 2                                 # optional context: n / 200 and T / 5000


def action_to_probs(a1, p_main=0.7, eng=None):
    """a1 < 18: lead (destroy, repair) pair, each sampled w.p. p_main, the others share the rest uniformly.
    a1 == 18: the engine's roulette-wheel distribution."""
    if int(a1) == RW_A1:
        return eng.rw_probs()
    dm, rm = divmod(int(a1), C.N_R)
    pD = np.full(C.N_D, (1 - p_main) / (C.N_D - 1))
    pD[dm] = p_main
    pR = np.full(C.N_R, (1 - p_main) / (C.N_R - 1))
    pR[rm] = p_main
    return pD, pR


def decode_action(a, eng, cfg):
    """(a1, a2, a3) -> (pD, pR, qmin, qmax, tau multiplier); ablated components use the expert value."""
    a1 = int(a[0]) if cfg.get("use_a1", True) else RW_A1
    a2 = int(a[1]) if cfg.get("use_a2", True) else FIXED_A2
    a3 = int(a[2]) if cfg.get("use_a3", True) else FIXED_A3
    pD, pR = action_to_probs(a1, cfg.get("p_main", 0.7), eng)
    qmin, qmax = eng.q_range(Q_RATIOS[a2])
    return pD, pR, qmin, qmax, TEMP_MULT[a3]


# ---------------------------------------------------------------------------------------------- state
def _static_features(inst):
    if not hasattr(inst, "_static"):
        H = inst.l[0]
        inst._static = np.stack([inst.xy[:, 0] / 100.0, inst.xy[:, 1] / 100.0, inst.dem / inst.cap,
                                 inst.e / H, inst.l / H, inst.s / H], 1).astype(np.float32)
        ke = inst.knn_edges
        inst._knn_attr = np.stack([inst.d[ke[0], ke[1]] / inst.dmax, np.zeros(ke.shape[1])], 1).astype(np.float32)
    return inst._static


def build_obs(eng, dyn=True, ctx=False):
    """Graph = sparse k-NN edges (flag 0) + current solution edges (flag 1); nodes = 6 static + 6 dynamic
    features; global = progress, temperature, stagnation, gaps, fleet ratio and per-operator success rates."""
    inst = eng.inst
    n = inst.n
    stat = _static_features(inst)
    dynf = np.empty((n + 1, 6))
    C.node_dyn(eng.st, inst.ins, dynf)
    if not dyn:
        dynf[:, :3] = 0.0                       # ablation: no wait / slack / FTS
    x = np.concatenate([stat, dynf.astype(np.float32)], 1)
    src = np.empty(2 * (n + eng.st[1].shape[0] + 2), np.int64)
    dst = np.empty_like(src)
    k = C.sol_edges(eng.st, src, dst)
    src, dst = src[:k], dst[:k]
    ei = np.concatenate([inst.knn_edges, np.stack([src, dst])], 1)
    ea = np.concatenate([inst._knn_attr,
                         np.stack([inst.d[src, dst] / inst.dmax, np.ones(k)], 1).astype(np.float32)], 0)
    f_best, f_cur = eng.f_best, eng.f_cur
    rate = np.where(eng.uses > 0, eng.succ / np.maximum(eng.uses, 1), 0.0)
    g = [eng.iter / eng.T, eng.tau / eng.tau0, min(eng.sc[2] / (0.2 * eng.T), 1.0),
         (f_cur - f_best) / f_best, (eng.f_start - f_best) / eng.f_start, (eng.st[1] > 0).sum() / eng.nv_init]
    if ctx:
        rate = np.concatenate([rate, [n / 200.0, eng.T / 5000.0]])
    return {"x": x, "ei": ei, "ea": ea, "g": np.concatenate([g, rate]).astype(np.float32)}


# ---------------------------------------------------------------------------------------------- environment
class SegmentEnv:
    """One episode = one ALNS run of T main iterations in T/K segments; one decision (a1, a2, a3) per segment.
    Reward (reward='td'): 100 * dTD_best / TD_init + lam_nv * dNV_best  (reward='f': with f instead of TD)."""

    def __init__(self, cfg, seed=0):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.eng = None

    def reset(self, inst=None, seed=None):
        cfg = self.cfg
        if inst is None:
            inst = random_instance(self.rng, tuple(cfg.get("sizes", (50, 100))))
        seed = int(self.rng.integers(1 << 30)) if seed is None else seed
        self.T, self.K = int(cfg.get("T", 2500)), int(cfg.get("K", 50))
        self.eng = Engine(inst, seed, self.T, cfg.get("red_iters"))
        self.f_init, self.td_init = self.eng.f_best, self.eng.td_best
        return build_obs(self.eng, cfg.get("dyn", True), cfg.get("ctx", False)), {}

    def step(self, action):
        eng, cfg = self.eng, self.cfg
        pD, pR, qmin, qmax, tm = decode_action(action, eng, cfg)
        f_prev, nv_prev, td_prev = eng.f_best, eng.nv_best, eng.td_best
        eng.segment(min(self.K, self.T - eng.iter), pD, pR, qmin, qmax, tm)
        lam = cfg.get("lam_nv", 1.0)
        if cfg.get("reward", "f") == "f":
            r = 100.0 * (f_prev - eng.f_best) / self.f_init + lam * (nv_prev - eng.nv_best)
        else:
            r = 100.0 * (td_prev - eng.td_best) / self.td_init + lam * (nv_prev - eng.nv_best)
        done = eng.iter >= self.T
        info = {"f_best": eng.f_best, "nv": eng.nv_best, "td": eng.td_best} if done else {}
        return build_obs(eng, cfg.get("dyn", True), cfg.get("ctx", False)), float(r), done, False, info


def _worker(conn, cfg, seed):
    env = SegmentEnv(cfg, seed)
    try:
        while True:
            cmd, data = conn.recv()
            if cmd == "reset":
                conn.send(env.reset()[0])
            elif cmd == "cfg":
                env.cfg.update(data)
            elif cmd == "step":
                obs, r, done, _, info = env.step(data)
                conn.send((obs, r, done, info))
            else:
                break
    finally:
        conn.close()


class VecEnv:
    """N SegmentEnv instances in spawned processes, stepped in lock-step."""

    def __init__(self, cfg, n, seed):
        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, cfg, seed * 1000 + i), daemon=True)
            p.start()
            b.close()
            self.pipes.append(a)
            self.procs.append(p)

    def set_cfg(self, **kw):
        for p in self.pipes:
            p.send(("cfg", kw))

    def reset(self):
        for p in self.pipes:
            p.send(("reset", None))
        return [p.recv() for p in self.pipes]

    def step(self, actions):
        for p, a in zip(self.pipes, actions):
            p.send(("step", a))
        out = [p.recv() for p in self.pipes]
        return [o[0] for o in out], np.array([o[1] for o in out]), np.array([o[2] for o in out]), [o[3] for o in out]

    def close(self):
        for p in self.pipes:
            try:
                p.send(("close", None))
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=2)
