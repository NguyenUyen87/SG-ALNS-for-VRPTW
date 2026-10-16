"""ALNS engine driven segment by segment, Solomon I1 construction, and the non-learned baselines.

Every method (baselines, SG-ALNS, iteration-level RL) shares: the I1 start, a 10n-iteration vehicle-reduction
phase (not counted in T, counted in wall-clock time), the operator portfolio, and the SA schedule
tau_0 -> 0.01 tau_0 over the T main iterations, with tau_0 such that a 5 % worse solution is accepted w.p. 0.5.
"""
import time
import numpy as np
from . import core as C

N_OPS = C.N_D + C.N_R
Q_LO, Q_HI = 0.05, 0.30          # default destroy range, as a fraction of n
RHO = 0.1                        # roulette-wheel reaction factor


def alloc_state(R, L, n):
    return (np.zeros((R, L), np.int32), np.zeros(R, np.int32), np.zeros((R, L + 1)),
            np.zeros((R, L + 1)), np.zeros(R), np.zeros(R), np.full(n + 1, -1, np.int32))


def clone_state(st):
    return tuple(a.copy() for a in st)


def state_routes(st):
    routes, rl = st[0], st[1]
    return [routes[r, :rl[r]].tolist() for r in range(len(rl)) if rl[r] > 0]


def flat_routes(routes):
    rid = [r for r, route in enumerate(routes) for _ in route]
    cs = [c for route in routes for c in route]
    return np.array(rid, np.int32), np.array(cs, np.int32)


def solomon_i1(inst, lam=2.0):
    """Solomon (1987) sequential insertion I1 (mu = 1, alpha1 = 1, alpha2 = 0), seeded with the farthest customer."""
    n = inst.n
    st = alloc_state(n, inst.max_cnt + 2, n)
    C.load_routes(st, inst.ins, np.zeros(0, np.int32), np.zeros(0, np.int32))
    unrouted = set(range(1, n + 1))
    r = 0
    while unrouted:
        seed = max(unrouted, key=lambda u: inst.d[0, u])
        C.insert_at(st, inst.ins, r, 0, seed)
        unrouted.discard(seed)
        while True:
            best_u, best_p, best_c2 = -1, -1, -1e18
            for u in unrouted:
                c, p = C.best_in_route(st, inst.ins, u, r, 0.0)
                if c < C.INF and lam * inst.d[0, u] - c > best_c2:
                    best_u, best_p, best_c2 = u, p, lam * inst.d[0, u] - c
            if best_u < 0:
                break
            C.insert_at(st, inst.ins, r, best_p, best_u)
            unrouted.discard(best_u)
        r += 1
    return state_routes(st)


class Engine:
    """One ALNS run. Baselines, the RL environment and the evaluation runner all drive it via `segment()`."""

    def __init__(self, inst, seed, T=5000, red_iters=None, eta=0.025, w_init=0.05, spare=6):
        self.inst, self.T, self.eta = inst, T, eta
        C.seed_rng(int(seed))
        routes0 = solomon_i1(inst)
        self.nv_init = len(routes0)
        R = min(inst.n, self.nv_init + spare)
        self.st = alloc_state(R, inst.max_cnt + 2, inst.n)
        C.load_routes(self.st, inst.ins, *flat_routes(routes0))
        self.bst, self.cst = clone_state(self.st), clone_state(self.st)
        self.rem = np.zeros(inst.n + 1, np.int32)
        self.unpl = np.zeros(inst.n + 1, np.int32)
        self.f_init = float(C.fval(self.st))
        td0 = C.nv_td(self.st)[1]

        # ---- shared vehicle-reduction phase
        t0 = time.perf_counter()
        red_iters = int(10 * inst.n) if red_iters is None else red_iters
        if red_iters > 0:
            tau_r = w_init * td0 / (inst.n * np.log(2.0))
            C.run_reduction(self.st, self.bst, self.cst, inst.ins, red_iters,
                            max(2, int(0.05 * inst.n)), max(3, int(0.15 * inst.n)),
                            self.rem, self.unpl, eta, tau_r)
            C.copy_state(self.st, self.bst)
        self.t_red = time.perf_counter() - t0

        # ---- SA schedule of the main phase
        self.tau0 = w_init * C.nv_td(self.st)[1] / np.log(2.0)
        self.cool = 0.01 ** (1.0 / T)
        f = float(C.fval(self.st))
        self.sc = np.array([self.tau0, self.cool, 0.0, f, f])
        self.uses = np.zeros(N_OPS, np.int64)
        self.succ = np.zeros(N_OPS, np.int64)
        self.scores = np.zeros(N_OPS)
        self.w = np.ones(N_OPS)                       # roulette-wheel weights
        self.iter = 0
        self.f_start = f
        self.trace = [(0, 0.0, f)]
        self.t_run = 0.0

    def segment(self, K, pD, pR, qmin, qmax, tau_mult=1.0, rho=RHO):
        """Run K iterations with fixed sampling distributions, destroy range and temperature deviation,
        then apply the roulette-wheel weight update with the segment's scores."""
        self.sc[0] = self.tau0 * self.cool ** self.iter * tau_mult   # deviation around the schedule, not cumulative
        self.uses[:] = 0
        self.succ[:] = 0
        self.scores[:] = 0
        t0 = time.perf_counter()
        C.run_segment(self.st, self.bst, self.cst, self.inst.ins, int(K),
                      np.ascontiguousarray(pD, dtype=np.float64), np.ascontiguousarray(pR, dtype=np.float64),
                      int(qmin), int(qmax), self.sc, self.uses, self.succ, self.scores,
                      self.rem, self.unpl, self.eta)
        self.t_run += time.perf_counter() - t0
        self.iter += int(K)
        self.trace.append((self.iter, self.t_run + self.t_red, float(self.sc[4])))
        mean_score = np.where(self.uses > 0, self.scores / np.maximum(self.uses, 1), 0.0)
        self.w = np.maximum((1 - rho) * self.w + rho * mean_score, 1e-3)

    def rw_probs(self):
        wd, wr = self.w[:C.N_D], self.w[C.N_D:]
        return wd / wd.sum(), wr / wr.sum()

    def q_range(self, ratio=None, qhi=Q_HI):
        """ratio=None: q ~ U[5 %, qhi] of n; otherwise a fixed q = ratio * n."""
        n = self.inst.n
        if ratio is None:
            return max(2, round(Q_LO * n)), max(3, round(qhi * n))
        q = max(2, int(round(ratio * n)))
        return q, q

    @property
    def f_best(self):
        return float(self.sc[4])

    @property
    def f_cur(self):
        return float(self.sc[3])

    @property
    def nv_best(self):
        return int(C.nv_td(self.bst)[0])

    @property
    def td_best(self):
        return float(C.nv_td(self.bst)[1])

    @property
    def tau(self):
        return float(self.sc[0])

    def best_routes(self):
        return state_routes(self.bst)

    def time_total(self):
        return self.t_run + self.t_red


def run_baseline(inst, method, T, seed, K=100, rho=RHO, red_iters=None, qhi=Q_HI, fixed_q=None, tau_mult=1.0):
    """'uni': uniform operator sampling; 'rw': Ropke-Pisinger roulette wheel, weights updated every K iterations."""
    eng = Engine(inst, seed, T, red_iters)
    qmin, qmax = eng.q_range(fixed_q, qhi)
    uni = (np.full(C.N_D, 1 / C.N_D), np.full(C.N_R, 1 / C.N_R))
    while eng.iter < T:
        pD, pR = eng.rw_probs() if method == "rw" else uni
        eng.segment(min(K, T - eng.iter), pD, pR, qmin, qmax, tau_mult, rho)
    return eng
