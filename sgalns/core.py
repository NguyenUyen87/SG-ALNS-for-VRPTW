"""Numba kernels of the VRPTW ALNS (hard time windows).

Solution state ``st = (routes, rl, bt, fs, ld, rc, where)``:
    routes int32[R, L]     customers of each route, in visiting order (0-padded)
    rl     int32[R]        number of customers per route
    bt     float64[R, L+1] start of service per position; bt[r, rl[r]] = return time at the depot
    fs     float64[R, L+1] forward time slack (Savelsbergh 1992), depot return included
    ld     float64[R]      route load
    rc     float64[R]      route distance
    where  int32[n+1]      route of each customer, -1 when it is not routed
Instance ``ins = (d, dem, e, l, s, cap, adj, dmax)``; node 0 is the depot.

Insertion feasibility is tested in O(1) per position with the forward time slack, so a repair scan of
a route is O(route length). Objective: f = OMEGA * NV + TD (vehicles first, then distance).
"""
import numpy as np
from numba import njit

OMEGA = 1e4
INF = 1e18
EPS = 1e-9
BIG = 1e7
SIG1, SIG2, SIG3 = 33.0, 9.0, 13.0   # Ropke & Pisinger scores: new best / better than current / accepted worse
N_D, N_R = 6, 3                      # destroy: random, worst, Shaw, string, route, waiting time; repair: greedy, regret-3, noisy greedy


@njit(cache=True)
def seed_rng(s):
    np.random.seed(s)


# ============================================================================ solution state
@njit(cache=True)
def copy_state(dst, src):
    dst[0][:] = src[0]
    dst[1][:] = src[1]
    dst[2][:] = src[2]
    dst[3][:] = src[3]
    dst[4][:] = src[4]
    dst[5][:] = src[5]
    dst[6][:] = src[6]


@njit(cache=True)
def rebuild(st, ins, r):
    """Recompute schedule, forward slack, load and distance of route r (and `where` of its customers)."""
    routes, rl, bt, fs, ld, rc, where = st
    d, dem, e, l, s, cap, adj, dmax = ins
    n = rl[r]
    if n == 0:
        bt[r, 0] = 0.0
        fs[r, 0] = l[0]
        ld[r] = 0.0
        rc[r] = 0.0
        return
    prev, dep, load, cost = 0, 0.0, 0.0, 0.0
    for p in range(n):
        c = routes[r, p]
        a = dep + d[prev, c]
        b = max(a, e[c])
        bt[r, p] = b
        dep = b + s[c]
        cost += d[prev, c]
        load += dem[c]
        where[c] = r
        prev = c
    back = dep + d[prev, 0]
    bt[r, n] = back
    cost += d[prev, 0]
    ld[r] = load
    rc[r] = cost
    fs[r, n] = l[0] - back
    for p in range(n - 1, -1, -1):
        c = routes[r, p]
        nxt = routes[r, p + 1] if p + 1 < n else 0
        wait = max(0.0, bt[r, p + 1] - (bt[r, p] + s[c] + d[c, nxt]))
        fs[r, p] = min(l[c] - bt[r, p], wait + fs[r, p + 1])


@njit(cache=True)
def nv_td(st):
    rl, rc = st[1], st[5]
    nv, td = 0, 0.0
    for r in range(rl.shape[0]):
        if rl[r] > 0:
            nv += 1
            td += rc[r]
    return nv, td


@njit(cache=True)
def fval(st):
    nv, td = nv_td(st)
    return OMEGA * nv + td


@njit(cache=True)
def insert_at(st, ins, r, p, u):
    routes, rl = st[0], st[1]
    n = rl[r]
    for k in range(n, p, -1):
        routes[r, k] = routes[r, k - 1]
    routes[r, p] = u
    rl[r] = n + 1
    rebuild(st, ins, r)


@njit(cache=True)
def remove_cust(st, ins, u):
    routes, rl, where = st[0], st[1], st[6]
    r = where[u]
    n = rl[r]
    p = 0
    while routes[r, p] != u:
        p += 1
    for k in range(p, n - 1):
        routes[r, k] = routes[r, k + 1]
    routes[r, n - 1] = 0
    rl[r] = n - 1
    where[u] = -1
    rebuild(st, ins, r)


@njit(cache=True)
def load_routes(st, ins, rid, rcust):
    """Fill a state from flat arrays: customer rcust[i] is visited by route rid[i] (in order)."""
    routes, rl, where = st[0], st[1], st[6]
    routes[:] = 0
    rl[:] = 0
    where[:] = -1
    for i in range(rcust.shape[0]):
        r = rid[i]
        routes[r, rl[r]] = rcust[i]
        rl[r] += 1
    for r in range(rl.shape[0]):
        rebuild(st, ins, r)


# ============================================================================ insertion / repair
@njit(cache=True)
def best_in_route(st, ins, u, r, noise_amp):
    """Cheapest feasible position for u in route r -> (cost increase, position) or (INF, -1)."""
    routes, rl, bt, fs, ld = st[0], st[1], st[2], st[3], st[4]
    d, dem, e, l, s, cap = ins[0], ins[1], ins[2], ins[3], ins[4], ins[5]
    n = rl[r]
    if ld[r] + dem[u] > cap or n >= routes.shape[1] - 2:
        return INF, -1
    best, bp = INF, -1
    prev, dep = 0, 0.0
    for p in range(n + 1):
        if p > 0:
            prev = routes[r, p - 1]
            dep = bt[r, p - 1] + s[prev]
        nxt = routes[r, p] if p < n else 0
        bu = max(dep + d[prev, u], e[u])
        if bu > l[u]:
            continue
        du = bu + s[u]
        bn = max(du + d[u, nxt], e[nxt]) if p < n else du + d[u, 0]
        if bn - bt[r, p] > fs[r, p] + EPS:          # push-forward exceeds the forward slack
            continue
        c = d[prev, u] + d[u, nxt] - d[prev, nxt]
        if noise_amp > 0.0:
            c += noise_amp * (2.0 * np.random.random() - 1.0)
        if c < best:
            best, bp = c, p
    return best, bp


@njit(cache=True)
def _fill_column(st, ins, rem, placed, m, cost_c, pos_c, r, noise_amp):
    """Cache the best insertion of every unplaced customer into route r (opening a route costs OMEGA)."""
    pen = OMEGA if st[1][r] == 0 else 0.0
    for i in range(m):
        if placed[i]:
            continue
        c, p = best_in_route(st, ins, rem[i], r, noise_amp)
        cost_c[i, r] = c + pen if c < INF else c
        pos_c[i, r] = p


@njit(cache=True)
def repair(st, ins, rem, m, mode, allow_new, eta, unpl):
    """Insert rem[:m]. mode 0 = greedy, 1 = regret-3, 2 = greedy with noise (amplitude eta*dmax).
    Only routes changed by the last insertion are re-evaluated. Returns the number of customers that
    could not be inserted (written to unpl)."""
    rl = st[1]
    R = rl.shape[0]
    noise_amp = eta * ins[7] if mode == 2 else 0.0
    cost_c = np.full((m, R), INF)
    pos_c = np.full((m, R), -1, np.int32)
    placed = np.zeros(m, np.bool_)
    empty_r = -1
    if allow_new:
        for r in range(R):
            if rl[r] == 0:
                empty_r = r
                break
    for r in range(R):
        if rl[r] > 0 or r == empty_r:
            _fill_column(st, ins, rem, placed, m, cost_c, pos_c, r, noise_amp)
    nleft = m
    while nleft > 0:
        best_val, bi, br = -INF, -1, -1
        for i in range(m):
            if placed[i]:
                continue
            c1, c2, c3, r1 = INF, INF, INF, -1
            for r in range(R):
                c = cost_c[i, r]
                if c < c1:
                    c3, c2, c1, r1 = c2, c1, c, r
                elif c < c2:
                    c3, c2 = c2, c
                elif c < c3:
                    c3 = c
            if r1 < 0:
                continue
            if mode == 1:
                x2 = c2 if c2 < INF else c1 + BIG
                x3 = c3 if c3 < INF else c1 + BIG
                val = (x2 - c1) + (x3 - c1) - 1e-6 * c1
            else:
                val = -c1
            if val > best_val:
                best_val, bi, br = val, i, r1
        if bi < 0:
            break
        insert_at(st, ins, br, pos_c[bi, br], rem[bi])
        placed[bi] = True
        nleft -= 1
        _fill_column(st, ins, rem, placed, m, cost_c, pos_c, br, noise_amp)
        if allow_new and br == empty_r:                # the spare route was used: open the next empty one
            empty_r = -1
            for r in range(R):
                if rl[r] == 0:
                    empty_r = r
                    break
            if empty_r >= 0:
                _fill_column(st, ins, rem, placed, m, cost_c, pos_c, empty_r, noise_amp)
    k = 0
    for i in range(m):
        if not placed[i]:
            unpl[k] = rem[i]
            k += 1
    return k


# ============================================================================ destroy operators
@njit(cache=True)
def _routed(st, n, cand):
    where = st[6]
    m = 0
    for u in range(1, n + 1):
        if where[u] >= 0:
            cand[m] = u
            m += 1
    return m


@njit(cache=True)
def d_random(st, ins, q, rem):
    n = st[6].shape[0] - 1
    cand = np.empty(n, np.int32)
    m = _routed(st, n, cand)
    q = min(q, m)
    for i in range(q):                                  # partial Fisher-Yates
        j = i + int(np.random.random() * (m - i))
        cand[i], cand[j] = cand[j], cand[i]
        rem[i] = cand[i]
        remove_cust(st, ins, cand[i])
    return q


@njit(cache=True)
def d_worst(st, ins, q, rem):
    """Randomised worst removal (p = 3): savings are recomputed after each removal."""
    routes, rl = st[0], st[1]
    d = ins[0]
    n = st[6].shape[0] - 1
    sav = np.empty(n)
    cand = np.empty(n, np.int32)
    k = 0
    while k < q:
        m = 0
        for r in range(rl.shape[0]):
            nr = rl[r]
            for p in range(nr):
                c = routes[r, p]
                prev = routes[r, p - 1] if p > 0 else 0
                nxt = routes[r, p + 1] if p + 1 < nr else 0
                cand[m] = c
                sav[m] = d[prev, c] + d[c, nxt] - d[prev, nxt]
                m += 1
        if m == 0:
            break
        order = np.argsort(-sav[:m])
        c = cand[order[int(np.random.random() ** 3.0 * m)]]
        rem[k] = c
        k += 1
        remove_cust(st, ins, c)
    return k


@njit(cache=True)
def d_shaw(st, ins, q, rem):
    """Shaw relatedness removal (distance 9, time 3, demand 2; p = 6)."""
    routes, rl, bt, where = st[0], st[1], st[2], st[6]
    d, dem, l, dmax = ins[0], ins[1], ins[3], ins[7]
    n = where.shape[0] - 1
    bs = np.zeros(n + 1)
    for r in range(rl.shape[0]):
        for p in range(rl[r]):
            bs[routes[r, p]] = bt[r, p]
    cand = np.empty(n, np.int32)
    m0 = _routed(st, n, cand)
    if m0 == 0:
        return 0
    q = min(q, m0)
    H, dm = l[0], dem.max()
    isrem = np.zeros(n + 1, np.bool_)
    rel = np.empty(n)
    tmp = np.empty(n, np.int32)
    seed = cand[int(np.random.random() * m0)]
    rem[0] = seed
    isrem[seed] = True
    k = 1
    while k < q:
        ref = rem[int(np.random.random() * k)]
        m = 0
        for u in range(1, n + 1):
            if where[u] >= 0 and not isrem[u]:
                tmp[m] = u
                rel[m] = (9.0 * d[ref, u] / dmax + 3.0 * abs(bs[ref] - bs[u]) / H
                          + 2.0 * abs(dem[ref] - dem[u]) / dm)
                m += 1
        if m == 0:
            break
        order = np.argsort(rel[:m])
        u = tmp[order[int(np.random.random() ** 6.0 * m)]]
        rem[k] = u
        isrem[u] = True
        k += 1
    for i in range(k):
        remove_cust(st, ins, rem[i])
    return k


@njit(cache=True)
def d_string(st, ins, q, rem):
    """String removal (SISR, Christiaens & Vanden Berghe 2020), without the split-string variant."""
    routes, rl, where = st[0], st[1], st[6]
    adj = ins[6]
    n = where.shape[0] - 1
    R = rl.shape[0]
    cand = np.empty(n, np.int32)
    m0 = _routed(st, n, cand)
    if m0 == 0:
        return 0
    nr = 0
    for r in range(R):
        if rl[r] > 0:
            nr += 1
    ls_max = min(10.0, m0 / nr)
    ks_max = max(1.0, 4.0 * q / (1.0 + ls_max) - 1.0)
    ks = int(1.0 + np.random.random() * ks_max)
    seed = cand[int(np.random.random() * m0)]
    done = np.zeros(R, np.bool_)
    buf = np.empty(n, np.int32)
    k, cnt = 0, 0
    for idx in range(-1, adj.shape[1]):
        c = seed if idx < 0 else adj[seed - 1, idx]
        r = where[c]
        if r < 0 or done[r]:
            continue
        lt = min(int(1.0 + np.random.random() * min(float(rl[r]), ls_max)), rl[r])
        p = 0
        while routes[r, p] != c:
            p += 1
        lo = max(0, p - lt + 1)
        hi = min(p, rl[r] - lt)
        s0 = lo + int(np.random.random() * (hi - lo + 1))
        for j in range(lt):
            buf[j] = routes[r, s0 + j]
        for j in range(lt):
            rem[k] = buf[j]
            k += 1
            remove_cust(st, ins, buf[j])
        done[r] = True
        cnt += 1
        if cnt >= ks:
            break
    return k


@njit(cache=True)
def d_route(st, ins, q, rem):
    """Remove whole routes, preferring short ones, until at least q customers are removed."""
    routes, rl = st[0], st[1]
    R = rl.shape[0]
    key = np.empty(R)
    for r in range(R):
        key[r] = rl[r] + 1.5 * np.random.random() if rl[r] > 0 else INF
    order = np.argsort(key)
    k = 0
    for oi in range(R):
        r = order[oi]
        if rl[r] == 0 or (k >= q and k > 0):
            break
        for p in range(rl[r] - 1, -1, -1):
            c = routes[r, p]
            rem[k] = c
            k += 1
            remove_cust(st, ins, c)
    return k


@njit(cache=True)
def d_wait(st, ins, q, rem):
    """Waiting-time removal: randomised (p = 3) removal of customers with the longest wait."""
    routes, rl, bt = st[0], st[1], st[2]
    d, s = ins[0], ins[4]
    n = st[6].shape[0] - 1
    w = np.empty(n)
    cand = np.empty(n, np.int32)
    m = 0
    for r in range(rl.shape[0]):
        dep, prev = 0.0, 0
        for p in range(rl[r]):
            c = routes[r, p]
            cand[m] = c
            w[m] = bt[r, p] - (dep + d[prev, c])
            m += 1
            dep = bt[r, p] + s[c]
            prev = c
    q = min(q, m)
    order = np.argsort(-w[:m])
    chosen = np.zeros(m, np.bool_)
    k = 0
    while k < q:
        j = order[int(np.random.random() ** 3.0 * m)]
        if chosen[j]:                                   # fall back to the longest wait not yet chosen
            for t in range(m):
                if not chosen[order[t]]:
                    j = order[t]
                    break
        chosen[j] = True
        rem[k] = cand[j]
        k += 1
    for i in range(k):
        remove_cust(st, ins, rem[i])
    return k


@njit(cache=True)
def destroy(op, st, ins, q, rem):
    if op == 0:
        return d_random(st, ins, q, rem)
    if op == 1:
        return d_worst(st, ins, q, rem)
    if op == 2:
        return d_shaw(st, ins, q, rem)
    if op == 3:
        return d_string(st, ins, q, rem)
    if op == 4:
        return d_route(st, ins, q, rem)
    return d_wait(st, ins, q, rem)


# ============================================================================ ALNS loops
@njit(cache=True)
def sample(p):
    u = np.random.random()
    c = 0.0
    for i in range(p.shape[0]):
        c += p[i]
        if u < c:
            return i
    return p.shape[0] - 1


@njit(cache=True)
def run_segment(st, bst, cst, ins, K, pD, pR, qmin, qmax, sc, uses, succ, scores, rem, unpl, eta):
    """K ALNS iterations with simulated-annealing acceptance.
    sc = [tau, cooling factor, iterations since last best, f_cur, f_best] (updated in place)."""
    tau, cool, stag = sc[0], sc[1], sc[2]
    f_cur = fval(st)
    f_best = fval(bst)
    for _ in range(K):
        copy_state(cst, st)
        di = sample(pD)
        ri = sample(pR)
        q = qmin + int(np.random.random() * (qmax - qmin + 1))
        k = destroy(di, cst, ins, q, rem)
        nu = repair(cst, ins, rem, k, ri, True, eta, unpl)
        uses[di] += 1
        uses[N_D + ri] += 1
        stag += 1.0
        if nu == 0:
            f_new = fval(cst)
            score = 0.0
            if f_new < f_best - 1e-7:
                copy_state(bst, cst)
                f_best = f_new
                score, stag, acc = SIG1, 0.0, True
            elif f_new < f_cur - 1e-7:
                score, acc = SIG2, True
            else:
                acc = np.random.random() < np.exp((f_cur - f_new) / tau)
                if acc and f_new > f_cur + 1e-7:
                    score = SIG3
            if f_new < f_cur - 1e-7:
                succ[di] += 1
                succ[N_D + ri] += 1
            scores[di] += score
            scores[N_D + ri] += score
            if acc:
                copy_state(st, cst)
                f_cur = f_new
        tau *= cool
    sc[0], sc[2], sc[3], sc[4] = tau, stag, f_cur, f_best


@njit(cache=True)
def _bank_route(st, ins, bank):
    """Move the customers of a (randomly perturbed) shortest route into the request bank."""
    routes, rl = st[0], st[1]
    best, bk = -1, INF
    for r in range(rl.shape[0]):
        if rl[r] > 0:
            key = rl[r] + 0.5 * np.random.random()
            if key < bk:
                bk, best = key, r
    if best < 0:
        return 0
    nb = rl[best]
    for i in range(nb):
        bank[i] = routes[best, i]
    for i in range(nb):
        remove_cust(st, ins, bank[i])
    return nb


@njit(cache=True)
def run_reduction(st, bst, cst, ins, n_iter, qlo, qhi, rem, unpl, eta, tau):
    """Vehicle-reduction phase with a request bank (Ropke & Pisinger 2006, Sect. 3.7): no new routes may be
    opened; a solution is accepted if it empties the bank further, or by SA on distance at equal bank size.
    The best complete solution (fewest vehicles) is left in bst; returns the final bank size."""
    n = st[6].shape[0] - 1
    bank = np.empty(n, np.int32)
    copy_state(bst, st)
    nb = _bank_route(st, ins, bank)
    for _ in range(n_iter):
        if nb == 0:
            copy_state(bst, st)
            nb = _bank_route(st, ins, bank)
            if nb == 0:
                break
        copy_state(cst, st)
        di = int(np.random.random() * N_D)
        ri = int(np.random.random() * N_R)
        q = qlo + int(np.random.random() * (qhi - qlo + 1))
        k = destroy(di, cst, ins, q, rem)
        for i in range(nb):
            rem[k + i] = bank[i]
        nu = repair(cst, ins, rem, k + nb, ri, False, eta, unpl)
        if nu < nb:
            acc = True
        elif nu == nb:
            acc = np.random.random() < np.exp((nv_td(st)[1] - nv_td(cst)[1]) / tau)
        else:
            acc = False
        if acc:
            copy_state(st, cst)
            nb = nu
            for i in range(nu):
                bank[i] = unpl[i]
    if nb == 0:
        copy_state(bst, st)
    return nb


# ============================================================================ RL features
@njit(cache=True)
def node_dyn(st, ins, out):
    """out[n+1, 6] = wait/H, slack/H, FTS/H, cumulative load ratio, position ratio, routed flag."""
    routes, rl, bt, fs = st[0], st[1], st[2], st[3]
    d, dem, l, s, cap = ins[0], ins[1], ins[3], ins[4], ins[5]
    H = l[0]
    out[:] = 0.0
    for r in range(rl.shape[0]):
        n = rl[r]
        dep, prev, cum = 0.0, 0, 0.0
        for p in range(n):
            c = routes[r, p]
            cum += dem[c]
            out[c, 0] = (bt[r, p] - (dep + d[prev, c])) / H
            out[c, 1] = (l[c] - bt[r, p]) / H
            out[c, 2] = fs[r, p] / H
            out[c, 3] = cum / cap
            out[c, 4] = (p + 1.0) / n
            out[c, 5] = 1.0
            dep = bt[r, p] + s[c]
            prev = c


@njit(cache=True)
def sol_edges(st, src, dst):
    """Undirected edges (both directions) of the current routes, depot legs included; returns the count."""
    routes, rl = st[0], st[1]
    k = 0
    for r in range(rl.shape[0]):
        n = rl[r]
        if n == 0:
            continue
        prev = 0
        for p in range(n + 1):
            c = routes[r, p] if p < n else 0
            src[k], dst[k] = prev, c
            src[k + 1], dst[k + 1] = c, prev
            k += 2
            prev = c
    return k
