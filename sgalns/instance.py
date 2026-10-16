"""VRPTW instances: Solomon / Gehring-Homberger reader and writer, Solomon-like generator, independent checker."""
import os
import numpy as np


class Instance:
    """Node 0 is the depot. Distances are unrounded Euclidean."""

    def __init__(self, name, xy, demand, ready, due, service, cap, knn=10):
        self.name = name
        self.xy = np.asarray(xy, float)
        self.n = len(self.xy) - 1
        self.dem = np.asarray(demand, float)
        self.e = np.asarray(ready, float)
        self.l = np.asarray(due, float)
        self.s = np.asarray(service, float)
        self.cap = float(cap)
        self.d = np.sqrt(((self.xy[:, None, :] - self.xy[None, :, :]) ** 2).sum(-1))
        self.dmax = float(self.d.max())
        # customers sorted by distance from each customer (row i -> customer i+1), self excluded
        order = np.argsort(self.d[1:, 1:], axis=1)
        self.adj = np.ascontiguousarray(
            np.array([[j + 1 for j in row if j != i] for i, row in enumerate(order)], dtype=np.int32))
        # sparse k-NN graph for the policy: customer -> k nearest customers, depot <-> its k nearest customers
        k = min(knn, self.n - 1)
        src = np.repeat(np.arange(1, self.n + 1), k)
        dst = self.adj[:, :k].reshape(-1)
        dn = np.argsort(self.d[0, 1:])[:k] + 1
        self.knn_edges = np.stack([np.concatenate([src, np.zeros(k, int), dn]),
                                   np.concatenate([dst, dn, np.zeros(k, int)])]).astype(np.int64)
        # upper bound on customers per route (capacity), used as the route array width
        self.max_cnt = int((np.cumsum(np.sort(self.dem[1:])) <= self.cap).sum())
        self.ins = (self.d, self.dem, self.e, self.l, self.s, self.cap, self.adj, self.dmax)


def read_solomon(path, knn=10):
    """Solomon / Gehring-Homberger text format."""
    with open(path, encoding="utf-8") as f:
        lines = [ln.split() for ln in f.read().splitlines()]
    cap, rows = None, []
    for i, tok in enumerate(lines):
        if tok and tok[0].upper() == "NUMBER" and i + 1 < len(lines):
            cap = float(lines[i + 1][1])
        if len(tok) == 7:
            try:
                rows.append([float(x) for x in tok])
            except ValueError:
                pass
    a = np.array(rows)
    a = a[np.argsort(a[:, 0], kind="stable")]
    name = os.path.splitext(os.path.basename(path))[0]
    return Instance(name, a[:, 1:3], a[:, 3], a[:, 4], a[:, 5], a[:, 6], cap, knn)


def write_solomon(inst, path):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{inst.name}\n\nVEHICLE\nNUMBER     CAPACITY\n  {inst.n:d}    {inst.cap:g}\n\n")
        f.write("CUSTOMER\nCUST NO.  XCOORD.   YCOORD.    DEMAND   READY TIME  DUE DATE   SERVICE TIME\n\n")
        for i in range(inst.n + 1):
            f.write(f"{i:5d} {inst.xy[i, 0]:10.4f} {inst.xy[i, 1]:10.4f} {inst.dem[i]:8.0f} "
                    f"{inst.e[i]:10.4f} {inst.l[i]:10.4f} {inst.s[i]:8.0f}\n")


def _clustered(n, k, rng):
    centres = rng.uniform(10, 90, (k, 2))
    sig = rng.uniform(4, 10, k)
    idx = rng.integers(0, k, n)
    return np.clip(centres[idx] + rng.normal(0, 1, (n, 2)) * sig[idx, None], 0, 100)


def generate(n, kind, wide, rng, name=None):
    """Solomon-like instance. kind in {R, C, RC}; wide=True -> type-2 (long horizon, wide windows)."""
    if kind == "R":
        xy = rng.uniform(0, 100, (n, 2))
    elif kind == "C":
        xy = _clustered(n, int(rng.integers(5, 11)), rng)
    else:
        xy = np.vstack([rng.uniform(0, 100, (n // 2, 2)), _clustered(n - n // 2, 8, rng)])
    depot = np.array([50.0, 50.0])
    H = 1000.0 if wide else 230.0
    d0 = np.linalg.norm(xy - depot, axis=1)
    s = np.full(n, 10.0)
    centre = rng.uniform(d0, H - d0 - s)
    half = rng.uniform(30, 120, n) if wide else rng.uniform(5, 30, n)
    e = np.maximum(d0, centre - half)                  # every customer is reachable directly from the depot
    l = np.maximum(np.minimum(H - d0 - s, centre + half), e)
    q = rng.integers(1, 51, n).astype(float)
    cap = (700.0 if kind == "C" else 1000.0) if wide else 200.0
    return Instance(name or f"gen_{kind}{'2' if wide else '1'}_{n}",
                    np.vstack([depot, xy]), np.r_[0, q], np.r_[0, e], np.r_[H, l], np.r_[0, s], cap)


def random_instance(rng, sizes=(50, 100)):
    n = int(rng.choice(sizes))
    return generate(n, str(rng.choice(["R", "C", "RC"])), bool(rng.integers(0, 2)), rng)


def check_solution(inst, routes, tol=1e-6):
    """Check a solution from scratch. routes: list of customer lists. Returns (ok, NV, TD, message)."""
    seen, td = set(), 0.0
    for r in routes:
        if not r:
            continue
        t, prev, load = 0.0, 0, 0.0
        for c in r:
            if c in seen or c < 1 or c > inst.n:
                return False, 0, 0.0, f"customer {c} duplicated/invalid"
            seen.add(c)
            t = max(t + inst.d[prev, c], inst.e[c])
            if t > inst.l[c] + tol:
                return False, 0, 0.0, f"time window violated at {c}"
            t += inst.s[c]
            load += inst.dem[c]
            td += inst.d[prev, c]
            prev = c
        t += inst.d[prev, 0]
        td += inst.d[prev, 0]
        if t > inst.l[0] + tol:
            return False, 0, 0.0, "depot horizon violated"
        if load > inst.cap + tol:
            return False, 0, 0.0, "capacity violated"
    if len(seen) != inst.n:
        return False, 0, 0.0, "not all customers served"
    return True, sum(1 for r in routes if r), td, "ok"
