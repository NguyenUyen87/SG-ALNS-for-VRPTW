"""Random-action episodes: finite features, feasible result, rewards telescope to the total improvement."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgalns.env import SegmentEnv, VecEnv, NODE_DIM, EDGE_DIM, G_DIM, N_A1, N_A2, N_A3
from sgalns.instance import check_solution

if __name__ == "__main__":
    env = SegmentEnv(dict(T=1000, K=50, sizes=(50, 100), red_iters=300), seed=1)
    rng = np.random.default_rng(0)
    for ep in range(3):
        obs, _ = env.reset()
        nv0 = env.eng.nv_best
        assert obs["x"].shape[1] == NODE_DIM and obs["ea"].shape == (obs["ei"].shape[1], EDGE_DIM) and obs["g"].shape == (G_DIM,)
        tot, done = 0.0, False
        while not done:
            obs, r, done, _, _ = env.step((rng.integers(N_A1), rng.integers(N_A2), rng.integers(N_A3)))
            assert np.isfinite(obs["x"]).all() and np.isfinite(obs["g"]).all()
            tot += r
        e = env.eng
        ok, nv, td, msg = check_solution(e.inst, e.best_routes())
        assert ok, msg
        assert abs(tot - (100 * (env.f_init - e.f_best) / env.f_init + (nv0 - e.nv_best))) < 1e-6
        print(f"n={e.inst.n} return {tot:.3f} NV {nv0}->{nv}")
    v = VecEnv(dict(T=300, K=50, sizes=(50,), red_iters=100), 2, 1)
    v.reset()
    done = np.zeros(2, bool)
    while not done.all():
        _, r, done, info = v.step([np.array([3, 1, 1])] * 2)
    v.close()
    print("env OK")
