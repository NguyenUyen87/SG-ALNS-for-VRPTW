"""Stress test of the kernels: every destroy/repair result is feasible and cached costs are exact."""
import os, sys, tempfile, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgalns import core as C
from sgalns.instance import generate, check_solution, read_solomon, write_solomon
from sgalns.alns import Engine, solomon_i1, run_baseline, state_routes

rng = np.random.default_rng(0)
t0 = time.time()
for kind, wide in [("R", False), ("C", False), ("RC", True)]:
    inst = generate(100, kind, wide, rng)
    ok, nv, td, msg = check_solution(inst, solomon_i1(inst))
    assert ok, msg
    print(f"{inst.name}: I1 NV {nv} TD {td:.1f}")
    path = os.path.join(tempfile.gettempdir(), "sgalns_x.txt")
    write_solomon(inst, path)
    i2 = read_solomon(path)
    assert abs(i2.d - inst.d).max() < 1e-3 and i2.cap == inst.cap
    eng = Engine(inst, 1, T=1000, red_iters=0)
    st, cst = eng.st, eng.cst
    for it in range(3000):
        C.copy_state(cst, st)
        op, rep, q = int(rng.integers(0, 6)), int(rng.integers(0, 3)), int(rng.integers(2, 30))
        k = C.destroy(op, cst, inst.ins, q, eng.rem)
        assert C.repair(cst, inst.ins, eng.rem, k, rep, True, 0.025, eng.unpl) == 0
        if it % 50 == 0:
            ok, nv, td, msg = check_solution(inst, state_routes(cst))
            assert ok, (op, rep, msg)
            nv2, td2 = C.nv_td(cst)
            assert nv == nv2 and abs(td - td2) < 1e-6
        if C.fval(cst) <= C.fval(st) + 5:
            C.copy_state(st, cst)
    for m in ("uni", "rw"):
        e = run_baseline(inst, m, 2000, 3)
        ok, nv, td, msg = check_solution(inst, e.best_routes())
        assert ok, msg
        print(f"  {m:3s} NV {nv} TD {td:.2f} (I1 NV {e.nv_init})  {e.time_total():.2f}s")
print(f"core OK {time.time() - t0:.1f}s")
