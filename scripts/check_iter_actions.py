"""Record the greedy actions of the iteration-level agents (K=1), which evaluate.py does not store.
python scripts/check_iter_actions.py results/iter_gnn/last.pt results/iter_mlp/last.pt -> results/iter_actions.csv"""
import glob, os, sys
from collections import Counter
import pandas as pd
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from sgalns.alns import Engine
from sgalns.env import build_obs, decode_action
from sgalns.instance import read_solomon
from sgalns.policy import load_policy

torch.set_num_threads(4)
rows = []
for path in sys.argv[1:]:
    net, cfg = load_policy(path)
    for f in sorted(glob.glob("data/solomon/*.txt"))[::4]:            # 14 instances, one seed, T=5000
        inst = read_solomon(f)
        eng = Engine(inst, 0, 5000)
        cnt = Counter()
        while eng.iter < 5000:
            a = net.act([build_obs(eng, cfg.get("dyn", True))], greedy=True)[0][0]
            cnt[tuple(int(x) for x in a)] += 1
            pD, pR, qmin, qmax, tm = decode_action(a, eng, cfg)
            eng.segment(1, pD, pR, qmin, qmax, tm)
        for k, v in cnt.items():
            rows.append(dict(model=path.split("/")[-2], instance=inst.name, action="-".join(map(str, k)), share=v / 5000))
df = pd.DataFrame(rows)
df.to_csv("results/iter_actions.csv", index=False)
print(df.groupby(["model", "action"]).share.agg(["count", "mean", "min"]))
