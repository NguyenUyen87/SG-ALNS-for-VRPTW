"""Train a policy:  python scripts/train.py --name main [--set env.K=25 ppo.episodes=800 seed=1 ...]"""
import argparse, os, sys
import yaml
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def set_path(cfg, dotted, value):
    *keys, last = dotted.split(".")
    d = cfg
    for k in keys:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(value)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--name", required=True)
    ap.add_argument("--set", nargs="*", default=[], help="section.key=value overrides")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    for s in a.set:
        set_path(cfg, *s.split("=", 1))
    from sgalns.ppo import train
    from sgalns.dqn import train_bdq
    out = os.path.join(a.out, a.name)
    os.makedirs(out, exist_ok=True)
    yaml.safe_dump(cfg, open(os.path.join(out, "config.yaml"), "w", encoding="utf-8"))
    env = cfg["env"]
    model = dict(cfg["model"], use_a1=env.get("use_a1", True), use_a2=env.get("use_a2", True),
                 use_a3=env.get("use_a3", True), expert=env.get("expert", True), ctx=env.get("ctx", False),
                 prior=env.get("prior", True))
    log = lambda m: print(m, flush=True)
    if cfg.get("algo", "ppo") == "bdq":
        train_bdq(env, cfg["bdq"], dict(cfg["model"], ctx=env.get("ctx", False)), out, seed=cfg.get("seed", 0), log=log)
    else:
        train(env, cfg["ppo"], model, out, seed=cfg.get("seed", 0), log=log)
