"""PPO (clipped surrogate, GAE) for the segment-level ALNS MDP with CPU-parallel environments.

Each update collects N complete episodes (one ALNS run each), normalises rewards by a running std, runs
`epochs` passes of minibatch PPO, and every `val_every` updates validates the greedy policy against ALNS-RW
on fixed generated instances. The final checkpoint (last.pt) is the one used for evaluation.
"""
import csv
import os
import time
import numpy as np
import torch

from .alns import run_baseline
from .env import VecEnv
from .instance import random_instance
from .methods import run_policy, NetPolicy
from .policy import Policy, collate, save_policy


class RunningStd:
    def __init__(self):
        self.n, self.mean, self.m2 = 0, 0.0, 0.0

    def update(self, x):
        for v in np.asarray(x).ravel():               # Welford
            self.n += 1
            d = v - self.mean
            self.mean += d / self.n
            self.m2 += d * (v - self.mean)

    @property
    def std(self):
        return max(np.sqrt(self.m2 / max(self.n - 1, 1)), 1e-3)


def validation_set(env_cfg, n_inst=24, size=100, seed=12345):
    rng = np.random.default_rng(seed)
    insts = [random_instance(rng, (size,)) for _ in range(n_inst)]
    ref = [(e.nv_best, e.td_best) for e in (run_baseline(inst, "rw", env_cfg["T"], 777 + i) for i, inst in enumerate(insts))]
    return insts, np.array(ref)


def validate(policy, env_cfg, insts, ref):
    """Score vs ALNS-RW with the reward weighting: 5 per vehicle saved + % distance saved (mean over instances)."""
    policy.eval()
    out = np.array([(e.nv_best, e.td_best) for e in
                    (run_policy(inst, NetPolicy(policy), env_cfg, env_cfg["T"], 777 + i) for i, inst in enumerate(insts))])
    policy.train()
    score = 5.0 * (ref[:, 0] - out[:, 0]) + 100 * (ref[:, 1] - out[:, 1]) / ref[:, 1]
    return float(score.mean()), float(np.mean(1e4 * out[:, 0] + out[:, 1]))


def gae(rewards, values, gamma, lam):
    """Advantages for one complete episode (terminal value 0)."""
    H = len(rewards)
    adv = np.zeros(H)
    last = 0.0
    for t in reversed(range(H)):
        nxt = values[t + 1] if t + 1 < H else 0.0
        last = rewards[t] + gamma * nxt - values[t] + gamma * lam * last
        adv[t] = last
    return adv


def train(env_cfg, ppo_cfg, model_cfg, out_dir, seed=0, log=print):
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, ppo_cfg.get("torch_threads", 2)))
    N, K = ppo_cfg["n_envs"], env_cfg["K"]
    H = env_cfg["T"] // K
    n_updates = max(1, ppo_cfg["episodes"] // N)
    max_sec = 60.0 * ppo_cfg.get("max_minutes", 0) or float("inf")
    policy = Policy(**model_cfg)
    opt = torch.optim.Adam(policy.parameters(), lr=ppo_cfg["lr"])
    log(f"parameters: {sum(p.numel() for p in policy.parameters()):,} | {N} envs x {H} decisions | {n_updates} updates")

    val_insts, val_ref = validation_set(env_cfg, ppo_cfg.get("val_instances", 24), ppo_cfg.get("val_size", 100))
    sizes = list(env_cfg.get("sizes", [50, 100]))
    curriculum = ppo_cfg.get("curriculum", True) and len(sizes) > 1
    venv = VecEnv({**env_cfg, "sizes": sizes[:1] if curriculum else sizes}, N, seed + 1)
    rstd = RunningStd()
    t_rng = np.random.default_rng(seed + 7)
    best, t0 = -1e9, time.time()
    with open(os.path.join(out_dir, "train_log.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["update", "episodes", "mean_return", "loss_pi", "loss_v", "entropy", "val_impr_vs_rw_pct", "val_f", "time_s",
                    "top_action_share", "n_actions"])
        try:
            for u in range(n_updates):
                if curriculum and u == int(ppo_cfg.get("curriculum_switch", 0.4) * n_updates):
                    venv.set_cfg(sizes=sizes)
                    log(f"curriculum: sizes -> {sizes}")
                if env_cfg.get("T_choices"):                   # budget randomisation: one T per update
                    Tu = int(t_rng.choice(env_cfg["T_choices"]))
                    venv.set_cfg(T=Tu)
                    H = Tu // K

                # ---- rollout: N complete episodes of H decisions
                obs = venv.reset()
                O, A, LP, V, R = ([[] for _ in range(N)] for _ in range(5))
                for _ in range(H):
                    a, lp, v = policy.act(obs)
                    nobs, r, _, _ = venv.step(a)
                    for i in range(N):
                        O[i].append(obs[i]); A[i].append(a[i]); LP[i].append(lp[i]); V[i].append(v[i]); R[i].append(r[i])
                    obs = nobs
                ep_ret = np.array([sum(R[i]) for i in range(N)])
                # diversity of the sampled actions in this rollout (logging only; no random numbers drawn)
                _, cnt = np.unique(np.array([x for Ai in A for x in Ai]).reshape(-1, 3), axis=0, return_counts=True)
                top_share, n_act = cnt.max() / cnt.sum(), len(cnt)
                rstd.update(np.concatenate([np.array(R[i]) for i in range(N)]))
                advs = [gae(np.array(R[i]) / rstd.std, np.array(V[i]), ppo_cfg["gamma"], ppo_cfg["lam"]) for i in range(N)]
                rets = [advs[i] + np.array(V[i]) for i in range(N)]
                obs_f = [o for Oi in O for o in Oi]
                act = torch.as_tensor(np.array([x for Ai in A for x in Ai]), dtype=torch.long)
                old_lp = torch.as_tensor(np.array([x for Li in LP for x in Li]), dtype=torch.float32)
                adv = torch.as_tensor(np.concatenate(advs), dtype=torch.float32)
                ret = torch.as_tensor(np.concatenate(rets), dtype=torch.float32)
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                # ---- PPO epochs (entropy coefficient annealed linearly)
                ent_c = ppo_cfg["ent_start"] + u / max(n_updates - 1, 1) * (ppo_cfg["ent_end"] - ppo_cfg["ent_start"])
                M = len(obs_f)
                lpi = lv = ent_m = 0.0
                nb = 0
                for _ in range(ppo_cfg["epochs"]):
                    perm = torch.randperm(M)
                    for s in range(0, M, ppo_cfg["minibatch"]):
                        idx = perm[s:s + ppo_cfg["minibatch"]].tolist()
                        logp, ent, val = policy.evaluate(collate([obs_f[j] for j in idx]), act[idx])
                        ratio = torch.exp(logp - old_lp[idx])
                        l_pi = -torch.min(ratio * adv[idx],
                                          torch.clamp(ratio, 1 - ppo_cfg["clip"], 1 + ppo_cfg["clip"]) * adv[idx]).mean()
                        l_v = ((val - ret[idx]) ** 2).mean()
                        loss = l_pi + 0.5 * l_v - ent_c * ent.mean()
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
                        opt.step()
                        lpi += l_pi.item(); lv += l_v.item(); ent_m += ent.mean().item(); nb += 1

                # ---- validation / checkpoints
                out_of_time = time.time() - t0 > max_sec
                vi = vf = float("nan")
                if (u + 1) % ppo_cfg["val_every"] == 0 or u == n_updates - 1 or out_of_time:
                    vi, vf = validate(policy, env_cfg, val_insts, val_ref)
                    save_policy(policy, os.path.join(out_dir, "last.pt"), env_cfg, {"update": u + 1, "episodes": (u + 1) * N})
                    if vi > best:
                        best = vi
                        save_policy(policy, os.path.join(out_dir, "best.pt"), env_cfg, {"val_impr": vi, "update": u + 1})
                w.writerow([u + 1, (u + 1) * N, ep_ret.mean(), lpi / nb, lv / nb, ent_m / nb, vi, vf, time.time() - t0, top_share, n_act])
                fh.flush()
                log(f"[{u + 1:4d}/{n_updates}] return {ep_ret.mean():7.3f} | pi {lpi / nb:+.3f} v {lv / nb:.3f} "
                    f"H {ent_m / nb:.2f} | val vs RW {vi:+.3f} (best {best:+.3f}) | {time.time() - t0:.0f}s")
                if out_of_time:
                    log(f"time limit reached after {(u + 1) * N} episodes")
                    break
        finally:
            venv.close()
    return os.path.join(out_dir, "last.pt")
