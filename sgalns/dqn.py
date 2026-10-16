"""Segment-level value-based agent: Branching Dueling Double DQN (BDQ, Tavakoli et al. 2018) with the same GINE
encoder as the PPO policy. One Q-branch per action component (a1: 19, a2: 5, a3: 3) on a shared state value.
Used as a second learner to test whether policy collapse is specific to PPO (cf. Nguyen et al. 2026)."""
import csv
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .env import VecEnv, N_A1, N_A2, N_A3
from .policy import Policy, collate
from .ppo import RunningStd, validation_set, validate

N_BRANCH = (N_A1, N_A2, N_A3)


class QPolicy(nn.Module):
    """Dueling branching Q-network on top of the Policy encoder (same embedding and trunk width)."""

    def __init__(self, h=64, layers=3, encoder="gnn", ctx=False, **_):
        super().__init__()
        self.cfg = dict(h=h, layers=layers, encoder=encoder, ctx=ctx, kind="bdq")
        self.enc = Policy(h=h, layers=layers, encoder=encoder, ctx=ctx, prior=False)
        z = 2 * h + 64
        self.trunk = nn.Sequential(nn.Linear(z, 128), nn.ReLU())
        self.value = nn.Linear(128, 1)
        self.adv = nn.ModuleList([nn.Linear(128, k) for k in N_BRANCH])

    def forward(self, bt):
        t = self.trunk(self.enc.embed(bt))
        v = self.value(t)
        return [v + a - a.mean(-1, keepdim=True) for a in (head(t) for head in self.adv)]

    @torch.no_grad()
    def act(self, obs_list, greedy=True, eps=0.0, rng=None):
        qs = self(collate(obs_list))
        a = torch.stack([q.argmax(-1) for q in qs], -1).numpy()
        if not greedy and eps > 0:
            rng = rng or np.random
            for i in range(len(obs_list)):
                for d, k in enumerate(N_BRANCH):
                    if rng.random() < eps:
                        a[i, d] = rng.integers(k) if hasattr(rng, "integers") else rng.randint(k)
        return a, None, None


def _compact(o):
    """Store graphs with int32 edges to halve replay memory."""
    return {"x": o["x"], "ei": o["ei"].astype(np.int32), "ea": o["ea"], "g": o["g"]}


def _expand(o):
    return {"x": o["x"], "ei": o["ei"].astype(np.int64), "ea": o["ea"], "g": o["g"]}


def train_bdq(env_cfg, cfg, model_cfg, out_dir, seed=0, log=print):
    """cfg keys: n_envs, episodes, lr, gamma, batch, buffer, target_every, eps_start, eps_end, eps_frac,
    updates_per_step, val_every (in rounds of n_envs episodes), torch_threads, curriculum(_switch)."""
    from .policy import save_policy
    os.makedirs(out_dir, exist_ok=True)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, cfg.get("torch_threads", 2)))
    N, H = cfg["n_envs"], env_cfg["T"] // env_cfg["K"]
    rounds = max(1, cfg["episodes"] // N)
    q, q_tgt = QPolicy(**model_cfg), QPolicy(**model_cfg)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=cfg["lr"])
    log(f"BDQ parameters: {sum(p.numel() for p in q.parameters()):,} | {N} envs x {H} decisions | {rounds} rounds")
    val_insts, val_ref = validation_set(env_cfg, cfg.get("val_instances", 24), cfg.get("val_size", 100))
    sizes = list(env_cfg.get("sizes", [50, 100]))
    curriculum = cfg.get("curriculum", True) and len(sizes) > 1
    venv = VecEnv({**env_cfg, "sizes": sizes[:1] if curriculum else sizes}, N, seed + 1)
    rstd = RunningStd()
    cap = cfg["buffer"]
    S, A, R, S2, D = [None] * cap, np.zeros((cap, 3), np.int64), np.zeros(cap, np.float32), [None] * cap, np.zeros(cap, np.float32)
    ptr = size = n_grad = 0
    total_steps = rounds * H
    best, t0, step = -1e9, time.time(), 0
    fh = open(os.path.join(out_dir, "train_log.csv"), "w", newline="")
    w = csv.writer(fh)
    w.writerow(["update", "episodes", "mean_return", "loss_pi", "loss_v", "entropy", "val_impr_vs_rw_pct", "val_f", "time_s"])
    try:
        for u in range(rounds):
            if curriculum and u == int(cfg.get("curriculum_switch", 0.4) * rounds):
                venv.set_cfg(sizes=sizes)
                log(f"curriculum: sizes -> {sizes}")
            obs = venv.reset()
            ep_ret = np.zeros(N)
            losses = []
            for h in range(H):
                frac = min(1.0, step / (cfg["eps_frac"] * total_steps))
                eps = cfg["eps_start"] + frac * (cfg["eps_end"] - cfg["eps_start"])
                a, _, _ = q.act(obs, greedy=False, eps=eps, rng=rng)
                nobs, r, done, _ = venv.step(a)
                rstd.update(r)
                ep_ret += r
                for i in range(N):
                    S[ptr], A[ptr], R[ptr], S2[ptr], D[ptr] = _compact(obs[i]), a[i], r[i], _compact(nobs[i]), float(done[i])
                    ptr = (ptr + 1) % cap
                    size = min(size + 1, cap)
                obs = nobs
                step += 1
                if size >= cfg["batch"] * 4:
                    for _ in range(cfg.get("updates_per_step", 2)):
                        idx = rng.integers(0, size, cfg["batch"])
                        bs, bs2 = collate([_expand(S[j]) for j in idx]), collate([_expand(S2[j]) for j in idx])
                        act = torch.as_tensor(A[idx])
                        rew = torch.as_tensor(R[idx] / rstd.std)
                        nd = 1.0 - torch.as_tensor(D[idx])
                        with torch.no_grad():                      # double DQN target, averaged over branches
                            qn_on, qn_tg = q(bs2), q_tgt(bs2)
                            nxt = torch.stack([tg.gather(1, on.argmax(1, keepdim=True)).squeeze(1)
                                               for on, tg in zip(qn_on, qn_tg)], 1).mean(1)
                            y = rew + cfg["gamma"] * nd * nxt
                        qs = q(bs)
                        pred = torch.stack([qd.gather(1, act[:, d:d + 1]).squeeze(1) for d, qd in enumerate(qs)], 1)
                        loss = F.smooth_l1_loss(pred, y[:, None].expand_as(pred))
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(q.parameters(), 10.0)
                        opt.step()
                        losses.append(loss.item())
                        n_grad += 1
                        if n_grad % cfg["target_every"] == 0:
                            q_tgt.load_state_dict(q.state_dict())
            vi = vf = float("nan")
            if (u + 1) % cfg["val_every"] == 0 or u == rounds - 1:
                vi, vf = validate(q, env_cfg, val_insts, val_ref)
                save_policy(q, os.path.join(out_dir, "last.pt"), env_cfg, {"update": u + 1, "episodes": (u + 1) * N})
                if vi > best:
                    best = vi
                    save_policy(q, os.path.join(out_dir, "best.pt"), env_cfg, {"val_impr": vi, "update": u + 1})
            w.writerow([u + 1, (u + 1) * N, ep_ret.mean(), np.mean(losses) if losses else float("nan"), 0.0, eps, vi, vf,
                        time.time() - t0])
            fh.flush()
            log(f"[{u + 1:4d}/{rounds}] return {ep_ret.mean():7.3f} | loss {np.mean(losses) if losses else 0:.4f} "
                f"eps {eps:.3f} | val vs RW {vi:+.3f} (best {best:+.3f}) | {time.time() - t0:.0f}s")
    finally:
        fh.close()
        venv.close()
    return os.path.join(out_dir, "last.pt")
