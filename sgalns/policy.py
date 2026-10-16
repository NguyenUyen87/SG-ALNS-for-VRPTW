"""Graph policy in plain PyTorch: 3 x GINE message passing (index_add, no PyG), mean+max pooling, global MLP,
three categorical heads (a1: 19, a2: 5, a3: 3) and a value head."""
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from .env import NODE_DIM, EDGE_DIM, G_DIM, CTX_DIM, N_A1, N_A2, N_A3, FIXED_A2, FIXED_A3, RW_A1

NEG = -1e9


def collate(obs_list, device="cpu"):
    """Batch graphs into one disjoint union graph."""
    xs, eis, eas, bidx = [], [], [], []
    off = 0
    for b, o in enumerate(obs_list):
        n = len(o["x"])
        xs.append(o["x"])
        eis.append(o["ei"] + off)
        eas.append(o["ea"])
        bidx.append(np.full(n, b, np.int64))
        off += n
    cat = lambda a, dt: torch.as_tensor(np.concatenate(a, 0), dtype=dt, device=device)
    return {"x": cat(xs, torch.float32), "ei": torch.as_tensor(np.concatenate(eis, 1), device=device),
            "ea": cat(eas, torch.float32), "b": cat(bidx, torch.long), "B": len(obs_list),
            "g": torch.as_tensor(np.stack([o["g"] for o in obs_list]), dtype=torch.float32, device=device)}


class GINE(nn.Module):
    """GINE layer (Hu et al. 2020) with a residual connection and LayerNorm."""

    def __init__(self, h, ed):
        super().__init__()
        self.edge = nn.Linear(ed, h)
        self.mlp = nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))
        self.eps = nn.Parameter(torch.zeros(1))
        self.norm = nn.LayerNorm(h)

    def forward(self, h, ei, ea):
        msg = torch.relu(h[ei[0]] + self.edge(ea))
        agg = torch.zeros_like(h).index_add(0, ei[1], msg)
        return h + torch.relu(self.norm(self.mlp((1.0 + self.eps) * h + agg)))


class Policy(nn.Module):
    """encoder='gnn': GINE layers; encoder='mlp': node-wise residual MLP (no message passing).
    expert/prior: keep the expert actions (RW operators, default q range, scheduled tau) and bias the initial
    policy towards them. use_a*: False fixes that component to its expert value."""

    def __init__(self, h=64, layers=3, encoder="gnn", use_a1=True, use_a2=True, use_a3=True,
                 expert=True, ctx=False, prior=True):
        super().__init__()
        self.cfg = dict(h=h, layers=layers, encoder=encoder, use_a1=use_a1, use_a2=use_a2, use_a3=use_a3,
                        expert=expert, ctx=ctx, prior=prior)
        self.gnn = encoder == "gnn"
        self.inp = nn.Linear(NODE_DIM, h)
        self.convs = nn.ModuleList([GINE(h, EDGE_DIM) if self.gnn else nn.Sequential(nn.Linear(h, h), nn.ReLU())
                                    for _ in range(layers)])
        self.gmlp = nn.Sequential(nn.Linear(G_DIM + (CTX_DIM if ctx else 0), 64), nn.ReLU(),
                                  nn.Linear(64, 64), nn.ReLU())
        z = 2 * h + 64
        self.trunk = nn.Sequential(nn.Linear(z, 64), nn.ReLU())
        self.pi1, self.pi2, self.pi3 = nn.Linear(64, N_A1), nn.Linear(64, N_A2), nn.Linear(64, N_A3)
        self.v = nn.Sequential(nn.Linear(z, 64), nn.ReLU(), nn.Linear(64, 1))
        with torch.no_grad():
            for head in (self.pi1, self.pi2, self.pi3):
                head.weight.mul_(0.1)
                head.bias.zero_()
            if expert and prior:                      # start close to ALNS-RW
                self.pi1.bias[RW_A1] = 3.0
                self.pi2.bias[FIXED_A2] = 2.0
                self.pi3.bias[FIXED_A3] = 2.0

    def embed(self, bt):
        h = torch.relu(self.inp(bt["x"]))
        for conv in self.convs:
            h = conv(h, bt["ei"], bt["ea"]) if self.gnn else h + conv(h)
        B, H = bt["B"], h.shape[1]
        cnt = torch.zeros(B).index_add(0, bt["b"], torch.ones(len(h)))
        mean = torch.zeros(B, H).index_add(0, bt["b"], h) / cnt[:, None]
        mx = torch.full((B, H), NEG).scatter_reduce(0, bt["b"][:, None].expand(-1, H), h,
                                                     reduce="amax", include_self=False)
        return torch.cat([mean, mx, self.gmlp(bt["g"])], -1)

    @staticmethod
    def _only(l, a):
        m = torch.full_like(l, NEG)
        m[:, a] = 0.0
        return m

    def forward(self, bt):
        z = self.embed(bt)
        t = self.trunk(z)
        l1, l2, l3 = self.pi1(t), self.pi2(t), self.pi3(t)
        c = self.cfg
        if not c["expert"]:                           # action space without the expert actions
            l1 = l1.clone()
            l1[:, RW_A1] = NEG
            if c["use_a2"]:
                l2 = l2.clone()
                l2[:, FIXED_A2] = NEG
        if not c["use_a1"]:
            l1 = self._only(l1, RW_A1)
        if not c["use_a2"]:
            l2 = self._only(l2, FIXED_A2)
        if not c["use_a3"]:
            l3 = self._only(l3, FIXED_A3)
        return (l1, l2, l3), self.v(z).squeeze(-1)

    def evaluate(self, bt, act):
        logits, v = self(bt)
        ds = [Categorical(logits=l) for l in logits]
        return sum(d.log_prob(act[:, i]) for i, d in enumerate(ds)), sum(d.entropy() for d in ds), v

    @torch.no_grad()
    def act(self, obs_list, greedy=False):
        logits, v = self(collate(obs_list))
        ds = [Categorical(logits=l) for l in logits]
        a = torch.stack([d.logits.argmax(-1) if greedy else d.sample() for d in ds], -1)
        logp = sum(d.log_prob(a[:, i]) for i, d in enumerate(ds))
        return a.numpy(), logp.numpy(), v.numpy()


def save_policy(policy, path, env_cfg, extra=None):
    torch.save({"state": policy.state_dict(), "policy_cfg": policy.cfg, "env_cfg": env_cfg, "extra": extra or {}}, path)


def load_policy(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck["policy_cfg"].get("kind") == "bdq":
        from .dqn import QPolicy
        p = QPolicy(**ck["policy_cfg"])
    else:
        p = Policy(**ck["policy_cfg"])
    p.load_state_dict(ck["state"])
    p.eval()
    return p, ck["env_cfg"]
