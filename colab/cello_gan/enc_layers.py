"""Layer-fusion encoder for a frozen multi-layer backbone (MERT hidden states). No bottleneck: output stays wide (d channels @100 Hz).

Each cached layer gets its own LayerNorm + linear adapter (layers have very different scales and carry different content),
a per-frame softmax gate chooses layers over time, a concat path keeps cross-layer interactions, then dilated temporal convs.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from ddsp_model import block


class LocalAttn(nn.Module):
    """Pre-norm self-attention over time limited to a window of past (and optionally future) frames, residual."""
    def __init__(self, d, heads=4, past=300, future=0):
        super().__init__()
        self.n, self.att, self.past, self.future = nn.GroupNorm(8, d), nn.MultiheadAttention(d, heads, batch_first=True), past, future

    def forward(self, x):                                                       # (B,d,T)
        T = x.shape[2]; i = torch.arange(T, device=x.device)
        dt = i[None, :] - i[:, None]                                            # key - query offset
        mask = (dt < -self.past) | (dt > self.future)                           # True = blocked
        h = self.n(x).transpose(1, 2)
        return x + self.att(h, h, h, attn_mask=mask, need_weights=False)[0].transpose(1, 2)


class LayerFusionEncoder(nn.Module):
    def __init__(self, n_layers=4, din=768, d=256, attn=0, past=300, future=0):
        super().__init__()
        self.attn = nn.ModuleList([LocalAttn(d, 4, past, future) for _ in range(attn)])      # attn=0 keeps the conv-only encoder
        self.L, self.d = n_layers, d
        self.norm = nn.ModuleList([nn.BatchNorm1d(din) for _ in range(n_layers)])     # per-channel stats over batch+time: keeps per-frame level/gain (a per-frame LayerNorm removed it)
        self.adapt = nn.ModuleList([nn.Linear(din, d) for _ in range(n_layers)])
        self.gate = nn.Linear(n_layers * d, n_layers)                           # per-frame layer weights
        self.cat = nn.Linear(n_layers * d, d)                                   # cross-layer interactions
        self.out_norm = nn.GroupNorm(8, d)
        self.trunk = nn.Sequential(block(d, d, 5, 1), block(d, d, 5, 2), block(d, d, 5, 4))

    def forward(self, f75, T):                                                  # f75 (B,T75,L,din) -> (B,d,T) at 100 Hz
        h = torch.stack([self.adapt[i](self.norm[i](f75[:, :, i].float().transpose(1, 2)).transpose(1, 2)) for i in range(self.L)], 2)     # (B,T75,L,d)
        g = torch.softmax(self.gate(h.flatten(2)), -1)                          # (B,T75,L)
        x = F.gelu((g[..., None] * h).sum(2) + self.cat(h.flatten(2)))
        x = F.interpolate(x.transpose(1, 2), size=T, mode="linear", align_corners=False)
        x = self.trunk(self.out_norm(x))
        for a in self.attn: x = a(x)
        return x, g


if __name__ == "__main__":
    import numpy as np, glob
    p = sorted(glob.glob("mertfeat/*.npy"))[0]; f = torch.from_numpy(np.load(p, mmap_mode="r")[:750].copy()).float()[None]
    enc = LayerFusionEncoder(f.shape[2], attn=2, past=300, future=0); y, g = enc(f, 1000)
    x2 = f.clone(); x2[:, 700:] += 5; y2, _ = enc(x2, 1000)                      # causal check: changing late frames must not change early outputs
    print("causal leak:", float((y[..., :900] - y2[..., :900]).abs().max()))
    print(p, tuple(f.shape), "->", tuple(y.shape), "gate", tuple(g.shape), "params %.2fM" % (sum(q.numel() for q in enc.parameters()) / 1e6))
