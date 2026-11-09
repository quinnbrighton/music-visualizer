"""Conditional flow-matching transformer (= colab/cello_gan/train_lat.LatFlow): used twice, as the latent model (controls -> 24-ch codec latent)
and as the note-conditioned curve model (notes -> 6 performance curves). Non-autoregressive over a window; a known prefix is clamped (sliding windows);
classifier-free guidance by dropping the controls."""
import math
import torch, torch.nn as nn, torch.nn.functional as F


def temb(t, dim):                                                           # sinusoidal embedding of the flow time t in [0,1]
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2)); a = t[:, None] * 1000 * f[None]; return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):                                                     # pre-norm transformer block, adaLN (scale/shift/gate) from time + local controls
    def __init__(self, D, HEADS):
        super().__init__(); self.n1 = nn.LayerNorm(D, elementwise_affine=False); self.n2 = nn.LayerNorm(D, elementwise_affine=False)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True, dropout=0.1); self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.ada = nn.Linear(D, 6 * D)

    def forward(self, h, c):
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c)).chunk(6, -1)
        x = self.n1(h) * (1 + s1) + b1; h = h + g1 * self.att(x, x, x, need_weights=False)[0]
        x = self.n2(h) * (1 + s2) + b2; return h + g2 * self.ff(x)


class LatFlow(nn.Module):
    def __init__(self, ZD=24, NC=6, D=384, LAYERS=8, HEADS=6):
        super().__init__(); self.zd = ZD
        self.inp = nn.Conv1d(ZD + ZD + 1, D, 3, padding=1)                  # noisy latent, known-context latent, known mask
        self.cond = nn.Sequential(nn.Conv1d(NC + 1, D, 5, padding=2), nn.GELU(), nn.Conv1d(D, D, 5, padding=2))   # last channel = "controls dropped"
        self.tm = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D)); self.pos = nn.Parameter(torch.zeros(1, 1024, D))
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(LAYERS)]); self.nout = nn.LayerNorm(D); self.out = nn.Conv1d(D, ZD, 3, padding=1)

    def forward(self, xt, t, ctrl, known, kmask, drop):                     # xt/known (B,L,ZD), t (B,), ctrl (B,L,NC), kmask (B,L), drop (B,) -> velocity (B,L,ZD)
        B_, L_, _ = xt.shape
        c = self.cond(torch.cat([ctrl * (1 - drop)[:, None, None], drop[:, None, None].expand(B_, L_, 1)], 2).transpose(1, 2)).transpose(1, 2) + self.tm(temb(t, 256))[:, None]
        h = self.inp(torch.cat([xt, known * kmask[..., None], kmask[..., None]], 2).transpose(1, 2)).transpose(1, 2) + self.pos[:, :L_] + c
        for b in self.blocks: h = b(h, c)
        return self.out(self.nout(h).transpose(1, 2)).transpose(1, 2)


@torch.no_grad()
def sample(net, ctrl, known=None, kmask=None, gs=1.0, nstep=32, seed=0, solver="euler", tend=1.0):   # integrate noise (t=0) -> data (t=1); known frames clamped
    dev = ctrl.device; g = torch.Generator(device=dev).manual_seed(seed); B_, L_ = ctrl.shape[:2]; x = torch.randn(B_, L_, net.zd, device=dev, generator=g)
    if known is None: known, kmask = torch.zeros_like(x), torch.zeros(B_, L_, device=dev)
    d0, d1 = torch.zeros(B_, device=dev), torch.ones(B_, device=dev)

    def vel(x_, t_):
        tt = torch.full((B_,), t_, device=dev); v_ = net(x_, tt, ctrl, known, kmask, d0)
        if gs != 1: vu = net(x_, tt, ctrl, known, kmask, d1); v_ = vu + gs * (v_ - vu)
        return v_

    ts = [tend * i / nstep for i in range(nstep + 1)]                      # tend < 1: stop there, then jump to t=1 along the last velocity
    for i in range(nstep):
        t0, t1 = ts[i], ts[i + 1]; v = vel(x, t0)
        if solver == "heun" and t1 < 1.0: x = x + (t1 - t0) / 2 * (v + vel(x + (t1 - t0) * v, t1))
        else: x = x + (t1 - t0) * v
    if tend < 1.0: x = x + (1.0 - tend) * vel(x, tend)
    return torch.where(kmask[..., None] > 0, known, x)


def windows(net, ctrl, L, hold, gs, seed=0, **kw):                         # long sequences: overlapping windows, each keeping the last `hold` frames of the previous one as known context
    dev = ctrl.device; zd = net.zd; n_ = ctrl.shape[0]; out = torch.zeros(0, zd, device=dev); done = 0
    while done < n_:
        h = min(hold, done); m = min(L - h, n_ - done); kn = torch.zeros(1, h + m, zd, device=dev); km = torch.zeros(1, h + m, device=dev); kn[0, :h] = out[done - h: done]; km[0, :h] = 1
        w = sample(net, ctrl[None, done - h: done + m], known=kn, kmask=km, gs=gs, seed=seed + done, **kw)[0]; out = torch.cat([out, w[h:]]); done += m
    return out
