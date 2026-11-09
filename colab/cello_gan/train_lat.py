"""Continuous latent generator: non-autoregressive conditional flow matching over the codec's 24-channel 25 Hz latent.
Every frame sees the whole window of controls (pitch, voicing, periodicity, loudness, brightness, noisiness) in both directions, so there is no
token-by-token drift. A random prefix of the window can be given as known latent (continuation / sliding windows). Classifier-free guidance by
dropping the controls. In-training eval: generated latents -> frozen codec pitch head -> pitch error, and frame-to-frame movement vs real.
usage: python train_lat.py NAME STEPS CODEC.pt     env: TOK_DIR (latents/, controls/, controls2/), CELLO_DIR (split), OUT, BS (32) L (300)
D (384) LAYERS (8) HEADS (6) LR (3e-4) CDROP (0.1) PCTX (0.5 = chance of a known prefix) EVERY (500) NSTEP (32 Euler steps)"""
import os, sys, math, json, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys_argv = sys.argv; sys.argv = ["x", "lat", "1"]
import train_tok3 as T3                                                     # ctrl_feats, pbins, CodecPitch (frozen latent -> pitch head)
sys.argv = sys_argv
name, STEPS = sys.argv[1], int(sys.argv[2]); TOK = os.environ["TOK_DIR"]; OUT = os.environ.get("OUT", TOK); dev = "cuda"; os.makedirs(OUT, exist_ok=True)
BS, L, D, LAYERS, HEADS = int(os.environ.get("BS", 32)), int(os.environ.get("L", 300)), int(os.environ.get("D", 384)), int(os.environ.get("LAYERS", 8)), int(os.environ.get("HEADS", 6))
LR, CDROP, PCTX, EVERY, NSTEP = float(os.environ.get("LR", 3e-4)), float(os.environ.get("CDROP", 0.1)), float(os.environ.get("PCTX", 0.5)), int(os.environ.get("EVERY", 500)), int(os.environ.get("NSTEP", 32))
ZD, NC = 24, T3.NC


def load(keys):
    out = {}
    for k in keys:
        z = np.load(f"{TOK}/latents/{k}.npy").astype(np.float32); c = np.load(f"{TOK}/controls/{k}.npy"); c2 = np.load(f"{TOK}/controls2/{k}.npy"); n = min(len(z), len(c), len(c2))
        out[k] = (z[:n], T3.ctrl_feats(c[:n], c2[:n]), T3.pbins(c[:n]))
    return out


def batch(data, keys, ws, n):
    Z, C, P = [], [], []
    for _ in range(n):
        k = keys[np.random.choice(len(keys), p=ws)]; s = random.randrange(0, len(data[k][0]) - L); d = data[k]; Z.append(d[0][s:s + L]); C.append(d[1][s:s + L]); P.append(d[2][s:s + L])
    return torch.from_numpy(np.stack(Z)).to(dev), torch.from_numpy(np.stack(C)).to(dev), torch.from_numpy(np.stack(P)).to(dev)


def temb(t, dim):                                                           # sinusoidal embedding of the flow time t in [0,1]
    f = torch.exp(-math.log(10000) * torch.arange(dim // 2, device=t.device) / (dim // 2)); a = t[:, None] * 1000 * f[None]; return torch.cat([a.sin(), a.cos()], 1)


class Block(nn.Module):                                                     # pre-norm transformer block, adaLN (scale/shift/gate) from time + local controls
    def __init__(self, D=D, HEADS=HEADS):
        super().__init__(); self.n1 = nn.LayerNorm(D, elementwise_affine=False); self.n2 = nn.LayerNorm(D, elementwise_affine=False)
        self.att = nn.MultiheadAttention(D, HEADS, batch_first=True, dropout=0.1); self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
        self.ada = nn.Linear(D, 6 * D); nn.init.zeros_(self.ada.weight); nn.init.zeros_(self.ada.bias)

    def forward(self, h, c):                                                # c (B,L,D) = time + control conditioning per frame
        s1, b1, g1, s2, b2, g2 = self.ada(F.silu(c)).chunk(6, -1)
        x = self.n1(h) * (1 + s1) + b1; h = h + g1 * self.att(x, x, x, need_weights=False)[0]
        x = self.n2(h) * (1 + s2) + b2; return h + g2 * self.ff(x)


class LatFlow(nn.Module):                                                   # zd/nc/D/LAYERS/HEADS overridable so the curve model (train_curve.py) reuses it
    def __init__(self, ZD=ZD, NC=NC, D=D, LAYERS=LAYERS, HEADS=HEADS):
        super().__init__(); self.zd = ZD
        self.inp = nn.Conv1d(ZD + ZD + 1, D, 3, padding=1)                  # noisy latent, known-context latent, known mask
        self.cond = nn.Sequential(nn.Conv1d(NC + 1, D, 5, padding=2), nn.GELU(), nn.Conv1d(D, D, 5, padding=2))   # last channel = "controls dropped"
        self.tm = nn.Sequential(nn.Linear(256, D), nn.SiLU(), nn.Linear(D, D)); self.pos = nn.Parameter(torch.randn(1, 1024, D) * 0.02)
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(LAYERS)]); self.nout = nn.LayerNorm(D); self.out = nn.Conv1d(D, ZD, 3, padding=1); nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, xt, t, ctrl, known, kmask, drop):                     # xt/known (B,L,ZD), t (B,), ctrl (B,L,NC), kmask (B,L), drop (B,) -> velocity (B,L,ZD)
        B_, L_, _ = xt.shape
        c = self.cond(torch.cat([ctrl * (1 - drop)[:, None, None], drop[:, None, None].expand(B_, L_, 1)], 2).transpose(1, 2)).transpose(1, 2) + self.tm(temb(t, 256))[:, None]
        h = self.inp(torch.cat([xt, known * kmask[..., None], kmask[..., None]], 2).transpose(1, 2)).transpose(1, 2) + self.pos[:, :L_] + c
        for b in self.blocks: h = b(h, c)
        return self.out(self.nout(h).transpose(1, 2)).transpose(1, 2)


@torch.no_grad()
def sample(net, ctrl, known=None, kmask=None, gs=1.0, nstep=NSTEP, seed=0, solver=None):  # integrate the flow from noise (t=0) to data (t=1); known frames are clamped
    solver = solver or os.environ.get("SOLVER", "euler")                    # "heun" = second order (two velocity evaluations per step, Euler on the last step)
    g = torch.Generator(device=dev).manual_seed(seed); B_, L_ = ctrl.shape[:2]; x = torch.randn(B_, L_, net.zd, device=dev, generator=g)
    if known is None: known, kmask = torch.zeros_like(x), torch.zeros(B_, L_, device=dev)
    d0, d1 = torch.zeros(B_, device=dev), torch.ones(B_, device=dev)

    def vel(x_, t_):
        tt = torch.full((B_,), t_, device=dev); v_ = net(x_, tt, ctrl, known, kmask, d0)
        if gs != 1: vu = net(x_, tt, ctrl, known, kmask, d1); v_ = vu + gs * (v_ - vu)
        return v_

    tend = float(os.environ.get("TEND", 1.0))                              # < 1: integrate to tend, then jump straight to t=1 along the last velocity (avoids the barely-trained t~1 region)
    ts = [tend * i / nstep for i in range(nstep + 1)]
    for i in range(nstep):
        t0, t1 = ts[i], ts[i + 1]; v = vel(x, t0)
        if solver == "heun" and t1 < 1.0: x = x + (t1 - t0) / 2 * (v + vel(x + (t1 - t0) * v, t1))
        else: x = x + (t1 - t0) * v
    if tend < 1.0: x = x + (1.0 - tend) * vel(x, tend)
    return torch.where(kmask[..., None] > 0, known, x)


if __name__ == "__main__":
    sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu"); train = [k for k in sp["train"] if os.path.exists(f"{TOK}/latents/{k}.npy")]; val = sp["val"]
    TR, VA = load(train), load(val); allz = np.concatenate([TR[k][0] for k in train]); MU, SD = torch.tensor(allz.mean(0), device=dev), torch.tensor(allz.std(0) + 1e-4, device=dev)
    w = np.array([len(TR[k][0]) for k in train], float); w /= w.sum(); cp = T3.CodecPitch(sys.argv[3]).to(dev)
    net = LatFlow().to(dev); print(len(TR), "train recordings | params %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6), "| latent sd range %.2f-%.2f" % (SD.min(), SD.max()), flush=True)
    opt = torch.optim.AdamW(net.parameters(), LR, betas=(0.9, 0.99), weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.03)
    ema = [p.detach().clone() for p in net.parameters()]
    vkeys = list(VA); vw = np.array([len(VA[k][0]) for k in vkeys], float); vw /= vw.sum(); random.seed(0); np.random.seed(0); VB = [batch(VA, vkeys, vw, 16) for _ in range(4)]
    log = {"args": dict(BS=BS, L=L, D=D, LAYERS=LAYERS, HEADS=HEADS, LR=LR, CDROP=CDROP, PCTX=PCTX), "norm": {"mu": MU.tolist(), "sd": SD.tolist()}, "evals": []}

    def fm_loss(z, ctrl, train_mode=True):                                  # rectified flow: x_t = t*x1 + (1-t)*noise, target velocity x1 - noise; loss on unknown frames
        x1 = (z - MU) / SD; B_ = x1.shape[0]; e = torch.randn_like(x1); t = torch.sigmoid(torch.randn(B_, device=dev)); xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * e
        kl = torch.where(torch.rand(B_, device=dev) < PCTX, torch.randint(1, L // 2, (B_,), device=dev), torch.zeros(B_, dtype=torch.long, device=dev)) if train_mode else torch.zeros(B_, dtype=torch.long, device=dev)
        km = (torch.arange(L, device=dev)[None] < kl[:, None]).float(); drop = (torch.rand(B_, device=dev) < CDROP).float() if train_mode else torch.zeros(B_, device=dev)
        v = net(xt, t, ctrl, x1, km, drop); err = ((v - (x1 - e)) ** 2).mean(-1); m = 1 - km; return (err * m).sum() / m.sum()

    @torch.no_grad()
    def evaluate():
        bk = [p.detach().clone() for p in net.parameters()]
        for p, e in zip(net.parameters(), ema): p.copy_(e)
        net.eval(); vl = float(np.mean([fm_loss(z, c, False).item() for z, c, _ in VB])); z, c, pb = VB[0]; r = {}
        for gs in (1.0, 2.0):
            x = sample(net, c, gs=gs) * SD + MU; zr = z
            pl = cp.ph(cp.up(F.interpolate(x.transpose(1, 2), size=L * 4, mode="linear", align_corners=False)))[:, : T3.NB]; tt = pb.repeat_interleave(4, 1); vv = tt != -100
            r[f"pitch50_g{gs:g}"] = float(((pl.argmax(1) - tt).abs() <= 2)[vv].float().mean()); r[f"move_g{gs:g}"] = float((x[:, 1:] - x[:, :-1]).norm(dim=-1).mean())
            r[f"dist_g{gs:g}"] = float(((x - zr) / SD).pow(2).mean())          # distance to the real latent of the same passage (normalised units; 2.0 = unrelated)
        r["move_real"] = float((z[:, 1:] - z[:, :-1]).norm(dim=-1).mean())
        for p, b in zip(net.parameters(), bk): p.copy_(b)
        net.train(); return vl, r

    t0 = time.time(); net.train()
    for s in range(1, STEPS + 1):
        z, c, _ = batch(TR, train, w, BS)
        with torch.autocast("cuda", dtype=torch.bfloat16): loss = fm_loss(z, c)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        with torch.no_grad():
            for p, e in zip(net.parameters(), ema): e.mul_(0.999).add_(p.detach(), alpha=0.001)
        if s % 250 == 0: print(s, "flow loss %.4f" % loss.item(), round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            vl, r = evaluate(); log["evals"].append({"step": s, "val_loss": vl, **r})
            print("EVAL", s, "val flow loss %.4f" % vl, "| pitch within 50c g1 %.2f g2 %.2f" % (r["pitch50_g1"], r["pitch50_g2"]), "| movement/frame g1 %.2f g2 %.2f real %.2f" % (r["move_g1"], r["move_g2"], r["move_real"]),
                  "| dist to real g1 %.2f g2 %.2f" % (r["dist_g1"], r["dist_g2"]), flush=True)
            sd_ema = {n: e for (n, _), e in zip(net.named_parameters(), ema)}
            torch.save({"model": net.state_dict(), "ema": sd_ema, "args": log["args"], "norm": log["norm"], "step": s}, f"{OUT}/{name}_last.pt"); json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)
            if s % 2500 == 0: torch.save({"model": net.state_dict(), "ema": sd_ema, "args": log["args"], "norm": log["norm"], "step": s}, f"{OUT}/{name}_s{s}.pt")
    print("training done")
