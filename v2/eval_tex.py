"""Texture distance of a checkpoint on the val clips (20 s, pitch from teacher unless PITCHIN=1). env: INIT, INZD, ZDIM, PITCHIN"""
import sys, os
sys.argv = ["x", "evaltex", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn6.py")).read(); exec(src[: src.index("log = {")])
if "nz.bias" not in ck0: nn.init.constant_(model.nz.bias, -20.0)                      # checkpoints from before the noise path: no noise
PIN = int(os.environ.get("PITCHIN", 0)); model.eval(); acc = {}
with torch.no_grad():
    for k in val:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); T = n // HOP
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
        mu, lv = model.encode(feats(x, f), T)
        for tag, z in (("samp", mu + torch.randn_like(mu) * (0.5 * lv).exp()), ("mean", mu)):
            torch.manual_seed(0); y = model.decode(z, None if PIN else f0, T); tp = texparts(y, x, 2000)
            for nm, v in zip(("tex_sd", "tex_corr", "tex_rough"), tp): acc.setdefault(f"{tag}_{nm}", []).append(v)
            acc.setdefault(f"{tag}_tex", []).append(float(np.mean(tp))); acc.setdefault(f"{tag}_mss", []).append(MSS(y, x).item())
print("TEX", os.environ.get("INIT"), {m: round(float(np.mean(v)), 4) for m, v in acc.items() if m.startswith("samp") or m.endswith("_tex")})
