"""Channel ablation for the 24-D model: replace one latent channel by its per-clip time-mean and measure the val spectral loss / log-mel. usage: INIT=ckpt python ablate_ch.py"""
import sys, os
sys.argv = ["x", "ablate", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn3.py")).read(); exec(src[: src.index("log = {")])
import json
@torch.no_grad()
def run(drop):
    model.eval(); a_, l_ = [], []
    for k in val:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); T = n // HOP
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
        mu, _ = model.encode(feats(x, f), T)
        for c in drop: mu[:, c] = mu[:, c].mean(-1, keepdim=True)
        torch.manual_seed(0); y = model.decode(mu, f0, T); a_.append(MSS(y, x).item()); l_.append(logmel_dist(y[0].cpu(), x[0].cpu())["lm"])
    return float(np.mean(a_)), float(np.mean(l_))
base = run([]); print("base", base, flush=True); out = {"base": base}
for c in range(ZD): out[c] = run([c]); print("drop", c, "d_mss %+.4f d_lm %+.4f" % (out[c][0] - base[0], out[c][1] - base[1]), flush=True)
json.dump(out, open(f"{B}/ablate_ch.json", "w"))
