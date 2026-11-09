"""ddsp_v1 with mean vs sampled latent, + empirical cost (bits/s) of the explicit side info (f0, loudness) if transmitted as quantised first-differences."""
import math, numpy as np, torch
from ddsp_model import *
B = "/home/qbrighto/music-visualizer/v2"; dev = "cuda"; ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); val, test = ck["val"], ck["test"]
c = torch.load(f"{B}/ddsp_v1.pt", map_location=dev); a = c["args"]; m = DDSPAE(a["ZD"], a["GD"], use_global=bool(a["UG"])).to(dev).eval(); m.load_state_dict(c["model"]); torch.manual_seed(0)
res = {0.0: [], 1.0: []}
def H(q): v, n = np.unique(q, return_counts=True); p = n / n.sum(); return float(-(p * np.log2(p)).sum())
side = []
with torch.no_grad():
    for k in val + test:
        A = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); n = 20 * SR; s0 = min(len(A) // 3, 30 * SR); s0 -= s0 % 240
        x = torch.from_numpy(np.asarray(A[s0:s0 + n], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(np.load(f"{B}/f0/{k}.npy")[s0 // 240: s0 // 240 + n // 240])[None].to(dev)
        mu, lv = m.encode(x); zg = m.encode_global(x)[0]; eps = torch.randn_like(mu)
        for t in res: y, _ = m.decode(f0, loudness(x), mu + t * eps * (0.5 * lv).exp(), zg, n); res[t].append((k in test, mss_loss(y, x).item()))
        lf = (12 * torch.log2(f0[0].clamp(40, 1200) / 110)).cpu().numpy(); ld = (loudness(x)[0] * 30 - 60).cpu().numpy()          # semitones, dB
        side.append((H(np.diff(np.round(lf / 0.5)).astype(int)) * 100, H(np.diff(np.round(ld / 1.0)).astype(int)) * 100))
for t in res: print(f"ddsp_v1 tau={t:.0f}: val mss {np.mean([r[1] for r in res[t] if not r[0]]):.3f} | Maisky mss {[r[1] for r in res[t] if r[0]][0]:.3f}")
s = np.mean(side, 0); print(f"side-info cost (zeroth-order entropy of quantised first differences; f0 in 0.5-semitone steps, loudness in 1 dB steps): f0 ~{s[0]:.0f} b/s + loudness ~{s[1]:.0f} b/s = ~{s.sum():.0f} b/s")
