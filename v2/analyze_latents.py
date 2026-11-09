"""Encode a whole recording with the 24/16-D model, then (1) per-channel frequency content of the latent trajectory (Welch PSD), (2) PCA of the trajectory. env: INIT INZD ZDIM KEY OUT"""
import sys, os
sys.argv = ["x", "an", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn6.py")).read(); exec(src[: src.index("log = {")])
import json; from scipy.signal import welch
key, out = os.environ["KEY"], os.environ["OUT"]; model.eval(); a = A[key]; N = len(a) // 960 * 960; CH, CTX = 20 * SR, SR; mus, lvs = [], []
with torch.no_grad():
    for c in range(0, N, CH):
        s, e = max(0, c - CTX), min(N, c + CH + CTX); e -= (e - s) % 960
        x = torch.from_numpy(np.asarray(a[s:e], dtype=np.float32))[None].to(dev); mu, lv = model.encode(feats(x, None), (e - s) // HOP)
        lo = (c - s) // HOP; hi = lo + min(CH, N - c) // HOP; mus.append(mu[0, :, lo:hi].cpu()); lvs.append(lv[0, :, lo:hi].cpu())
mu = torch.cat(mus, 1); lv = torch.cat(lvs, 1); print("frames", mu.shape, "seconds", mu.shape[1] / 100)
kd = (kl_vec(mu[None].to(dev), lv[None].to(dev), 0.0).cpu().numpy() * 100 / math.log(2)); print("bits/s by channel", kd.round(1).tolist(), "total", round(float(kd.sum()), 1))
Z = mu.numpy().T; res = {"bits": kd.round(2).tolist(), "groups": [n for n, w in RATES for _ in range(n)], "widths": [w for n, w in RATES for _ in range(n)]}
psd = []; f50, f90, f99 = [], [], []
for c in range(ZD):
    f, p = welch(Z[:, c] - Z[:, c].mean(), fs=100, nperseg=2048); psd.append(p); cp = np.cumsum(p) / p.sum()
    f50.append(float(f[np.searchsorted(cp, 0.5)])); f90.append(float(f[np.searchsorted(cp, 0.9)])); f99.append(float(f[np.searchsorted(cp, 0.99)]))
res.update(f=f.tolist(), f50=f50, f90=f90, f99=f99, psd=[p.tolist() for p in psd])
def pca(X):
    X = X - X.mean(0); u, s, vt = np.linalg.svd(X, full_matrices=False); ev = s ** 2 / (s ** 2).sum(); return X @ vt.T, ev, vt
Zs = (Z - Z.mean(0)) / Z.std(0); pr, evr, vr = pca(Z); ps, evs, vs = pca(Zs)
res.update(ev_raw=evr.tolist(), ev_std=evs.tolist(), load_std=vs[:4].tolist(), pcs=ps[::10, :3].tolist(), std_per_channel=Z.std(0).tolist())
for nm, ev in (("raw", evr), ("standardised", evs)): c = np.cumsum(ev); print(nm, "PCA explained", ev[:6].round(3).tolist(), "| comps for 90/95/99%:", [int(np.searchsorted(c, q) + 1) for q in (.9, .95, .99)])
print("channel f50/f90/f99 Hz:"); [print(c, "grp-width", res["widths"][c], "std %.2f" % Z[:, c].std(), "bits %.1f" % kd[c], "f50 %.2f f90 %.2f f99 %.2f" % (f50[c], f90[c], f99[c])) for c in range(ZD)]
json.dump(res, open(out + ".json", "w")); np.save(out + "_mu.npy", Z.astype(np.float32))
