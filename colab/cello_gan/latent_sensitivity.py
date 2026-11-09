"""Is the codec latent perceptually uniform? Nudge each of the 24 channels by DELTA of its standard deviation on real passages, decode, and measure
the change in sound (multi-scale spectral distance MSS, log-mel distance) and in pitch (cents), vs random directions of the same size.
usage: python latent_sensitivity.py CODEC.pt LAT.pt OUT.json     env: as gen_lat.py; DELTA (0.5)"""
import os, sys
_codec, _lm, _out = sys.argv[1:4]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval()
import json
SD = torch.tensor(torch.load(_lm, map_location="cpu")["norm"]["sd"], device=dev); TD = os.environ["TOK_DIR"]; DS_ = HOP * DSR; L = 300; DELTA = float(os.environ.get("DELTA", 0.5))
wins = [(val[0], 30 * 25), (val[2], 60 * 25), (test[0], 5572800 // DS_)]
Z = [torch.from_numpy(np.load(f"{TD}/latents/{k}.npy").astype(np.float32))[f: f + L].to(dev) for k, f in wins]
V = [torch.from_numpy(np.repeat(np.load(f"{TD}/controls/{k}.npy")[f: f + L, 1] > 0.5, DSR)).to(dev) for k, f in wins]


@torch.no_grad()
def dec(z):
    T = z.shape[0] * DSR; hh = dec_feats(z.t()[None], T); fp = model.pitch_f0(hh); return model.synth(hh, fp, T), fp[0]


base = [dec(z) for z in Z]


@torch.no_grad()
def effect(d):                                                              # d (24,) raw-unit offset added to every frame -> mean over windows
    m, lm, ce = [], [], []
    for z, v, (y0, f0) in zip(Z, V, base):
        y, fp = dec(z + d); m.append(MSS(y, y0).item()); lm.append(logmel_dist(y[0].cpu(), y0[0].cpu())["lm"])
        ce.append(float((1200 * torch.log2(fp.clamp(40, 1200) / f0.clamp(40, 1200))).abs()[v[: len(fp)]].median()))
    return dict(mss=round(float(np.mean(m)), 4), lm=round(float(np.mean(lm)), 4), cents=round(float(np.mean(ce)), 1))


rows = []
for i in range(24):
    e = torch.zeros(24, device=dev); e[i] = DELTA * SD[i]; r = {"ch": i, **effect(e)}; e[i] = -DELTA * SD[i]; r2 = effect(e)
    rows.append({"ch": i, "sd": round(float(SD[i]), 2), **{k: round((r[k] + r2[k]) / 2, 4) for k in ("mss", "lm", "cents")}}); print(rows[-1], flush=True)
g = torch.Generator(device=dev).manual_seed(0); rnd = []
for j in range(8):
    u = torch.randn(24, device=dev, generator=g); u = u / u.norm() * DELTA; rnd.append(effect(u * SD))
allz = torch.cat(Z); ev = torch.linalg.eigvalsh(torch.cov(((allz - allz.mean(0)) / SD).t())).flip(0)
out = dict(delta=DELTA, channels=rows, random_dirs={k: round(float(np.mean([r[k] for r in rnd])), 4) for k in ("mss", "lm", "cents")},
           pca_var_share=[round(float(x), 3) for x in (ev / ev.sum()).tolist()])
json.dump(out, open(_out, "w"), indent=1)
ms = np.array([r["mss"] for r in rows]); print("MSS change per channel: max/min %.1fx, top-6 share of total %.2f" % (ms.max() / ms.min(), np.sort(ms)[::-1][:6].sum() / ms.sum()))
print("random dirs", out["random_dirs"], "| PCA variance share top 6:", out["pca_var_share"][:6])
