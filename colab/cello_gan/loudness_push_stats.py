"""Latent loudness push across many recordings: for each recording, decode a 12 s passage as is and pushed +20 along the latent's
loudness direction, remove the push's true level shift, and measure how the tone changes. Writes docs/assets/loudness_push_stats.png
and loudness_push_stats.json.   usage: python loudness_push_stats.py     env: as gen_lat.py (codec skeleton)"""
import os, sys, json
sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import scipy.signal as ss, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/music-visualizer"); R = f"{H}/colab_results"; LAT = f"{R}/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; OUTD = f"{H}/docs/assets"
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
c_ = torch.load(f"{R}/reg3/codec_snd_s8000.pt", map_location=dev, weights_only=False); model.load_state_dict(c_["model"]); model.eval()
spl = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); PUSH, SEC = 20.0, 12

# loudness direction: least-squares slope of the pitch-removed latent on loudness (dB), from the training recordings
rng = np.random.default_rng(0); Z, S, L = [], [], []
for t in spl["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); c = np.load(f"{CTL}/{t}.npy"); n = min(len(z), len(c)); pick = rng.choice(n, min(800, n), replace=False)
    Z.append(z[pick]); S.append(np.where(c[pick, 1] > 0.5, np.round(69 + 12 * np.log2(np.clip(c[pick, 0], 40, 1200) / 440)), -1)); L.append(c[pick, 2])
Z, S, L = np.concatenate(Z), np.concatenate(S).astype(int), np.concatenate(L)
r = Z - Z.mean(0)
for k in np.unique(S): r[S == k] -= r[S == k].mean(0)
Lc = L - L.mean(); dirL = (r * Lc[:, None]).sum(0) / (Lc ** 2).sum()


@torch.no_grad()
def dec_ola(zz, W=100, Hh=75):                                              # 4 s windows with crossfades
    zz = torch.from_numpy(zz.astype(np.float32)).to(dev); nn_ = zz.shape[0]; Sm = HOP * DSR; out = np.zeros(nn_ * Sm); ws = np.zeros(nn_ * Sm); st = list(range(0, max(1, nn_ - W) + 1, Hh))
    if st[-1] + W < nn_: st.append(nn_ - W)
    for a in st:
        e = min(nn_, a + W); TT = (e - a) * DSR; torch.manual_seed(0); hh = dec_feats(zz[a:e].t()[None], TT); y = model.synth(hh, model.pitch_f0(hh), TT)[0].cpu().numpy()
        w = np.ones(len(y)); rr = (W - Hh) * Sm
        if a > 0: w[:rr] = np.linspace(0, 1, rr)
        if e < nn_: w[-rr:] = np.linspace(1, 0, rr)
        out[a * Sm: a * Sm + len(y)] += y * w; ws[a * Sm: a * Sm + len(y)] += w
    return out / np.maximum(ws, 1e-6)


rms = lambda v: np.sqrt(np.mean(v ** 2)); EDGES = 62.5 * 2 ** np.arange(0, 7.6, 1 / 3); CTR = np.sqrt(EDGES[:-1] * EDGES[1:])
def bands(y):
    f_, P_ = ss.welch(y, SR, nperseg=4096); return np.array([10 * np.log10(P_[(f_ >= lo) & (f_ < hi)].sum() + 1e-14) for lo, hi in zip(EDGES[:-1], EDGES[1:])])
def metrics(y, f0):
    f, t, Z_ = ss.stft(y, SR, nperseg=2048, noverlap=1536); P = np.abs(Z_) ** 2; out = {}
    out["brightness"] = np.median((f[:, None] * P).sum(0) / P.sum(0))
    out["upper"] = 10 * np.log10(P[(f > 1000) & (f < 4000)].sum() / P[f < 1000].sum())
    pk, vl = [], []
    for j, tt in enumerate(t):
        fr = f0[min(int(tt * 25), len(f0) - 1)]; fi = fr[0] if fr[1] > 0.5 else 0
        if fi < 60: continue
        h = np.arange(int(1000 / fi) + 1, int(5000 / fi)) * fi
        if len(h) < 2: continue
        Ld = 10 * np.log10(P[:, j] + 1e-12); pk.append(np.interp(h, f, Ld).mean()); vl.append(np.interp(h + fi / 2, f, Ld).mean())
    out["clarity"] = np.mean(np.array(pk) - np.array(vl))
    b, a = ss.butter(4, 2000, "high", fs=SR); e = 20 * np.log10(np.sqrt(np.convolve(ss.filtfilt(b, a, y) ** 2, np.ones(240) / 240, "same"))[::240] + 1e-9)
    out["attack"] = np.percentile(np.diff(e), 99); out["crest"] = 20 * np.log10(np.abs(y).max() / rms(y)); return out


rows, bandd = [], []
recs = [k for k in list(spl["test"]) + list(spl["val"]) + list(spl["train"]) if os.path.exists(f"{LAT}/{k}.npy") and k in A]
for k in recs:
    z = np.load(f"{LAT}/{k}.npy").astype(np.float32); c = np.load(f"{CTL}/{k}.npy"); n = min(len(z), len(c)); win = SEC * 25
    if n < win + 50: continue
    starts = np.arange(25, n - win, 250); best = starts[np.argmax([np.mean(c[s_: s_ + win, 1] > 0.5) for s_ in starts])]   # the most-pitched 12 s
    zz = z[best: best + win]; y0 = dec_ola(zz); y1 = dec_ola(zz + PUSH * dirL)
    shift = 20 * np.log10(rms(y1) / rms(y0)); y1 = y1 / rms(y1) * rms(y0)   # remove the push's true level shift
    m0, m1 = metrics(y0, c[best: best + win]), metrics(y1, c[best: best + win])
    rows.append(dict(rec=k, split="test" if k in spl["test"] else "val" if k in spl["val"] else "train", level_shift_db=shift,
                     brightness_pct=100 * (m1["brightness"] / m0["brightness"] - 1), upper_db=m1["upper"] - m0["upper"], clarity_db=m1["clarity"] - m0["clarity"],
                     attack=m1["attack"] - m0["attack"], crest_db=m1["crest"] - m0["crest"])); bandd.append(bands(y1) - bands(y0))
    print(len(rows), k[:12], {a: round(b, 2) for a, b in rows[-1].items() if isinstance(b, float)}, flush=True)
json.dump(rows, open(f"{OUTD}/loudness_push_stats.json", "w"), indent=1); bandd = np.array(bandd); np.save(f"{OUTD}/loudness_push_bands.npy", bandd)

KEYS = [("level_shift_db", "true level shift\n(dB)", None), ("brightness_pct", "brightness\ncentroid change (%)", +1), ("upper_db", "upper harmonics\nvs lows (dB)", +1),
        ("clarity_db", "harmonic clarity (dB)\nlower = scratchier", -1), ("crest_db", "crest (dB)\nlower = more sustained", -1), ("attack", "attack speed\n(dB per 10 ms)", None)]
fig = plt.figure(figsize=(14, 9)); gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.7, wspace=0.9)
ax = fig.add_subplot(gs[0, :]); med, q1, q3 = np.median(bandd, 0), np.percentile(bandd, 25, 0), np.percentile(bandd, 75, 0)
ax.fill_between(CTR, q1, q3, color=BLUE, alpha=0.2, lw=0, label="middle half of recordings"); ax.plot(CTR, med, color=BLUE, lw=2.4, marker="o", ms=4, label="median across recordings")
ax.axhline(0, color=GRID, lw=1); ax.set_xscale("log"); ax.set_xticks([125, 250, 500, 1000, 2000, 4000, 8000]); ax.set_xticklabels(["125", "250", "500", "1k", "2k", "4k", "8k"])
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("push minus reconstruction (dB)"); ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Spectrum change from the latent loudness push, after removing its level shift (%d recordings)" % len(rows), loc="left")
for i, (key, lab, sign) in enumerate(KEYS):
    a2 = fig.add_subplot(gs[1, i]); v = np.array([r_[key] for r_ in rows]); jit = rng.uniform(-0.18, 0.18, len(v))
    a2.scatter(jit, v, s=14, color=BLUE, alpha=0.6, lw=0); a2.plot([-0.3, 0.3], [np.median(v)] * 2, color=INK, lw=2); a2.axhline(0, color=GRID, lw=1); a2.set_xticks([]); a2.set_xlim(-0.5, 0.5)
    frac = "" if sign is None else "\n%d of %d in that direction" % (np.sum(np.sign(v) == sign), len(v))
    a2.set_title("%s\nmedian %+.1f%s" % (lab, np.median(v), frac), loc="left", fontsize=8.5)
fig.text(0.125, 0.03, "Each dot is one recording (12 s, its most-pitched passage), pushed +20 along the loudness direction and level-matched to its own reconstruction. Black line: median.", color=INK2, fontsize=8.5)
fig.savefig(f"{OUTD}/loudness_push_stats.png", dpi=100, bbox_inches="tight"); print("wrote loudness_push_stats.png")
for key, lab, sign in KEYS:
    v = np.array([r_[key] for r_ in rows]); print("%-16s median %+.2f  IQR [%+.2f, %+.2f]" % (key, np.median(v), np.percentile(v, 25), np.percentile(v, 75)) + ("" if sign is None else "  %d/%d in expected direction" % (np.sum(np.sign(v) == sign), len(v))))
