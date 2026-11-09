"""Three README figures: rate-distortion of the quantized codec, the codec's journey on one fixed held-out window, and what each
latent channel does. Writes docs/assets/{rate_distortion,journey,channel_knobs}.png.
usage: python make_insight_figs.py     env: as gen_lat.py (codec skeleton)"""
import os, sys, json
sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import scipy.signal as ss, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/music-visualizer"); R = f"{H}/colab_results"; OUTD = f"{H}/docs/assets"
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
k = test[0]

# 1. rate-distortion: quality vs bit rate as residual quantizer stages are added (held-out validation mean, final RVQ eval)
ev = json.load(open(f"{R}/harm_rvq64.json"))["evals"][-1]; pts = sorted((v["bits_per_s"], v["mss"], v["cents"]) for v in ev.values() if isinstance(v, dict) and "mss" in v)
b_, m_, c_ = map(np.array, zip(*pts))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2))
for ax, yv, lab, fmt in ((a1, m_, "spectral error (lower is better)", "%.3f"), (a2, c_, "median pitch error, cents (log scale)", "%.0f")):
    ax.plot(b_, yv, color=BLUE, lw=2, marker="o", ms=7, mec=BG, mew=2); ax.set_xlabel("bits per second"); ax.set_ylabel(lab); ax.margins(y=0.12); ax.grid(axis="y", color=GRID, lw=0.8)
    for x_, y_ in zip(b_, yv): ax.annotate(fmt % y_, (x_, y_), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, color=INK2)
    ax.axvline(1800, color=ORANGE, lw=1.2, ls="--"); ax.text(1790, ax.get_ylim()[1], "chosen: 1.8 kbit/s ", ha="right", va="top", color=ORANGE, fontsize=9)
a2.set_yscale("log"); a2.axhspan(1, 10, color=GRID, alpha=0.5, zorder=0); a2.text(160, 6.0, "roughly what trained ears notice (5–10 cents)", color=INK2, fontsize=8.5)
a1.set_title("Quality rises fast, then flattens", loc="left"); a2.set_title("Pitch: 177 cents at 1 stage, 9 at 12", loc="left")
fig.suptitle("Each quantizer stage adds 150 bit/s (12 stages × 64 entries)", x=0.01, ha="left", fontsize=10, color=INK2)
fig.tight_layout(); fig.savefig(f"{OUTD}/rate_distortion.png", dpi=110); plt.close(fig); print("wrote rate_distortion.png")

# 2. journey: log-mel error on the fixed 15 s held-out window (232.2 s) used throughout the lab notebook
S0, N = 5572800, 15 * SR; T = N // HOP; x = torch.from_numpy(np.asarray(A[k][S0: S0 + N], dtype=np.float32))[None].to(dev)
@torch.no_grad()
def lm_of(sd, nq=None, rvq_sd=None):
    model.load_state_dict(sd, strict=False); model.eval(); z = to25(feats(x, None), T)
    if nq: rvq.load_state_dict(rvq_sd); z, _, _ = quant(z, nq)
    hh = dec_feats(z, T); y = model.synth(hh, model.pitch_f0(hh), T); return logmel_dist(y[0].cpu(), x[0].cpu())["lm"]
pre = torch.load(f"{H}/v2/harm_final_z24lo_s2000.pt", map_location=dev, weights_only=False); pre = pre.get("model", pre) if isinstance(pre, dict) else pre
rq = torch.load(f"{R}/harm_rvq64_final.pt", map_location=dev, weights_only=False); snd = torch.load(f"{R}/reg3/codec_snd_s8000.pt", map_location=dev, weights_only=False)
steps = [("Spectrogram VAE\n+ Griffin-Lim\n(v1)", 0.871, "lab"), ("DDSP synth\n+ explicit pitch,\nloudness", 0.443, "lab"),
         ("Harmonic decoder\n+ noise path\n+ texture loss", lm_of(pre), "new"), ("+ quantizer\n(1.8 kbit/s,\nadversarial)", lm_of(rq["model"], 12, rq["rvq"]), "new"),
         ("+ sound-space\nregularization\n(final, continuous)", lm_of(snd["model"]), "new")]
for s_ in steps: print("journey", s_[0].replace("\n", " "), round(s_[1], 3))
fig, ax = plt.subplots(figsize=(12, 4.6)); xs = np.arange(len(steps))
ax.bar(xs, [s_[1] for s_ in steps], width=0.6, color=[INK2, INK2, BLUE, BLUE, BLUE], alpha=0.9)
for i, s_ in enumerate(steps): ax.text(i, s_[1] + 0.015, "%.3f" % s_[1], ha="center", color=INK, fontsize=10)
ax.axhline(0.146, color=ORANGE, lw=1.2, ls="--"); ax.text(4.4, 0.62, "dashed line: uncompressed ceiling\n(true spectrogram + Griffin-Lim): 0.146", ha="right", color=ORANGE, fontsize=9)
ax.set_xticks(xs); ax.set_xticklabels([s_[0] for s_ in steps], fontsize=9); ax.set_ylabel("log-mel error (lower is better)"); ax.set_ylim(0, 1.0)
ax.set_title("The codec's journey, measured on one fixed 15 s window of the held-out recording", loc="left")
ax.text(0, -0.30, "Grey: from the lab notebook. Blue: recomputed for this figure.\nLog-mel error rewards matching the average spectrum, so it can't see what the adversarial stage adds (see the GAN figure).",
        transform=ax.get_xaxis_transform(), color=INK2, fontsize=8.5)
fig.tight_layout(); fig.savefig(f"{OUTD}/journey.png", dpi=110, bbox_inches="tight"); plt.close(fig); print("wrote journey.png")

# 3. channel knobs: push each latent channel by the same amount on a sustained passage, decode, measure what changes
model.load_state_dict(snd["model"]); model.eval()
Z = np.load(f"{R}/reg3/tok/latents/{k}.npy").astype(np.float32); f0_ = S0 // 960 + 75; n = 50              # 2 s, 3 s into the README clip
z0 = torch.from_numpy(Z[f0_: f0_ + n]).t()[None].to(dev); step = 1.5 * float(Z.std())                      # one global step: the latent is in sound-space units
@torch.no_grad()
def dec(z): torch.manual_seed(0); hh = dec_feats(z, n * DSR); fp = model.pitch_f0(hh); return model.synth(hh, fp, n * DSR)[0].cpu().numpy(), fp[0].cpu().numpy()
def feats_of(y, fp):
    f, t, S = ss.stft(y, SR, nperseg=1024, noverlap=768); P = np.abs(S) ** 2 + 1e-12; cen = (f[:, None] * P).sum(0) / P.sum(0)
    return dict(pitch=1200 * np.log2(np.median(fp[fp > 40])), loud=10 * np.log10(np.mean(y ** 2)), bright=np.log2(np.median(cen)), air=10 * np.log10(P[f > 4000].sum() / P.sum()))
y0, p0 = dec(z0); b0 = feats_of(y0, p0); eff = np.zeros((ZD, 4)); keys = ("pitch", "loud", "bright", "air")
ys = {}
for c in range(ZD):
    z1 = z0.clone(); z1[:, c] += step; y1, p1 = dec(z1); b1 = feats_of(y1, p1); ys[c] = y1
    eff[c] = [b1["pitch"] - b0["pitch"], b1["loud"] - b0["loud"], 1200 * (b1["bright"] - b0["bright"]) / 1200 * 100, b1["air"] - b0["air"]]
units = ["pitch (cents)", "loudness (dB)", "brightness (% centroid)", "bow noise / air (dB above 4 kHz)"]
eff[:, 2] = 100 * (2 ** (eff[:, 2] / 100) - 1)                                                              # log2 ratio x100 -> percent change
fig = plt.figure(figsize=(12, 8.6)); gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.5, wspace=0.15)
ax = fig.add_subplot(gs[0, :]); En = eff / np.abs(eff).max(0)
ax.imshow(En.T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
ax.set_yticks(range(4)); ax.set_yticklabels(units); ax.set_xticks(range(ZD)); ax.set_xlabel("latent channel")
for c in range(ZD):
    for j in range(4):
        if abs(En[c, j]) > 0.35: ax.text(c, j, "%+.0f" % eff[c, j] if abs(eff[c, j]) >= 10 else "%+.1f" % eff[c, j], ha="center", va="center", fontsize=7, color="white" if abs(En[c, j]) > 0.7 else INK)
ax.set_title("What each channel does: the same push on every channel, and how the decoded sound changes", loc="left")
picks = []
for j in (0, 3):                                                         # strongest channel for pitch, brightness, air, each channel used once
    o = np.argsort(-np.abs(eff[:, j])); picks.append(int(next(c for c in o if c not in picks)))
labels = ["pitch", "bow noise / air (and quieter overall)"]
_, ps = dec(z0 + 0)                                                         # pitch track of the base decode
def avg_spec(y): f, P = ss.welch(y, SR, nperseg=4096); return f, 10 * np.log10(P + 1e-14)
for i, (c, lab) in enumerate(zip(picks, labels)):
    a3 = fig.add_subplot(gs[1, i])
    if i == 0:
        z1 = z0.clone(); z1[:, c] += step; _, p1 = dec(z1); tt = np.arange(len(p0)) / 100
        cen = lambda p: np.where(p > 40, 1200 * np.log2(np.clip(p, 40, None) / 440) / 100 + 69, np.nan)
        a3.plot(tt, cen(p0), color=INK2, lw=1.4, label="as decoded"); a3.plot(tt, cen(p1), color=BLUE, lw=1.4, label="channel %d pushed" % c)
        a3.set_xlabel("seconds"); a3.set_ylabel("pitch (MIDI)")
    else:
        f, B = avg_spec(y0); _, P1 = avg_spec(ys[c]); mm = (f > 300) & (f < 10500); sm = lambda v: np.convolve(np.pad(v, 7, mode="edge"), np.ones(15) / 15, "valid")
        a3.plot(f[mm] / 1000, sm(B[mm]), color=INK2, lw=1.4, label="as decoded"); a3.plot(f[mm] / 1000, sm(P1[mm]), color=BLUE, lw=1.4, label="channel %d pushed" % c)
        a3.set_xlabel("kHz"); a3.set_ylabel("average level (dB)"); a3.set_yticks([])
    a3.set_title("channel %d: mostly %s" % (c, lab), loc="left", fontsize=10); a3.legend(frameon=False, fontsize=8.5)
fig.text(0.125, 0.43, "Two channels up close: pitch lives in a few channels; most others mix loudness and tone colour", fontsize=10.5, weight="bold")
fig.text(0.01, 0.01, "Final codec, 2 s of the held-out passage, every channel pushed by the same 1.5 latent SDs. Heatmap colours are scaled per row; numbers show the size of the larger effects.", color=INK2, fontsize=8.5)
fig.savefig(f"{OUTD}/channel_knobs.png", dpi=110, bbox_inches="tight"); plt.close(fig); print("wrote channel_knobs.png", "| picks", picks)
np.set_printoptions(precision=1, suppress=True); print(eff)

# 3b. the same pushes heard as spectrograms: as decoded, then the pitch, brightness and bow-noise channels pushed
show = [(None, "as decoded")] + [(c, "channel %d pushed: %s" % (c, lab)) for c, lab in ((picks[0], "pitch +38 cents"), (1, "brightness, centroid +35%"), (picks[1], "bow noise / air +7.8 dB"))]
fig, axs = plt.subplots(1, 4, figsize=(14, 4), sharey=True); vmax = None
for ax, (c, lab) in zip(axs, show):
    y = y0 if c is None else ys[c]; f, t, Sd = ss.stft(y, SR, nperseg=1024, noverlap=896); S_ = 20 * np.log10(np.abs(Sd) + 1e-6); mm = f <= 8000
    vmax = S_[mm].max() if vmax is None else vmax
    ax.pcolormesh(t, f[mm] / 1000, S_[mm], shading="auto", cmap="magma", vmin=vmax - 80, vmax=vmax); ax.set_title(lab, loc="left", fontsize=10); ax.set_xlabel("seconds")
axs[0].set_ylabel("kHz")
fig.suptitle("The same 2 s decoded with one channel pushed at a time (same push size, same colour scale)", x=0.01, ha="left", fontsize=11, weight="bold")
fig.tight_layout(); fig.savefig(f"{OUTD}/channel_spectrograms.png", dpi=110); plt.close(fig); print("wrote channel_spectrograms.png")
