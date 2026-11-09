"""Before / after adversarial training on the same held-out passage: waveform macro- and micro-structure, spectrogram detail, audio.
Before = harm_final_z24lo_s2000 (the codec the GAN branch started from); after = codec_snd_s8000 (adversarial lineage, final codec).
Both decode the continuous latent from their own encoder. Writes docs/assets/gan_compare.png and audio/9_before_gan.*, 10_after_gan.*.
usage: python make_gan_fig.py     env: as gen_lat.py (codec skeleton)"""
import os, sys, copy, subprocess
sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import soundfile as sf, scipy.signal as ss, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/music-visualizer"); OUTD = f"{H}/docs/assets"; AUD = f"{OUTD}/audio"
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
pre = torch.load(f"{H}/v2/harm_final_z24lo_s2000.pt", map_location=dev, weights_only=False)
pre = pre.get("model", pre.get("ema", pre)) if isinstance(pre, dict) else pre
post = torch.load(f"{H}/colab_results/reg3/codec_snd_s8000.pt", map_location=dev, weights_only=False)["model"]
k = test[0]; S0 = 5572800 + 3 * SR; N = 4 * SR; T = N // HOP
x = torch.from_numpy(np.asarray(A[k][S0: S0 + N], dtype=np.float32))[None].to(dev)


@torch.no_grad()
def run(sd):
    missing, unexpected = model.load_state_dict(sd, strict=False); model.eval()
    if missing: print("missing keys", len(missing), missing[:3])
    z = to25(feats(x, None), T); hh = dec_feats(z, T); y = model.synth(hh, model.pitch_f0(hh), T)[0].cpu().numpy(); return y


yb, ya = run(pre), run(post); xo = x[0].cpu().numpy()
r = np.sqrt(np.mean(xo ** 2)); yb, ya = yb / np.sqrt(np.mean(yb ** 2)) * r, ya / np.sqrt(np.mean(ya ** 2)) * r


def env(y, w=480): return np.sqrt(np.convolve(y ** 2, np.ones(w) / w, "same"))                    # 20 ms RMS envelope (macro)
def spec(y): f, t, Z = ss.stft(y, SR, nperseg=1024, noverlap=896); return f, t, 20 * np.log10(np.abs(Z) + 1e-6)
def valley(y):                                                              # mean level between harmonics minus at harmonics (dB): fine structure
    f, t, S = spec(y); f0 = F0[k][S0 // 240: S0 // 240 + T]; out = []
    for j, tt in enumerate(t):
        fi = f0[min(int(tt * 100), len(f0) - 1)]
        if fi < 50: continue
        h = np.arange(1, int(4000 / fi)) * fi; pk = np.interp(h, f, S[:, j]); vl = np.interp(h + fi / 2, f, S[:, j]); out.append(np.mean(pk - vl))
    return float(np.mean(out))
def edge(y, lo=60, hi=1000):                                                # fastest low-band energy drop (dB per 10 ms), the note-edge measure
    b, a = ss.butter(4, [lo, hi], "band", fs=SR); e = 10 * np.log10(env(ss.filtfilt(b, a, y), 240) ** 2 + 1e-9)[::240]; d = np.diff(e); return float(np.percentile(d, 1))
ex = env(xo); m = {n: dict(env_corr=float(np.corrcoef(ex, env(y))[0, 1]), contrast=valley(y), edge=edge(y)) for n, y in (("original", xo), ("before", yb), ("after", ya))}
for n, v in m.items(): print(n, {a: round(b, 3) for a, b in v.items()})
for name, y in (("9_before_gan", yb), ("10_after_gan", ya)): sf.write(f"{AUD}/{name}.wav", y.astype(np.float32), SR)

# figure: shape (envelopes) matches for both; waveform detail and texture (microstructure) only look like the original after adversarial training
ev = env(xo, 2400); qd = int(np.argmin(np.diff(ev[::240]))) * 240                 # biggest loudness drop in the original: a note ending
qs = int(np.argmax(ev[: max(qd - SR // 10, SR // 10)]))                           # loudest moment before it: a sustained note
fig = plt.figure(figsize=(12, 9.6)); gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 0.9, 1.4], hspace=0.95, wspace=0.08)
ax = fig.add_subplot(gs[0, :]); tt = np.arange(N) / SR
for y, col, lw, lab in ((xo, INK2, 2.4, "original"), (yb, ORANGE, 1.1, "before adversarial training"), (ya, BLUE, 1.1, "after adversarial training")):
    ax.plot(tt, env(y), color=col, lw=lw, label=lab)
ax.axvline(qs / SR, color=GRID, lw=6, zorder=0); ax.axvspan((qd - int(0.3 * SR)) / SR, (qd + int(0.4 * SR)) / SR, color=GRID, alpha=0.6, zorder=0)
ax.set_xlim(0, N / SR); ax.set_yticks([]); ax.set_xlabel("seconds"); ax.set_ylabel("loudness"); ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
ax.set_title("Shape: both decoders follow the original's loudness (envelope correlation %.2f before, %.2f after)" % (m["before"]["env_corr"], m["after"]["env_corr"]), loc="left")
w = int(0.03 * SR); zt = np.arange(w) / SR * 1000; lim = np.abs(xo[qs: qs + w]).max() * 1.2
cols = ((xo, "Original", INK2, "original"), (yb, "Before adversarial training", ORANGE, "before"), (ya, "After adversarial training", BLUE, "after"))
for i, (y, lab, col, key) in enumerate(cols):
    a2 = fig.add_subplot(gs[1, i]); a2.plot(zt, y[qs: qs + w], color=col, lw=0.9); a2.set_ylim(-lim, lim); a2.set_yticks([]); a2.set_xlabel("ms")
    a2.set_title(lab, loc="left", color=INK if i == 0 else col, fontsize=10)
fig.text(0.125, a2.get_position().y1 + 0.045, "Waveform: 30 ms of a sustained note (narrow grey line above)", fontsize=10.5, weight="bold")
s0, s1 = qd - int(0.3 * SR), qd + int(0.4 * SR)
for i, (y, lab, col, key) in enumerate(cols):
    f, t, Sd = ss.stft(y[s0:s1], SR, nperseg=512, noverlap=448); S_ = 20 * np.log10(np.abs(Sd) + 1e-6); mm = f <= 5000
    a3 = fig.add_subplot(gs[2, i]); a3.pcolormesh(t * 1000, f[mm] / 1000, S_[mm], shading="auto", cmap="magma", vmin=S_[mm].max() - 75, vmax=S_[mm].max())
    a3.set_title("%s\nnote endings decay %.1f dB per 10 ms" % (lab, m[key]["edge"]), loc="left", color=INK if i == 0 else col, fontsize=10); a3.set_xlabel("ms")
    if i == 0: a3.set_ylabel("kHz")
    else: a3.set_yticks([])
fig.text(0.125, a3.get_position().y1 + 0.06, "Texture: a note ending up close (spectrogram of the wide grey band; brighter = louder)", fontsize=10.5, weight="bold")
fig.text(0.01, 0.01, "Same held-out Maisky passage, never trained on. Before: the codec the adversarial stage started from. After: the final codec. Both decode their own unquantized latent.", color=INK2, fontsize=8.5)
fig.savefig(f"{OUTD}/gan_compare.png", dpi=110, bbox_inches="tight"); print("wrote gan_compare.png")
for name in ("9_before_gan", "10_after_gan"):
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{AUD}/{name}.wav", "-filter_complex", "[0:a]showwaves=s=640x160:mode=cline:colors=0x2a78d6,format=yuv420p[v]",
                    "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "30", "-c:a", "aac", "-b:a", "128k", f"{AUD}/{name}.mp4"], check=True)
