"""Video: old vs regularized codec latent on the Maisky passage, side by side, with the real audio.
Top: each latent's trajectory in its own top-2 principal directions (faint = whole passage, coloured trail = last 2 s by pitch, dot = now).
Bottom: all 24 channels over a scrolling +-4 s window (each channel standardized over the passage), centre line = now.
usage: python latent_video.py NEW_CODEC.pt OUT.mp4     env: as gen_lat.py; SECONDS (40) START (Maisky test window, samples)"""
import os, sys, subprocess
_new, _mp4 = sys.argv[1:3]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_new, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval(); step = c.get("step")
import soundfile as sf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
TD = os.environ["TOK_DIR"]; DS_ = HOP * DSR; SEC = float(os.environ.get("SECONDS", 40)); k = test[0]; s0 = int(os.environ.get("START", 5572800)); f0 = s0 // DS_; n = int(SEC * 25)

with torch.no_grad():                                                       # regularized latent, encoded the same way as encode_latents.py (20 s chunks, 1 s context)
    a = A[k]; parts = []; CH, CT = 20 * SR, SR
    for c0 in range(s0, s0 + n * DS_, CH):
        lo_, hi_ = max(0, c0 - CT), min(len(a), c0 + CH + CT); lo_ -= lo_ % DS_; x = torch.from_numpy(np.asarray(a[lo_:hi_], dtype=np.float32))[None].to(dev)
        z = to25(feats(x, None), (hi_ - lo_) // HOP); o = (c0 - lo_) // DS_; parts.append(z[0, :, o: o + CH // DS_].t().cpu())
    Znew = torch.cat(parts)[:n].numpy()
Zold = np.load(f"{TD}/latents/{k}.npy").astype(np.float32)[f0: f0 + n]
craw = np.load(f"{TD}/controls/{k}.npy")[f0: f0 + n]; voiced = craw[:, 1] > 0.5; midi = np.where(voiced, 69 + 12 * np.log2(np.clip(craw[:, 0], 40, 1200) / 440), np.nan)


def pca2(Z):
    Zc = Z - Z.mean(0); U, S, Vt = np.linalg.svd(Zc, full_matrices=False); ev = S ** 2 / (S ** 2).sum(); return Zc @ Vt[:2].T, ev[:2]


def corr_stats(Z):
    C = np.corrcoef(Z.T); off = np.abs(C[~np.eye(len(C), dtype=bool)]); return off.mean(), off.max()


P = {"old": pca2(Zold), "new": pca2(Znew)}; Zs = {"old": (Zold - Zold.mean(0)) / Zold.std(0), "new": (Znew - Znew.mean(0)) / Znew.std(0)}; CS = {"old": corr_stats(Zold), "new": corr_stats(Znew)}
TITLE = {"old": "Old codec latent (64x12 RVQ codec, unquantized)", "new": f"Regularized codec latent (step {step})"}
FPS, TRAIL, WIN = 25, 50, 100                                               # video at the latent frame rate; 2 s trail; +-4 s channel window
vmin, vmax = np.nanpercentile(midi, 2), np.nanpercentile(midi, 98); cmap = plt.get_cmap("viridis")
names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
fig = plt.figure(figsize=(16, 9), dpi=80, facecolor="#111"); gs = fig.add_gridspec(2, 2, height_ratios=[1.35, 1], hspace=0.28, wspace=0.12, left=0.04, right=0.98, top=0.9, bottom=0.06)
txt = fig.text(0.5, 0.955, "", color="w", ha="center", fontsize=18)
art = {}
for col, key in enumerate(["old", "new"]):
    xy, ev = P[key]; ax = fig.add_subplot(gs[0, col], facecolor="#111"); ax.plot(xy[:, 0], xy[:, 1], color="#444", lw=0.5)
    pad = 0.05 * (xy.max(0) - xy.min(0)); ax.set_xlim(xy[:, 0].min() - pad[0], xy[:, 0].max() + pad[0]); ax.set_ylim(xy[:, 1].min() - pad[1], xy[:, 1].max() + pad[1])
    lc = LineCollection([], cmap=cmap, norm=plt.Normalize(vmin, vmax), lw=2.5); ax.add_collection(lc); dot, = ax.plot([], [], "o", ms=12, mfc="w", mec="k")
    mc, xc = CS[key]; ax.set_title(f"{TITLE[key]}\ntop-2 directions hold {ev.sum()*100:.0f}% of variation | channel correlation mean {mc:.2f}, max {xc:.2f}", color="w", fontsize=11)
    ax.tick_params(colors="#888"); [sp.set_color("#444") for sp in ax.spines.values()]
    ax2 = fig.add_subplot(gs[1, col], facecolor="#111"); padz = np.pad(Zs[key], ((WIN, WIN), (0, 0)), constant_values=np.nan)
    im = ax2.imshow(padz[:2 * WIN].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[-4, 4, 23.5, -0.5], interpolation="nearest"); ax2.axvline(0, color="w", lw=1)
    ax2.set_ylabel("channel", color="#aaa"); ax2.set_xlabel("seconds from now", color="#aaa"); ax2.tick_params(colors="#888"); ax2.set_yticks(range(0, 24, 4))
    art[key] = (xy, lc, dot, im, padz)
cb = fig.add_axes([0.44, 0.405, 0.12, 0.012]); fig.colorbar(plt.cm.ScalarMappable(plt.Normalize(vmin, vmax), cmap), cax=cb, orientation="horizontal"); cb.tick_params(colors="#aaa", labelsize=8); cb.set_title("trail colour = pitch (MIDI)", color="#aaa", fontsize=8)

silent = _mp4.replace(".mp4", "_silent.mp4"); W, H = fig.canvas.get_width_height()
ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "20", silent], stdin=subprocess.PIPE)
for t in range(n):
    for key in ("old", "new"):
        xy, lc, dot, im, padz = art[key]; a0 = max(0, t - TRAIL); seg = np.stack([xy[a0:t], xy[a0 + 1:t + 1]], 1) if t > a0 else np.zeros((0, 2, 2))
        lc.set_segments(seg); lc.set_array(np.nan_to_num(midi[a0 + 1:t + 1], nan=vmin)); lc.set_alpha(None); dot.set_data([xy[t, 0]], [xy[t, 1]]); im.set_data(padz[t: t + 2 * WIN].T)
    m = midi[t]; txt.set_text(f"Maisky   {t/25:5.1f} s   " + (f"{names[int(round(m)) % 12]}{int(round(m)) // 12 - 1}" if np.isfinite(m) else "(no pitch)"))
    fig.canvas.draw(); ff.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
ff.stdin.close(); ff.wait()
wav = _mp4.replace(".mp4", ".wav"); x = np.asarray(A[k][s0: s0 + n * DS_], np.float32); sf.write(wav, x / (np.abs(x).max() + 1e-6) * 0.7, SR)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", silent, "-i", wav, "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", _mp4], check=True); os.remove(silent)
print("wrote", _mp4, "| old corr", CS["old"], "new corr", CS["new"], "| PCA top2 old", P["old"][1], "new", P["new"][1])
