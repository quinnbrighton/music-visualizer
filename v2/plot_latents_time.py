"""PCA components + all channels over time for a whole encoded recording. usage: plot_latents_time.py MU.npy OUT.png"""
import sys, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
Z = np.load(sys.argv[1]); Zs = (Z - Z.mean(0)) / Z.std(0); u, s, vt = np.linalg.svd(Zs, full_matrices=False); ev = s ** 2 / (s ** 2).sum(); P = Zs @ vt.T
t = np.arange(len(Z)) / 100 / 60; BG, TX, TS, GR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"; C = ["#2a78d6", "#eb6834", "#1baf7a"]
def sm(x, w): return np.convolve(x, np.ones(w) / w, "same")
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TX, "axes.labelcolor": TS, "xtick.color": TS, "ytick.color": TS, "axes.edgecolor": GR})
fig = plt.figure(figsize=(13, 10.8), facecolor=BG); gs = fig.add_gridspec(3, 2, width_ratios=[40, 1], height_ratios=[1.1, 1.5, 1.0], hspace=0.42, wspace=0.03)
ax = fig.add_subplot(gs[0, 0])
for i in range(3): ax.plot(t, sm(P[:, i], 200), color=C[i], lw=0.9, label=f"PC{i+1} ({ev[i]*100:.0f}% of variance)")
ax.set_xlim(0, t[-1]); ax.set_ylabel("value (2 s smoothing)"); ax.set_xlabel("minutes"); ax.set_title("Top 3 PCA components across the whole recording", loc="left", fontsize=12, color=TX, pad=26); ax.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
ax2 = fig.add_subplot(gs[1, 0], sharex=ax); nb = len(Z) // 100; H = Zs[:nb * 100].reshape(nb, 100, 16).mean(1).T
cm = LinearSegmentedColormap.from_list("d", ["#2a78d6", "#f1f0ec", "#eb6834"]); im = ax2.imshow(H, aspect="auto", cmap=cm, vmin=-2.5, vmax=2.5, extent=[0, nb / 60, 16, 0], interpolation="nearest")
ax2.set_yticks(np.arange(16) + 0.5); ax2.set_yticklabels([f"ch{c}" for c in range(16)], fontsize=8)
for y0, y1, nm in ((0, 6, "fast"), (6, 11, "medium"), (11, 16, "slow")): ax2.text(-0.075, 1 - (y0 + y1) / 2 / 16, nm, transform=ax2.transAxes, rotation=90, va="center", ha="center", fontsize=9.5, color=TS)
for y in (6, 11): ax2.axhline(y, color=BG, lw=2)
ax2.set_xlabel("minutes"); ax2.set_title("Every channel over time (standardised, 1 s average): orange above its mean, blue below", loc="left", fontsize=12, color=TX)
cax = fig.add_subplot(gs[1, 1]); cb = fig.colorbar(im, cax=cax); cb.ax.tick_params(labelsize=8, colors=TS); cb.outline.set_edgecolor(GR)
ax3 = fig.add_subplot(gs[2, 0]); a, b = 1000, 3000; tt = np.arange(a, b) / 100
for i in range(3): ax3.plot(tt, P[a:b, i], color=C[i], lw=1.0, label=f"PC{i+1}")
ax3.set_xlim(10, 30); ax3.set_xlabel("seconds (0:10 to 0:30, full 100 Hz detail; the audio clip I sent is the first 15 s of this)"); ax3.set_ylabel("value"); ax3.set_title("Zoom: how the components move note to note", loc="left", fontsize=12, color=TX, pad=26); ax3.legend(frameon=False, ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=10)
for x_ in (ax, ax3): x_.set_facecolor(BG); x_.grid(color=GR, lw=0.6); [x_.spines[k].set_visible(False) for k in ("top", "right")]
fig.savefig(sys.argv[2], dpi=120, facecolor=BG, bbox_inches="tight")
