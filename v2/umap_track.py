"""UMAP of a whole encoded recording (16-D latent means, 10 Hz) and the path through it over time. usage: umap_track.py MU.npy OUT_PREFIX"""
import sys, numpy as np, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
Z = np.load(sys.argv[1]); Zs = (Z - Z.mean(0)) / Z.std(0); n = len(Zs) // 10; X = Zs[:n * 10].reshape(n, 10, -1).mean(1); t = np.arange(n) / 10.0
E = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=0).fit_transform(X); np.save(sys.argv[2] + "_umap2d.npy", E)
BG, TX, TS, GR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"; cm = LinearSegmentedColormap.from_list("t", ["#cfe0f7", "#2a78d6", "#0a2a52"])
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TX, "axes.labelcolor": TS, "xtick.color": TS, "ytick.color": TS})
def smooth(x, w): k = np.ones(w) / w; p = np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"); return np.convolve(p, k, "valid")
def path(ax, a, b, lw=1.4, lab=2.0):
    i, j = int(a * 10), int(b * 10); xy = np.stack([smooth(E[i:j, 0], 3), smooth(E[i:j, 1], 3)], 1); seg = np.stack([xy[:-1], xy[1:]], 1)
    lc = LineCollection(seg, cmap=cm, array=t[i:j - 1], linewidths=lw, zorder=3); lc.set_clim(a, b); ax.add_collection(lc)
    ax.scatter(*xy[0], s=60, color="#1baf7a", zorder=5, edgecolor=BG, linewidth=1.5); ax.scatter(*xy[-1], s=60, color="#eb6834", zorder=5, edgecolor=BG, linewidth=1.5)
    for tt in np.arange(a, b, lab): k = int((tt - a) * 10); ax.annotate(f"{int(tt//60)}:{int(tt%60):02d}", xy[k], xytext=(4, 4), textcoords="offset points", fontsize=8, color=TS, zorder=6)
    return lc
fig, axs = plt.subplots(1, 3, figsize=(17, 6.2), facecolor=BG, gridspec_kw={"wspace": 0.08})
ax = axs[0]; sc = ax.scatter(E[:, 0], E[:, 1], c=t / 60, s=2.5, cmap=cm, linewidths=0); ax.set_title("Whole recording: every 0.1 s, coloured by time", loc="left", fontsize=12)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.set_label("minutes", color=TS); cb.outline.set_edgecolor(GR)
for ax, (a, b, ttl) in zip(axs[1:], ((10, 30, "Path over 0:10 to 0:30 (the audio clip I sent starts here)"), (480, 500, "Path over 8:00 to 8:20 (another passage)"))):
    ax.scatter(E[:, 0], E[:, 1], s=1.5, color="#dddcd6", linewidths=0, zorder=1); path(ax, a, b); ax.set_title(ttl, loc="left", fontsize=12)
for ax in axs: ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([]); [s.set_visible(False) for s in ax.spines.values()]; ax.set_aspect("equal", "datalim")
fig.text(0.01, 0.01, "UMAP of 16 standardised latent channels (10 frames per second). Green dot = start of the window, orange dot = end, darker line = later. Axes have no units: only nearness means anything.", fontsize=9, color=TS)
fig.savefig(sys.argv[2] + "_umap_track.png", dpi=110, facecolor=BG, bbox_inches="tight")
