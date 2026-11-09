"""Re-plot a saved UMAP map: panel 1 coloured by pitch (are the islands notes?), panels 2-3 the path over short windows coloured by time. usage: plot_umap.py UMAP2D.npy KEY OUT.png"""
import sys, os, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
E = np.load(sys.argv[1]); n = len(E); f0 = np.load(os.path.expanduser(f"~/music-visualizer/v2/f0clean/{sys.argv[2]}.npy"))[:n * 10]; cents = (1200 * np.log2(np.clip(f0, 40, 1200) / 65.41)).reshape(n, 10); pc = np.median(cents, 1) / 100    # semitones above C2
t = np.arange(n) / 10.0; BG, TX, TS, GR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
seq = LinearSegmentedColormap.from_list("s", ["#d6e6fa", "#2a78d6", "#08204a"]); plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TX, "axes.labelcolor": TS})
fig, axs = plt.subplots(1, 3, figsize=(18, 6.4), facecolor=BG, gridspec_kw={"wspace": 0.06})
ax = axs[0]; o = np.argsort(pc); sc = ax.scatter(E[o, 0], E[o, 1], c=pc[o], s=3, cmap=seq, linewidths=0); ax.set_title("Map coloured by pitch", loc="left", fontsize=13)
cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01, location="bottom"); cb.set_label("pitch, semitones above C2 (65 Hz); each octave is 12", color=TS); cb.outline.set_edgecolor(GR)
def path(ax, a, b, ttl):
    i, j = int(a * 10), int(b * 10); xy = E[i:j]; ax.scatter(E[:, 0], E[:, 1], s=1.5, color="#e0dfd9", linewidths=0, zorder=1)
    seg = np.stack([xy[:-1], xy[1:]], 1); lc = LineCollection(seg, cmap=seq, array=t[i:j - 1], linewidths=1.3, alpha=0.9, zorder=3); lc.set_clim(a, b); ax.add_collection(lc)
    ax.scatter(xy[:, 0], xy[:, 1], c=t[i:j], cmap=seq, vmin=a, vmax=b, s=14, zorder=4, edgecolor=BG, linewidth=0.5)
    ax.scatter(*xy[0], s=90, color="#1baf7a", zorder=6, edgecolor=BG, linewidth=1.5); ax.scatter(*xy[-1], s=90, color="#eb6834", zorder=6, edgecolor=BG, linewidth=1.5)
    for k in range(0, j - i, 5): ax.annotate(f"{int(t[i+k]//60)}:{t[i+k]%60:04.1f}", xy[k], xytext=(5, 5), textcoords="offset points", fontsize=8, color=TS, zorder=7)
    ax.set_title(ttl, loc="left", fontsize=13); return lc
path(axs[1], 10, 16, "Path, 0:10 to 0:16"); path(axs[2], 480, 486, "Path, 8:00 to 8:06")
for a_ in axs: a_.set_facecolor(BG); a_.set_xticks([]); a_.set_yticks([]); [s.set_visible(False) for s in a_.spines.values()]; a_.set_aspect("equal", "datalim")
fig.text(0.01, 0.005, os.environ.get("LABEL", "UMAP") + " of 16 standardised latent channels, 10 frames per second. Green = start of the window, orange = end, darker = later, a label every 0.5 s. Axes have no units: only nearness means anything.", fontsize=9, color=TS)
fig.savefig(sys.argv[3], dpi=110, facecolor=BG, bbox_inches="tight")
