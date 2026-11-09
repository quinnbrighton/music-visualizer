"""Where each recording sits once pitch (and then loudness) is subtracted: the median map position of every training recording,
drawn on the two residual maps. Writes docs/assets/umap_recording_centroids.png.   usage: python make_recording_centroids.py"""
import os, re, json, numpy as np, torch, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/music-visualizer"); LAT = f"{H}/colab_results/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; OUTD = f"{H}/docs/assets"
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold"})
sp = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); meta = json.load(open(f"{H}/docs/recording_sources.json"))


def short(title):                                                           # a few words that identify the recording (usually the performer)
    t = re.split(r" plays | - | – | \| |: |, ", title)[0]; t = re.sub(r"\[.*?\]|\(.*?\)|#\w+", "", t).strip(); return t[:22]


rng = np.random.default_rng(0); Z, P, LD, REC, names = [], [], [], [], []      # same sample as the other residual maps
for t in sp["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); c = np.load(f"{CTL}/{t}.npy"); n = min(len(z), len(c)); v = np.flatnonzero(c[:n, 1] > 0.5)
    pick = rng.choice(v, min(600, len(v)), replace=False); Z.append(z[pick]); P.append(69 + 12 * np.log2(np.clip(c[pick, 0], 40, 1200) / 440))
    LD.append(c[pick, 2]); REC.append(np.full(len(pick), len(names))); names.append(short(meta.get(t, {}).get("title", t)))
Z, P, LD, REC = map(np.concatenate, (Z, P, LD, REC)); semi = np.round(P).astype(int); lb = np.digitize(LD, np.percentile(LD, np.linspace(0, 100, 25)[1:-1]))
Zs = (Z - Z.mean(0)) / Z.std(); r1 = Zs.copy()
for k in np.unique(semi): r1[semi == k] -= Zs[semi == k].mean(0)
r2 = r1.copy()
for k in np.unique(lb): r2[lb == k] -= r1[lb == k].mean(0)
cm = plt.get_cmap("tab20"); fig, axs = plt.subplots(1, 2, figsize=(16, 7.4))
for ax, res, ttl in ((axs[0], r1, "Pitch subtracted"), (axs[1], r2, "Pitch and loudness subtracted")):
    xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(res)
    cen = np.array([np.median(xy[REC == r], 0) for r in range(len(names))])
    spread = np.median([np.median(np.linalg.norm(xy[REC == r] - cen[r], axis=1)) for r in range(len(names))])   # typical distance of a frame from its recording's centre
    between = np.median(np.linalg.norm(cen - np.median(cen, 0), axis=1))                                       # typical distance of a recording's centre from the middle
    ax.scatter(xy[:, 0], xy[:, 1], color="#c9c8c3", s=1.5, lw=0, alpha=0.5, rasterized=True)
    ax.scatter(cen[:, 0], cen[:, 1], c=[cm(r % 20) for r in range(len(names))], s=55, edgecolor=INK, lw=0.7, zorder=4)
    d = np.linalg.norm(cen - np.median(cen, 0), axis=1)
    for r in np.flatnonzero(d > 1.5 * between):                             # label only the recordings that sit apart from the crowd
        ax.annotate(names[r], cen[r], textcoords="offset points", xytext=(5, 4), fontsize=7.5, color=INK, zorder=5, bbox=dict(boxstyle="round,pad=0.12", fc=BG, ec="none", alpha=0.8))
    ax.set_title("%s\nrecording centres spread %.1f; frames sit %.1f from their own centre" % (ttl, between, spread), loc="left")
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
    print(ttl, "| centre spread %.2f | within-recording spread %.2f" % (between, spread))
fig.suptitle("Each dot is one training recording's centre (median position of its frames) on the residual maps", x=0.01, ha="left", fontsize=12, weight="bold")
fig.text(0.01, 0.01, "Grey: all %d training frames. Distances in map units; 'spread' = median distance of the recording centres from the middle of the map. Labelled: recordings more than 1.5x that far out." % len(Z), color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(f"{OUTD}/umap_recording_centroids.png", dpi=110); print("wrote umap_recording_centroids.png")
