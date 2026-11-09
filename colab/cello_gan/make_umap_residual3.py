"""What is left after subtracting pitch, loudness and each recording's average: map the remainder and colour it by the per-frame
playing features we can measure. Writes docs/assets/umap_minus_pitch_loudness_recording.png.   usage: python make_umap_residual3.py"""
import os, colorsys, numpy as np, torch, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
H = os.path.expanduser("~/music-visualizer"); LAT = f"{H}/colab_results/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; OUTD = f"{H}/docs/assets"
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "font.size": 10, "axes.titlesize": 10, "axes.titleweight": "bold"})
sp = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False)


def note_col(m, conf):
    pc = np.mod(np.round(m), 12); octv = np.floor(np.round(m) / 12) - 1; L = np.clip(0.22 + 0.13 * (octv - 1), 0.18, 0.8)
    S = np.clip((conf - 0.5) / 0.45, 0, 1) * 0.9 + 0.05; return np.array([(*colorsys.hls_to_rgb(h / 12, l, s), 0.55) for h, l, s in zip(pc, L, S)])


rng = np.random.default_rng(0); Z, P, CF, LD, SL, REC = [], [], [], [], [], []   # same sample as the other residual maps
for t in sp["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); c = np.load(f"{CTL}/{t}.npy"); n = min(len(z), len(c)); v = np.flatnonzero(c[:n, 1] > 0.5)
    pick = rng.choice(v, min(600, len(v)), replace=False); Z.append(z[pick]); P.append(69 + 12 * np.log2(np.clip(c[pick, 0], 40, 1200) / 440))
    CF.append(c[pick, 1]); LD.append(c[pick, 2]); a, b = np.clip(pick - 2, 0, n - 1), np.clip(pick + 2, 0, n - 1); SL.append((c[b, 2] - c[a, 2]) / 4 * 25 / 10)   # dB per 40 ms... per 100 ms
    REC.append(np.full(len(pick), len(REC)))
Z, P, CF, LD, SL, REC = map(np.concatenate, (Z, P, CF, LD, SL, REC)); semi = np.round(P).astype(int); CENTS = 100 * (P - semi)
lb = np.digitize(LD, np.percentile(LD, np.linspace(0, 100, 25)[1:-1]))
Zs = (Z - Z.mean(0)) / Z.std(); res = Zs.copy()
for g in (semi, lb, REC):                                                   # peel the layers in order: pitch, loudness, recording
    for k in np.unique(g): res[g == k] -= res[g == k].mean(0)
left = (res ** 2).sum() / ((Zs - Zs.mean(0)) ** 2).sum(); print("variance left after removing pitch, loudness, recording: %.0f%%" % (100 * left))
def bins(x, k=12): return np.digitize(x, np.percentile(x, np.linspace(0, 100, k + 1)[1:-1]))
idx = NearestNeighbors(n_neighbors=11).fit(res).kneighbors(res, return_distance=False)[:, 1:]
def r2(g): tot = (res ** 2).sum(); return 1 - sum(((res[g == k] - res[g == k].mean(0)) ** 2).sum() for k in np.unique(g)) / tot
stats = {}
for nm, g in (("note", semi), ("recording", REC), ("loudness", lb), ("offset from the semitone", bins(CENTS)), ("loudness rising / falling", bins(SL)), ("pitch confidence", bins(CF))):
    same = (g[idx] == g[:, None]).mean(); chance = np.sum((np.bincount(g) / len(g)) ** 2); stats[nm] = (same, chance, r2(g))
    print("%-26s 10-NN same %.2f (chance %.2f) | explains %.1f%% of what is left" % (nm, same, chance, 100 * r2(g)))
xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(res)
fig, axs = plt.subplots(2, 3, figsize=(16, 10)); axs = axs.ravel(); o = np.argsort(CF)
panels = [("note", None), ("recording", None), ("loudness", (LD, "viridis", "dB")), ("offset from the semitone", (CENTS, "RdBu_r", "cents")),
          ("loudness rising / falling", (SL, "PuOr_r", "dB per 100 ms")), ("pitch confidence", (CF, "magma", "confidence"))]
for ax, (nm, spec) in zip(axs, panels):
    if nm == "note": ax.scatter(xy[o, 0], xy[o, 1], c=note_col(P[o], CF[o]), s=1.6, lw=0, rasterized=True)
    elif nm == "recording": cm = plt.get_cmap("tab20"); ax.scatter(xy[:, 0], xy[:, 1], c=[cm(r % 20) for r in REC], s=1.6, lw=0, alpha=0.6, rasterized=True)
    else:
        x, cmap, unit = spec; lo, hi = np.percentile(x, 2), np.percentile(x, 98)
        if cmap in ("RdBu_r", "PuOr_r"): hi = max(abs(lo), abs(hi)); lo = -hi
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=x, cmap=cmap, vmin=lo, vmax=hi, s=1.6, lw=0, alpha=0.6, rasterized=True); cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.01); cb.set_label(unit, color=INK2)
    same, chance, r = stats[nm]; ax.set_title("%s\nneighbours alike %.0f%% (chance %.0f%%); explains %.1f%%" % (nm.capitalize(), 100 * same, 100 * chance, 100 * r), loc="left")
    ax.set_xticks([]); ax.set_yticks([]); [s_.set_visible(False) for s_ in ax.spines.values()]
fig.suptitle("Pitch, loudness and recording all subtracted: %.0f%% of the latent's variance is left. What organizes it?" % (100 * left), x=0.01, ha="left", fontsize=12, weight="bold")
fig.text(0.01, 0.01, "Each layer's group averages removed in turn (semitone, 24 loudness bands, recording), then a new UMAP. 'Explains' = share of the remaining variance accounted for by 12 bins of each feature.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.95)); fig.savefig(f"{OUTD}/umap_minus_pitch_loudness_recording.png", dpi=100); print("wrote umap_minus_pitch_loudness_recording.png")
