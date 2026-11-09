"""The sound space with pitch subtracted (and, with STAGE=2, loudness too): remove each semitone's average latent (then each loudness
band's), map what is left with UMAP and colour it by note, recording and loudness. Writes docs/assets/umap_minus_pitch.png or
umap_minus_pitch_loudness.png.   usage: [STAGE=2] python make_umap_residual.py"""
import os, json, colorsys, numpy as np, torch, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
H = os.path.expanduser("~/music-visualizer"); LAT = f"{H}/colab_results/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; OUTD = f"{H}/docs/assets"
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "text.color": INK, "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold"})
sp = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); meta = json.load(open(f"{H}/docs/recording_sources.json"))


def note_col(m, conf):                                                      # same colouring as the other maps
    pc = np.mod(np.round(m), 12); octv = np.floor(np.round(m) / 12) - 1; L = np.clip(0.22 + 0.13 * (octv - 1), 0.18, 0.8)
    S = np.clip((conf - 0.5) / 0.45, 0, 1) * 0.9 + 0.05; return np.array([(*colorsys.hls_to_rgb(h / 12, l, s), 0.55) for h, l, s in zip(pc, L, S)])


rng = np.random.default_rng(0); Z, P, CF, LD, REC, names = [], [], [], [], [], []
for t in sp["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); c = np.load(f"{CTL}/{t}.npy"); n = min(len(z), len(c)); v = np.flatnonzero(c[:n, 1] > 0.5)
    pick = rng.choice(v, min(600, len(v)), replace=False); Z.append(z[pick]); P.append(69 + 12 * np.log2(np.clip(c[pick, 0], 40, 1200) / 440))
    CF.append(c[pick, 1]); LD.append(c[pick, 2]); REC.append(np.full(len(pick), len(names))); names.append(meta.get(t, {}).get("title", t))
Z, P, CF, LD, REC = map(np.concatenate, (Z, P, CF, LD, REC)); semi = np.round(P).astype(int)
Zs = (Z - Z.mean(0)) / Z.std(); res = Zs.copy()
for k in np.unique(semi): res[semi == k] -= Zs[semi == k].mean(0)          # subtract each semitone's average position: pitch removed
STAGE = int(os.environ.get("STAGE", 1)); lb = np.digitize(LD, np.percentile(LD, np.linspace(0, 100, 25)[1:-1]))   # 24 equal-count loudness bands
if STAGE == 2:
    for k in np.unique(lb): res[lb == k] -= res[lb == k].mean(0)           # then each loudness band's average: loudness removed too
def knn_same(X, g):
    idx = NearestNeighbors(n_neighbors=11).fit(X).kneighbors(X, return_distance=False)[:, 1:]; return (g[idx] == g[:, None]).mean()
st = {nm: (knn_same(Zs, g), knn_same(res, g)) for nm, g in (("semitone", semi), ("recording", REC), ("loudness band", lb))}
for nm, (a, b) in st.items(): print("10-NN same %s: before %.2f, after subtracting pitch %.2f" % (nm, a, b))
xy = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit_transform(res)
fig, axs = plt.subplots(1, 3, figsize=(16, 5.8))
o = np.argsort(CF); axs[0].scatter(xy[o, 0], xy[o, 1], c=note_col(P[o], CF[o]), s=2, lw=0, rasterized=True)
axs[0].set_title("Coloured by note: mostly mixed, some pitch left\n(same-note neighbours %.0f%% → %.0f%%; chance 3%%)" % (100 * st["semitone"][0], 100 * st["semitone"][1]), loc="left")
cm = plt.get_cmap("tab20"); axs[1].scatter(xy[:, 0], xy[:, 1], c=[cm(r % 20) for r in REC], s=2, lw=0, alpha=0.6, rasterized=True)
axs[1].set_title("Coloured by recording: close neighbours often share one\n(same-recording neighbours %.0f%% → %.0f%%; chance 1%%)" % (100 * st["recording"][0], 100 * st["recording"][1]), loc="left")
sc = axs[2].scatter(xy[:, 0], xy[:, 1], c=LD, cmap="viridis", vmin=np.percentile(LD, 2), vmax=np.percentile(LD, 98), s=2, lw=0, alpha=0.6, rasterized=True)
axs[2].set_title(("Coloured by loudness: now the main direction\nacross the whole space" if STAGE == 1 else "Coloured by loudness: now mixed\n(same-band neighbours %.0f%% → %.0f%%; chance 4%%)" % (100 * st["loudness band"][0], 100 * st["loudness band"][1])), loc="left"); cb = fig.colorbar(sc, ax=axs[2], fraction=0.04, pad=0.01); cb.set_label("loudness (dB)", color=INK2)
for a in axs: a.set_xticks([]); a.set_yticks([]); [s_.set_visible(False) for s_ in a.spines.values()]
fig.suptitle(("The sound space with pitch subtracted: loudness becomes the main axis, and frames from the same recording stay close together" if STAGE == 1 else "Pitch and loudness both subtracted: what is left"), x=0.01, ha="left", fontsize=12, weight="bold")
fig.text(0.01, 0.01, ("Each semitone's average latent removed from its frames" + (", then each loudness band's average" if STAGE == 2 else "") + ", then a new UMAP of the remainder." + " %d training frames from %d recordings; colours repeat every 20 recordings." % (len(Z), len(names))), color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.03, 1, 0.94)); out = "umap_minus_pitch.png" if STAGE == 1 else "umap_minus_pitch_loudness.png"; fig.savefig(f"{OUTD}/{out}", dpi=110); print("wrote", out)
