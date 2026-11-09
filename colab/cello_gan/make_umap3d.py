"""Spinning 3D UMAP of the codec's latent space, every frame coloured by its note (hue = note name, brightness = octave,
saturation = pitch confidence). Writes docs/assets/umap3d_spin.gif.   usage: python make_umap3d.py"""
import os, subprocess, colorsys, numpy as np, torch, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/music-visualizer"); LAT = f"{H}/colab_results/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; OUTD = f"{H}/docs/assets"
sp = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False)
BG = "#fcfcfb"; NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def midi_of(c): return np.where(c[:, 1] > 0.5, 69 + 12 * np.log2(np.clip(c[:, 0], 40, 1200) / 440), np.nan)
def note_col(m, conf, alpha=0.55):                                          # same colouring as the 2D map
    pc = np.mod(np.round(m), 12); octv = np.floor(np.round(m) / 12) - 1; L = np.clip(0.22 + 0.13 * (octv - 1), 0.18, 0.8)
    S = np.clip((conf - 0.5) / 0.45, 0, 1) * 0.9 + 0.05; return np.array([(*colorsys.hls_to_rgb(h / 12, l, s), alpha) for h, l, s in zip(pc, L, S)])


rng = np.random.default_rng(0); Z, P, CF = [], [], []
for t in sp["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); cc = np.load(f"{CTL}/{t}.npy"); m = midi_of(cc); n = min(len(z), len(m)); ok = np.flatnonzero(~np.isnan(m[:n]))
    pick = rng.choice(ok, min(600, len(ok)), replace=False); Z.append(z[pick]); P.append(m[pick]); CF.append(cc[pick, 1])
Z, P, CF = np.concatenate(Z), np.concatenate(P), np.concatenate(CF)
xyz = umap.UMAP(n_components=3, n_neighbors=30, min_dist=0.15, random_state=0).fit_transform((Z - Z.mean(0)) / Z.std(0))
xyz = (xyz - xyz.mean(0)) / xyz.std(0); keep = rng.choice(len(xyz), 24000, replace=False); o = keep[np.argsort(CF[keep])]
col = note_col(P[o], CF[o])
fig = plt.figure(figsize=(6.4, 6.4), facecolor=BG); ax = fig.add_axes([0, 0.0, 1, 0.9], projection="3d"); ax.set_facecolor(BG)
def rot(a, axis):                                                          # rotation matrix about the x or y axis
    c_, s_ = np.cos(a), np.sin(a); return {"x": np.array([[1, 0, 0], [0, c_, -s_], [0, s_, c_]]), "y": np.array([[c_, 0, s_], [0, 1, 0], [-s_, 0, c_]]), "z": np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])}[axis]
xyz = xyz @ rot(np.pi / 2, "y").T                                          # turn the cloud 90 degrees, then spin it about the vertical axis
xyz -= (np.percentile(xyz, 1, 0) + np.percentile(xyz, 99, 0)) / 2           # centre the bulk of the cloud in the frame
sc = ax.scatter(xyz[o, 0], xyz[o, 1], xyz[o, 2], c=col, s=3, lw=0, depthshade=False, alpha=None); ax.set_axis_off()
Pr = np.round(P).astype(int); semis = np.array([s_ for s_ in range(Pr.min(), Pr.max() + 1) if (Pr == s_).sum() >= 100])
cnt = np.array([(Pr == s_).sum() for s_ in semis]); cen = np.array([np.median(xyz[Pr == s_], 0) for s_ in semis])   # centroid of every semitone
bc = note_col(semis.astype(float), np.ones(len(semis)), 1.0); bs = 30 + 520 * cnt / cnt.max()     # blob area proportional to how often the note is played
sb = ax.scatter(cen[:, 0], cen[:, 1], cen[:, 2], c=bc, s=bs, edgecolor="#0b0b0b", lw=0.8, depthshade=False, zorder=5)
OPEN = {36: "open C", 43: "open G", 50: "open D", 57: "open A"}
r = np.percentile(np.linalg.norm(xyz, axis=1), 99); c0 = np.zeros(3)                  # sphere that holds the cloud at any rotation
ax.set_xlim(c0[0] - r, c0[0] + r); ax.set_ylim(c0[1] - r, c0[1] + r); ax.set_zlim(c0[2] - r, c0[2] + r); ax.set_box_aspect((1, 1, 1), zoom=1.6)
fig.text(0.03, 0.95, "The cello sound space in 3D", fontsize=12, weight="bold")
fig.text(0.03, 0.885, "hue = note name, brightness = octave, saturation = pitch confidence\nblobs = centroid of each note, sized by how often it is played", fontsize=8.5, color="#52514e")
tmp = "/tmp/_umap3d"; os.makedirs(tmp, exist_ok=True)
ax.view_init(elev=10, azim=-90)                                            # camera fixed; z is vertical on screen
for i, ang in enumerate(range(0, 360, 5)):
    R = rot(np.radians(ang), "z").T; q = xyz[o] @ R; sc._offsets3d = (q[:, 0], q[:, 1], q[:, 2]); qc = cen @ R; sb._offsets3d = (qc[:, 0], qc[:, 1], qc[:, 2])
    lab = [ax.text(qc[j, 0], qc[j, 1], qc[j, 2] + 0.12 * r, OPEN[s_], fontsize=9, weight="bold", ha="center", zorder=10, bbox=dict(boxstyle="round,pad=0.2", fc=BG, ec="#0b0b0b", lw=0.6))
           for j, s_ in enumerate(semis) if s_ in OPEN]
    fig.savefig(f"{tmp}/f{i:03d}.png", dpi=80, facecolor=BG)
    for t_ in lab: t_.remove()
plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-i", f"{tmp}/f%03d.png", "-vf",
                "scale=420:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", f"{OUTD}/umap3d_spin.gif"], check=True)
print("wrote umap3d_spin.gif", os.path.getsize(f"{OUTD}/umap3d_spin.gif") // 1024, "KB")
