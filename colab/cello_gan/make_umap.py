"""UMAP of the codec's latent space: fit on frames from the training recordings, then trace the held-out Maisky passage through it over time.
Writes docs/assets/umap_path.png (static), umap_path.gif (animation) and audio/8_umap_path.mp4 (animation with the original audio).
usage: python make_umap.py     (no GPU needed; uses the encoded latents in colab_results/reg3/tok/latents)"""
import os, subprocess, numpy as np, torch, umap, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
H = os.path.expanduser("~/music-visualizer"); LAT = f"{H}/colab_results/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; OUTD = f"{H}/docs/assets"
sp = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); k = sp["test"][0]
f0, SEC, FPS = 5572800 // 960, 12, 12.5                                     # the same 12 s as clip 1 (sample 5572800 at 24 kHz, 960 samples per frame)
BG, INK2, GRID, ORANGE = "#fcfcfb", "#52514e", "#e6e5e0", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "text.color": "#0b0b0b", "axes.labelcolor": INK2,
                     "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})


def midi_of(c): return np.where(c[:, 1] > 0.5, 69 + 12 * np.log2(np.clip(c[:, 0], 40, 1200) / 440), np.nan)


rng = np.random.default_rng(0); Z, P = [], []                              # ~600 voiced frames from each training recording
for t in sp["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); m = midi_of(np.load(f"{CTL}/{t}.npy")); n = min(len(z), len(m)); ok = np.flatnonzero(~np.isnan(m[:n]))
    pick = rng.choice(ok, min(600, len(ok)), replace=False); Z.append(z[pick]); P.append(m[pick])
Z, P = np.concatenate(Z), np.concatenate(P)
zt = np.load(f"{LAT}/{k}.npy").astype(np.float32)[f0: f0 + int(SEC * 25)]; pt = midi_of(np.load(f"{CTL}/{k}.npy"))[f0: f0 + int(SEC * 25)]
mu, sd = Z.mean(0), Z.std(0); red = umap.UMAP(n_neighbors=30, min_dist=0.15, random_state=0).fit((Z - mu) / sd)
xy, xt = red.embedding_, red.transform((zt - mu) / sd)                     # test frames are only placed, never used to fit
print("train frames", len(Z), "| test frames", len(zt))


def base(ax):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=P, cmap="Blues", vmin=np.percentile(P, 2) - 8, vmax=np.percentile(P, 99), s=2, lw=0, alpha=0.5, rasterized=True)
    ax.set_xticks([]); ax.set_yticks([]); return sc


# the pitch path: median position of every semitone (training frames), traced in pitch order through the map
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]; Pr = np.round(P).astype(int)
semis = [s_ for s_ in range(Pr.min(), Pr.max() + 1) if (Pr == s_).sum() >= 100]
med = np.array([np.median(xy[Pr == s_], 0) for s_ in semis]); semis = np.array(semis)
gap = np.r_[0, np.linalg.norm(np.diff(med, axis=0), axis=1)]               # map distance from the previous semitone
fig = plt.figure(figsize=(11, 8.6)); ax = fig.add_axes([0.04, 0.3, 0.92, 0.64]); ax2 = fig.add_axes([0.07, 0.06, 0.88, 0.17])
ax.scatter(xy[:, 0], xy[:, 1], c=P, cmap="Blues", vmin=np.percentile(P, 2) - 8, vmax=np.percentile(P, 99), s=2, lw=0, alpha=0.35, rasterized=True)
from matplotlib.collections import LineCollection; from matplotlib.colors import LinearSegmentedColormap as LSC; pcm = LSC.from_list("p", ["#f6b48f", "#eb6834", "#7a2508"])
lc = LineCollection(np.stack([med[:-1], med[1:]], 1), cmap=pcm, linewidths=2.4, zorder=3); lc.set_array(semis[:-1].astype(float)); ax.add_collection(lc)
ax.scatter(med[:, 0], med[:, 1], c=semis, cmap=pcm, s=26, edgecolor=BG, lw=1, zorder=4)
OPEN = {36: "C string", 43: "G string", 50: "D string", 57: "A string"}
for s_, (x_, y_) in zip(semis, med):
    if s_ % 12 == 0 or s_ in OPEN:
        off = {36: (-60, 10), 43: (-30, 12), 48: (10, 4), 50: (10, -14), 57: (10, -4)}.get(s_, (8, 6))
        ax.annotate("%s%d" % (NAMES[s_ % 12], s_ // 12 - 1) + ("  (open %s)" % OPEN[s_] if s_ in OPEN else ""), (x_, y_), textcoords="offset points",
                    xytext=off, fontsize=9, weight="bold", color="#0b0b0b", zorder=5, bbox=dict(boxstyle="round,pad=0.15", fc=BG, ec="none", alpha=0.8))
ax.annotate("lowest", med[0], textcoords="offset points", xytext=(-10, -14), fontsize=8.5, color=INK2); ax.annotate("highest", med[-1], textcoords="offset points", xytext=(6, -12), fontsize=8.5, color=INK2)
ax.set_xticks([]); ax.set_yticks([]); ax.set_title("The pitch path: where each semitone sits in the sound space, traced from the lowest note to the highest", loc="left")
ax2.bar(semis, gap, width=0.8, color=[ORANGE if s_ in OPEN else "#2a78d6" for s_ in semis], alpha=0.85)
for s_ in OPEN: ax2.axvline(s_ - 0.5, color=GRID, lw=1, zorder=0)
ax2.set_xticks([s_ for s_ in semis if s_ % 12 == 0 or s_ in OPEN]); ax2.set_xticklabels(["%s%d" % (NAMES[s_ % 12], s_ // 12 - 1) for s_ in semis if s_ % 12 == 0 or s_ in OPEN])
ax2.set_ylabel("step on the map"); ax2.set_title("How far each semitone moves from the one below (orange: open strings)", loc="left", fontsize=10)
fig.text(0.04, 0.005, "Median map position of all training frames at each semitone (at least 100 frames); same UMAP as the figures above.", color=INK2, fontsize=8.5)
fig.savefig(f"{OUTD}/umap_pitch_path.png", dpi=110); plt.close(fig)
for s_, g_ in zip(semis, gap): print("%s%d %4d frames step %.2f" % (NAMES[s_ % 12], s_ // 12 - 1, (Pr == s_).sum(), g_))

# static picture: the whole passage, coloured from early (light) to late (dark orange)
fig, ax = plt.subplots(figsize=(8, 6.4)); sc = base(ax); tt = np.arange(len(xt)) / 25
from matplotlib.colors import LinearSegmentedColormap; tcm = LinearSegmentedColormap.from_list("t", ["#f6b48f", "#8a2f0c"])
ax.plot(xt[:, 0], xt[:, 1], color=ORANGE, lw=0.6, alpha=0.5); s2 = ax.scatter(xt[:, 0], xt[:, 1], c=tt, cmap=tcm, vmin=0, vmax=SEC, s=9, lw=0)
ax.set_title("A held-out recording travelling through the learned sound space", loc="left")
ax.text(0.0, -0.06, "Blue: frames from the training recordings (darker = higher pitch).  Orange: the unseen Maisky passage.", transform=ax.transAxes, color=INK2, fontsize=9)
c2 = fig.colorbar(s2, ax=ax, fraction=0.04, pad=0.02); c2.set_label("test passage: seconds", color=INK2)
fig.tight_layout(); fig.savefig(f"{OUTD}/umap_path.png", dpi=110); plt.close(fig)

# animation: a moving dot with a fading trail, and a pitch timeline with a cursor underneath
fig = plt.figure(figsize=(7, 6.6)); ax = fig.add_axes([0.06, 0.3, 0.9, 0.62]); ax2 = fig.add_axes([0.1, 0.07, 0.86, 0.15]); base(ax)
ax.set_title("Held-out Maisky passage moving through the sound space", loc="left")
ax2.plot(tt, pt, color=INK2, lw=1); ax2.set_xlim(0, SEC); ax2.set_xlabel("seconds"); ax2.set_ylabel("pitch"); ax2.set_yticks([])
trail = ax.scatter([], [], s=12, lw=0, color=ORANGE); dot = ax.scatter([], [], s=90, color=ORANGE, edgecolor=BG, lw=2, zorder=5); cur = ax2.axvline(0, color=ORANGE, lw=2)
step = int(25 / FPS); TR = 40


def draw(i):
    j = min(len(xt) - 1, i * step); a = max(0, j - TR); seg = xt[a: j + 1]
    trail.set_offsets(seg); trail.set_alpha(None); trail.set_facecolor([(0.92, 0.41, 0.2, x) for x in np.linspace(0.05, 0.8, len(seg))])
    dot.set_offsets(xt[j: j + 1]); cur.set_xdata([j / 25, j / 25]); return trail, dot, cur


nf = int(SEC * FPS); an = FuncAnimation(fig, draw, frames=nf, blit=False)
an.save("/tmp/_umap_silent.mp4", writer=FFMpegWriter(fps=FPS, bitrate=900), dpi=100); plt.close(fig)
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/_umap_silent.mp4", "-vf", "fps=10,scale=520:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                f"{OUTD}/umap_path.gif"], check=True)              # palette GIF keeps the file small for GitHub
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", "/tmp/_umap_silent.mp4", "-i", f"{OUTD}/audio/1_original_maisky.wav", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "128k", "-shortest", f"{OUTD}/audio/8_umap_path.mp4"], check=True)
print("wrote umap_path.png, umap_path.gif, audio/8_umap_path.mp4")
