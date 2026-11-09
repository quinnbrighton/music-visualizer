"""Figure for the README: the codec as an encoder-decoder (one real second of the held-out recording -> 24x25 latent -> reconstruction),
and the size of that second at each stage drawn as area-proportional squares. Writes docs/assets/compression.png.
usage: python make_compression_fig.py     (no GPU; uses docs/assets/audio clips 1-2 and colab_results/reg3/tok/latents)"""
import os, numpy as np, torch, soundfile as sf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
H = os.path.expanduser("~/music-visualizer"); A = f"{H}/docs/assets"
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "font.size": 10})
k = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False)["test"][0]
T0 = 3.0                                                                    # which second of the 12 s clip to show
x, sr = sf.read(f"{A}/audio/1_original_maisky.wav"); y, _ = sf.read(f"{A}/audio/2_codec_reconstruction.wav")
x, y = x[int(T0 * sr): int((T0 + 1) * sr)], y[int(T0 * sr): int((T0 + 1) * sr)]
z = np.load(f"{H}/colab_results/reg3/tok/latents/{k}.npy").astype(np.float32)[5572800 // 960:][int(T0 * 25): int(T0 * 25) + 25]
z = (z - z.mean(0)) / (z.std(0) + 1e-6)

fig = plt.figure(figsize=(12, 7.6))
# top: the hourglass
ax = fig.add_axes([0.02, 0.44, 0.96, 0.52]); ax.set_xlim(0, 12); ax.set_ylim(0, 4); ax.axis("off")
t = np.linspace(0, 2.6, len(x))
ax.plot(0.2 + t, 2 + x / np.abs(x).max() * 1.3, color=INK2, lw=0.35); ax.plot(9.2 + t, 2 + y / np.abs(y).max() * 1.3, color=BLUE, lw=0.35)
ax.text(1.5, 3.65, "1 s of a held-out recording", ha="center", fontsize=11, weight="bold"); ax.text(1.5, 0.25, "24,000 numbers", ha="center", color=INK2)
ax.text(10.5, 3.65, "Rebuilt by the decoder", ha="center", fontsize=11, weight="bold", color=BLUE); ax.text(10.5, 0.25, "24,000 numbers", ha="center", color=INK2)
ax.add_patch(Polygon([[3.0, 0.6], [3.0, 3.4], [5.0, 2.55], [5.0, 1.45]], color=GRID)); ax.text(4.0, 2.0, "Encoder", ha="center", va="center", weight="bold")
ax.add_patch(Polygon([[7.0, 1.45], [7.0, 2.55], [9.0, 3.4], [9.0, 0.6]], color=GRID)); ax.text(8.0, 2.0, "Decoder", ha="center", va="center", weight="bold")
ax.imshow(z.T, extent=[5.15, 6.85, 1.15, 2.85], cmap="RdBu_r", vmin=-2.5, vmax=2.5, aspect="auto", interpolation="nearest")
ax.text(6.0, 3.15, "Latent", ha="center", fontsize=11, weight="bold", color=ORANGE); ax.text(6.0, 0.75, "24 channels × 25 frames\n= 600 numbers", ha="center", color=INK2)
# bottom: area-proportional squares, in bits per second
ax2 = fig.add_axes([0.06, 0.03, 0.9, 0.36]); ax2.axis("off")
levels = [("Raw audio (16-bit)", 384000, INK2), ("Latent (float32)", 24 * 25 * 32, BLUE), ("Quantized latent", 1800, ORANGE)]
side = {n: np.sqrt(b) for n, b, _ in levels}; S = side[levels[0][0]]
for n, b, c in levels:
    s = side[n] / S; ax2.add_patch(Rectangle((0, 0), s, s, facecolor=c, alpha=0.18 if c != ORANGE else 0.9, edgecolor=c, lw=1.5))
ax2.set_xlim(-0.02, 2.2); ax2.set_ylim(-0.02, 1.02); ax2.set_aspect("equal")
ys = [0.86, 0.5, 0.14]
for (n, b, c), yy in zip(levels, ys):
    s = side[n] / S; ax2.annotate("", xy=(s, min(s, yy) if s > 0.3 else s), xytext=(1.12, yy), arrowprops=dict(arrowstyle="-", color=c, lw=1))
    ax2.text(1.15, yy, f"{n}: {b / 1000:,.1f} kbit/s" + ("" if b == 384000 else f"   ({384000 / b:,.0f}× smaller)"), va="center", color=INK, fontsize=11, weight="bold" if c == ORANGE else None)
ax2.text(1.15, 1.0, "One second of sound, area = bits", va="center", color=INK2, fontsize=10)
fig.savefig(f"{A}/compression.png", dpi=110); print("wrote", f"{A}/compression.png")
