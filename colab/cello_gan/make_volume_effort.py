"""Volume and effort as two independent latent controls: decode the held-out Maisky passage on a grid of gain pushes x effort (tone)
pushes and measure level and tone at each point; plus two listening clips (volume only, effort only at matched level).
Writes docs/assets/volume_vs_effort.png, docs/assets/audio/13_volume_only.{wav,mp4}, 14_effort_only.{wav,mp4}.
usage: python make_volume_effort.py   (needs neuralcello on PYTHONPATH and models/gain_direction.npy, tone_direction.npy from gain_vs_tone_directions.py)"""
import os, subprocess, numpy as np, soundfile as sf, scipy.signal as ss, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import neuralcello as nc
H = os.path.expanduser("~/music-visualizer"); A = f"{H}/docs/assets"; m = nc.load(); SR = nc.SR
BG, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "font.size": 10,
                     "axes.titlesize": 10.5, "axes.titleweight": "bold"})
g, t = np.load(f"{H}/models/gain_direction.npy"), np.load(f"{H}/models/tone_direction.npy")
x, sr = sf.read(f"{A}/audio/1_original_maisky.wav"); z = m.encode(x, sr); rms = lambda v: np.sqrt(np.mean(v ** 2))
def upper(y): f, P = ss.welch(y, SR, nperseg=4096); return 10 * np.log10(P[(f > 1000) & (f < 4000)].sum() / P[f < 1000].sum())   # level-independent tone measure
y0 = m.decode(z); L0, U0 = 20 * np.log10(rms(y0)), upper(y0)
GAINS, EFFORT = [-10, -5, 0, 5, 10], [-60, -30, 0, 30, 60]                  # gain in dB-equivalents of the encoder's volume axis; effort in latent steps
LV, TN, Y = np.zeros((5, 5)), np.zeros((5, 5)), {}
for i, gd in enumerate(GAINS):
    for j, e in enumerate(EFFORT):
        y = m.decode(z + gd * g + e * t); LV[i, j], TN[i, j] = 20 * np.log10(rms(y)) - L0, upper(y) - U0; Y[gd, e] = y
        print("gain %+3d effort %+3d | level %+5.1f dB | upper harmonics %+5.2f dB" % (gd, e, LV[i, j], TN[i, j]), flush=True)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, M_, ttl, cm in ((axs[0], LV, "Level change (dB)", "PuOr_r"), (axs[1], TN, "Tone change: upper harmonics vs fundamentals (dB)", "RdBu_r")):
    v = np.abs(M_).max(); ax.imshow(M_, cmap=cm, vmin=-v, vmax=v, origin="lower")
    for i in range(5):
        for j in range(5): ax.text(j, i, "%+.1f" % M_[i, j], ha="center", va="center", fontsize=9, color="white" if abs(M_[i, j]) > 0.6 * v else INK)
    ax.set_xticks(range(5)); ax.set_xticklabels(["%+d" % e for e in EFFORT]); ax.set_yticks(range(5)); ax.set_yticklabels(["%+d" % gd for gd in GAINS])
    ax.set_xlabel("effort push (latent steps)"); ax.set_ylabel("volume push (dB on the encoder's volume axis)"); ax.set_title(ttl, loc="left")
fig.suptitle("Two independent controls: volume moves the level (left, top to bottom); effort moves the tone (right, left to right)", x=0.01, ha="left", fontsize=11.5, weight="bold")
fig.text(0.01, 0.01, "Held-out Maisky passage (12 s), decoded at every combination. Tone is measured on level-normalised audio, so it does not depend on volume.", color=INK2, fontsize=8.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.93)); fig.savefig(f"{A}/volume_vs_effort.png", dpi=110); plt.close(fig)

seg = lambda y: y[7 * SR: 12 * SR]; gap = np.zeros(int(0.7 * SR)); font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
def clip(name, parts):                                                      # parts: (label, audio); one shared gain so relative levels are kept
    a = np.concatenate(sum([[seg(y), gap] for _, y in parts], [])[:-1]); a = a / np.abs(a).max() * 0.9; sf.write(f"{A}/audio/{name}.wav", a.astype(np.float32), SR)
    ins, ov, prev, D = [], "", "[w]", 5.0 + 0.7
    for i, (lab, _) in enumerate(parts):
        im = Image.new("RGBA", (720, 40), (0, 0, 0, 0)); ImageDraw.Draw(im).text((16, 8), lab, font=font, fill=(255, 255, 255, 255)); im.save(f"/tmp/_ve{i}.png"); ins += ["-loop", "1", "-i", f"/tmp/_ve{i}.png"]
        ov += f"{prev}[{i + 1}:v]overlay=0:0:enable='gte(t,{i * D:.2f})*lt(t,{(i + 1) * D:.2f})':shortest=1[o{i}];"; prev = f"[o{i}]"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{A}/audio/{name}.wav"] + ins + ["-filter_complex", "[0:a]showwaves=s=720x200:mode=cline:draw=full:colors=0x2a78d6,format=yuv420p[w];"
                    + ov + f"{prev}format=yuv420p[v]", "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "28", "-c:a", "aac", "-b:a", "160k", "-shortest", f"{A}/audio/{name}.mp4"], check=True)
clip("13_volume_only", [("1. Volume push -10", Y[-10, 0]), ("2. As decoded", y0), ("3. Volume push +10", Y[10, 0])])
lm = lambda y: y / rms(y) * rms(y0)                                         # effort clips at matched level: only the playing changes
clip("14_effort_only", [("1. Effort -60 (lighter bow)", lm(Y[0, -60])), ("2. As decoded", y0), ("3. Effort +60 (heavier bow)", lm(Y[0, 60]))])
print("wrote volume_vs_effort.png, 13_volume_only.mp4, 14_effort_only.mp4")
