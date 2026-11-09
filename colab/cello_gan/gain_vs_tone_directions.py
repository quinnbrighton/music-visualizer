"""Split the learned loudness direction into pure gain and tone. Gain direction: encode the same audio at -6, 0 and +6 dB and average the
latent change per dB. Tone direction: the loudness direction with its gain component removed. Each is then decoded on the held-out
Maisky passage to check what it changes. Writes models/gain_direction.npy, models/tone_direction.npy, docs/assets/gain_vs_tone.json.
usage: python gain_vs_tone_directions.py     (uses the neuralcello package and the dataset audio under ~/music-visualizer)"""
import os, glob, json, numpy as np, soundfile as sf, torch, scipy.signal as ss
import neuralcello as nc
H = os.path.expanduser("~/music-visualizer"); m = nc.load(); SR = nc.SR
spl = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); meta = json.load(open(f"{H}/docs/recording_sources.json"))
FILES = {os.path.basename(f): f for d in ("cello_dataset", "cleaned/cello_dataset", "oos_test") for f in glob.glob(f"{H}/{d}/*.wav")}
dirL = np.load(f"{H}/models/loudness_direction.npy")


def clip(k, sec=20):                                                        # 20 s from the middle of a recording, 24 kHz mono
    f = FILES.get(meta[k]["file"]) if k in meta else None
    if f is None: return None
    info = sf.info(f); a = max(0, int(info.frames / 2 - sec * info.samplerate / 2)); x, sr = sf.read(f, start=a, frames=int(sec * info.samplerate), dtype="float32")
    x = x.mean(1) if x.ndim > 1 else x; x = ss.resample_poly(x, SR, sr) if sr != SR else x; return x / (np.abs(x).max() + 1e-9) * 0.3   # headroom for +6 dB


G, per = [], []
recs = [k for k in list(spl["val"]) + list(spl["train"])[::3]]
for k in recs:
    x = clip(k)
    if x is None: continue
    z = {g: m.encode(x * 10 ** (g / 20), SR) for g in (-6, 0, 6)}
    up, dn = (z[6] - z[0]) / 6, (z[0] - z[-6]) / 6; G.append(np.r_[up, dn]); per.append(np.mean(np.r_[up, dn], 0))
    print(len(per), k[:12], "up/down agreement (cosine) %.2f" % (up.mean(0) @ dn.mean(0) / np.linalg.norm(up.mean(0)) / np.linalg.norm(dn.mean(0))), flush=True)
g = np.concatenate(G).mean(0); per = np.array(per); cos = lambda a, b: float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
consist = np.median([cos(p, g) for p in per]); gh = g / np.linalg.norm(g); tone = dirL - (dirL @ gh) * gh
print("gain direction: %.3f latent units per dB; recordings agree with it at median cosine %.2f" % (np.linalg.norm(g), consist))
print("learned loudness direction vs gain: cosine %.2f; share of its length along gain %.0f%%, tonal %.0f%%" % (cos(dirL, g), 100 * (dirL @ gh) ** 2 / (dirL @ dirL), 100 * (tone @ tone) / (dirL @ dirL)))

# decode checks on the held-out Maisky clip: what does each direction change?
x, sr = sf.read(f"{H}/docs/assets/audio/1_original_maisky.wav"); z = m.encode(x, sr); y0 = m.decode(z); rms = lambda v: np.sqrt(np.mean(v ** 2))
def tone_stats(y):
    f, P = ss.welch(y / rms(y), SR, nperseg=4096); return dict(upper_db=10 * np.log10(P[(f > 1000) & (f < 4000)].sum() / P[f < 1000].sum()), low_db=10 * np.log10(P[f < 125].sum() / P.sum()),
                                                              centroid=float((f * P).sum() / P.sum()))
b0 = tone_stats(y0); out = {"gain_norm_per_db": float(np.linalg.norm(g)), "gain_consistency_cos": float(consist), "cos_loudness_gain": cos(dirL, g)}
for nm, v in (("gain direction x10 dB", 10 * g), ("tone direction (same length as a +20 loudness push's tonal part)", 20 * tone), ("learned loudness direction x20", 20 * dirL)):
    y1 = m.decode(z + v); b1 = tone_stats(y1); r = dict(level_db=float(20 * np.log10(rms(y1) / rms(y0))), **{a: float(b1[a] - b0[a]) for a in b0})
    out[nm] = r; print("%-66s level %+5.1f dB | upper harmonics %+.2f dB | below 125 Hz %+.2f dB | centroid %+.0f Hz" % (nm, r["level_db"], r["upper_db"], r["low_db"], r["centroid"]))
np.save(f"{H}/models/gain_direction.npy", g.astype(np.float32)); np.save(f"{H}/models/tone_direction.npy", tone.astype(np.float32))
json.dump(out, open(f"{H}/docs/assets/gain_vs_tone.json", "w"), indent=1); print("wrote models/gain_direction.npy, models/tone_direction.npy")
