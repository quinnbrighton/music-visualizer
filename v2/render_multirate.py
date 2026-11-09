import numpy as np, torch, librosa, soundfile as sf
from model import AE, read
ck = torch.load("multirate_24.pt"); m = AE(*ck["args"], rates=ck["rates"]).cuda().eval(); m.load_state_dict(ck["model"])
mean, std = ck["mean"], ck["std"]; k = ck["test"][0]; s0, W = 20000, 1300
x = ((read(k) - mean[:, None]) / std[:, None]).cuda()[None, :, s0:s0 + W]
with torch.no_grad():
    mu = m.enc_frame(x)[0]; full = m.decode(mu)[0]
    nos = mu.clone(); nos[:, 16:] = 0; noslow = m.decode(nos)[0]          # slow groups (dims 16-23) removed
    oso = torch.zeros_like(mu); oso[:, 16:] = mu[:, 16:]; onlyslow = m.decode(oso)[0]   # ONLY slow groups
    nofast = mu.clone(); nofast[:, :6] = 0; nofast = m.decode(nofast)[0]   # fastest group removed
mb = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=512); pinv = np.linalg.pinv(mb)
def wav(y):
    L = (y.cpu() * std[:, None] + mean[:, None]).numpy(); return librosa.griffinlim(np.maximum(pinv @ np.exp(L), 0), n_iter=100, hop_length=512, n_fft=4096)
out = {"1_original": None, "2_true_spectrogram_griffinlim": x[0], "3_multirate_reconstruction": full, "4_slow_groups_removed": noslow, "5_only_slow_groups": onlyslow, "6_fastest_group_removed": nofast}
src, sr = sf.read("/home/qbrighto/music-visualizer/oos_test/Bach - Cello Suite No. 2 [BWV 1008] - Mischa Maisky.wav", dtype="float32", always_2d=True)
t0 = s0 * 512 / 44100; orig = src.mean(1)[int(t0 * sr): int((t0 + W * 512 / 44100) * sr)]; orig = librosa.resample(orig, orig_sr=sr, target_sr=44100)
for n, y in out.items():
    w = orig if y is None else wav(y); sf.write(f"mr_{n}.wav", w / max(1e-6, np.abs(w).max()) * 0.9, 44100)
print("ok", [n for n in out])
