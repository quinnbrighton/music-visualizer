"""What does each latent group encode? Ridge probes trained on TRAIN recordings, scored on held-out val+Maisky (out-of-sample R^2)."""
import sys, numpy as np, torch, torchaudio, librosa
from sklearn.linear_model import Ridge
from voc_model import *
B = "/home/qbrighto/music-visualizer/v2"; name = sys.argv[1]; dev = "cuda"
CK = torch.load(f"{B}/{name}.pt", map_location=dev); MEL = CK["args"].get("MELENC", 0)
m = VocModel(extra=2 * CK["args"].get("EXPL", 0), mel_in=bool(MEL)).to(dev).eval(); m.load_state_dict(CK["model"])
mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
rng = np.random.RandomState(0); tr = [k for k in train if np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r").shape[0] > 50 * SR]; tr = list(rng.choice(tr, 10, replace=False))
def frames(k, off_s):
    a = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); n = 20 * SR; s0 = 960 * (min(len(a) // 3, off_s * SR) // 960); T = n // HOP
    x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev)
    with torch.no_grad():
        f = torch.log(mel(x) + 1e-5)[..., :T] if False else ((torch.log(mel(x) + 1e-5) + 5) / 3)[..., :T] if MEL else torch.from_numpy(np.asarray(np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r")[s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev)
        mu = m.encode(f, T)[0][0].T.cpu().numpy()                                    # (T,24)
    xn = x[0].cpu().numpy(); sl = slice(s0 // 240, s0 // 240 + T)
    per = np.load(f"{B}/per/{k}.npy").astype(np.float32)[sl]; f0 = np.load(f"{B}/f0crepe/{k}.npy")[sl]
    ld = loudness(x)[0].cpu().numpy(); cen = librosa.feature.spectral_centroid(y=xn, sr=SR, n_fft=1024, hop_length=HOP)[0][:T]
    n_ = min(len(mu), len(per), len(f0), len(ld), len(cen))
    return mu[:n_], {"loudness": ld[:n_], "log2 f0 (voiced)": np.log2(f0[:n_]), "log brightness (centroid)": np.log(cen[:n_] + 1), "periodicity (harmonic vs noisy)": per[:n_]}, per[:n_] > 0.5
def collect(keys, off):
    M, Y, V = [], {}, []
    for k in keys:
        mu, y, v = frames(k, off); M.append(mu); V.append(v)
        for t in y: Y.setdefault(t, []).append(y[t])
    return np.concatenate(M), {t: np.concatenate(v) for t, v in Y.items()}, np.concatenate(V)
Mtr, Ytr, Vtr = collect(tr, 30); Mte, Yte, Vte = collect(val + test, 30)
mu_, sd_ = Mtr.mean(0), Mtr.std(0) + 1e-6; Mtr, Mte = (Mtr - mu_) / sd_, (Mte - mu_) / sd_
groups = {"all 24 dims": slice(0, 24)}; i = 0
for g, (n, w) in enumerate(RATES): groups[f"group {g+1} ({w/100:.2f} s window, dims {i}-{i+n-1})"] = slice(i, i + n); i += n
print(f"probe = ridge regression from latent mean to the quantity, fit on {len(tr)} TRAIN recordings, scored on {len(val)+len(test)} held-out recordings (out-of-sample R^2; negative = worse than constant)")
print(f"{'quantity':36s}" + "".join(f"{g[:22]:>24s}" for g in groups))
for t in Ytr:
    msk_tr, msk_te = (Vtr, Vte) if "f0" in t else (np.ones_like(Vtr), np.ones_like(Vte)); row = f"{t:36s}"
    for g, sl in groups.items():
        r = Ridge(alpha=10.0).fit(Mtr[msk_tr][:, sl], Ytr[t][msk_tr]); p = r.predict(Mte[msk_te][:, sl]); y = Yte[t][msk_te]
        row += f"{1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum():24.2f}"
    print(row)
