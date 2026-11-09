"""Quick Griffin-Lim fixes. Variants differ in (mel->linear inverse, iterations). Scored against the ORIGINAL audio's STFT."""
import time, json, numpy as np, torch, librosa, soundfile as sf
from model import AE, read
sr0 = 44100; N, H = 4096, 512
ck = torch.load("multirate_nested.pt"); m = AE(*ck["args"], rates=ck["rates"]).cuda().eval(); m.load_state_dict(ck["model"])
mean, std = ck["mean"], ck["std"]; k = ck["test"][0]; s0, W = 20000, 650          # 7.5 s to keep nnls fast
x = ((read(k) - mean[:, None]) / std[:, None]).cuda()[None, :, s0:s0 + W]
with torch.no_grad(): rec = m.decode(m.enc_frame(x)[0])[0].cpu()
true_L = x[0].cpu() * std[:, None] + mean[:, None]; rec_L = rec * std[:, None] + mean[:, None]
src, sr = sf.read("/home/qbrighto/music-visualizer/oos_test/Bach - Cello Suite No. 2 [BWV 1008] - Mischa Maisky.wav", dtype="float32", always_2d=True)
t0 = s0 * H / sr0; orig = librosa.resample(src.mean(1)[int(t0 * sr): int((t0 + W * H / sr0) * sr)], orig_sr=sr, target_sr=sr0)
S_ref = np.abs(librosa.stft(orig, n_fft=N, hop_length=H))
mb = librosa.filters.mel(sr=sr0, n_fft=N, n_mels=512); pinv = np.linalg.pinv(mb)
def inv(M, how):
    if how == "pinv": return np.maximum(pinv @ M, 0)
    return librosa.feature.inverse.mel_to_stft(M, sr=sr0, n_fft=N, power=1.0)      # librosa: NNLS-based
def score(y, S_hat):
    S = np.abs(librosa.stft(y, n_fft=N, hop_length=H)); n = min(S.shape[1], S_ref.shape[1], S_hat.shape[1])
    lsd = float(np.sqrt(((np.log(S[:, :n] + 1e-3) - np.log(S_ref[:, :n] + 1e-3)) ** 2).mean()))
    cons = float(np.linalg.norm(S[:, :n] - S_hat[:, :n]) / np.linalg.norm(S_hat[:, :n]))
    return {"lsd_vs_original": round(lsd, 3), "inconsistency": round(cons, 3)}
res = {}
for src_name, L in (("true_mel", true_L), ("model_recon", rec_L)):
    M = np.exp(L.numpy())
    for how, it in (("pinv", 100), ("nnls", 100), ("nnls", 400)):
        t = time.time(); S_hat = inv(M, how)
        y = librosa.griffinlim(S_hat, n_iter=it, hop_length=H, n_fft=N, momentum=0.99, random_state=0)
        key = f"{src_name}|{how}|{it}it"; res[key] = {**score(y, S_hat), "sec": round(time.time() - t)}
        sf.write(f"gl_{src_name}_{how}_{it}.wav", y / np.abs(y).max() * 0.9, sr0); print(key, res[key], flush=True)
json.dump(res, open("gl_quickfix.json", "w"), indent=1)
