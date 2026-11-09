import numpy as np, torch, librosa, soundfile as sf, time, json
from scipy.ndimage import zoom
from model import AE, read
mb = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=512); pinv = np.linalg.pinv(mb); fs = 44100 / 512
ck = torch.load("multirate_nested.pt"); m = AE(*ck["args"], rates=ck["rates"]).cuda().eval(); m.load_state_dict(ck["model"])
k = ck["test"][0]; s0, W = 20000, 650
x = ((read(k) - ck["mean"][:, None]) / ck["std"][:, None]).cuda()[None, :, s0:s0 + W]
with torch.no_grad(): rec = m.decode(m.enc_frame(x)[0])[0].cpu()
Lrec = (rec * ck["std"][:, None] + ck["mean"][:, None]).numpy()
def jit(y):
    L_ = np.log(mb @ np.abs(librosa.stft(y, n_fft=4096, hop_length=512)) + 1e-5)
    P = np.abs(np.fft.rfft(L_ - L_.mean(1, keepdims=True), axis=1)) ** 2; f = np.fft.rfftfreq(L_.shape[1], 1 / fs)
    return round(float(np.abs(np.diff(L_, axis=1)).mean()), 4), round(float(P[:, f > 10].sum() / P.sum()), 4)
print("input spectrogram (model recon) jitter:", jit(librosa.griffinlim(np.maximum(pinv @ np.exp(Lrec), 0), n_iter=1, hop_length=512, n_fft=4096)) if False else "(0.030, 0.0004) from before")
res = {}
for up in (1, 2, 4, 8):
    t = time.time(); Lu = zoom(Lrec, (1, up), order=1) if up > 1 else Lrec            # interpolate log-mel in time
    S = np.maximum(pinv @ np.exp(Lu), 0); hop = 512 // up
    y = librosa.griffinlim(S, n_iter=100, hop_length=hop, n_fft=4096, momentum=0.99, random_state=0)
    res[up] = {"hop": hop, "jitter": jit(y), "sec": round(time.time() - t)}; print(up, res[up], flush=True)
    sf.write(f"gl_overlap_x{up}.wav", y / np.abs(y).max() * 0.9, 44100)
json.dump(res, open("gl_overlap.json", "w"), indent=1)
