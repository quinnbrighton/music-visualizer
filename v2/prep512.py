"""Cache 512-bin log-magnitude mel (44.1 kHz, n_fft 4096, hop 512: same front end as the original model), keyed by content hash."""
import glob, hashlib, json, os
import numpy as np, torch, librosa, soundfile as sf, torchaudio
B = os.path.expanduser("~/music-visualizer"); OUT = f"{B}/v2/mel512"
mb = torch.from_numpy(librosa.filters.mel(sr=44100, n_fft=4096, n_mels=512)).float().cuda()
win = torch.hann_window(4096).cuda()
def feats(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    if sr != 44100: x = torchaudio.functional.resample(x, sr, 44100)
    S = torch.stft(x.cuda(), 4096, 512, window=win, return_complex=True).abs()
    return torch.log(mb @ S + 1e-5).cpu().numpy().astype(np.float16)
srcs = sorted(glob.glob(f"{B}/cleaned/cello_dataset/*.wav")) + sorted(glob.glob(f"{B}/cello_dataset/*.wav"))
srcs += [f"{B}/downloads/Yo-Yo Ma Bach Cello Suite No.1 in G Major.wav", f"{B}/oos_test/Bach - Cello Suite No. 2 [BWV 1008] - Mischa Maisky.wav"]
index, seen = {}, {}
for f in srcs:
    h = hashlib.md5(open(f, "rb").read(1 << 22)).hexdigest()[:12] + f"_{os.path.getsize(f)}"
    if h in seen: continue
    seen[h] = f; o = f"{OUT}/{h}.npy"
    if not os.path.exists(o): np.save(o, feats(f))
    index[h] = {"src": f, "frames": int(np.load(o, mmap_mode='r').shape[1])}
json.dump(index, open(f"{B}/v2/mel512_index.json", "w"), indent=1)
print(len(index), "recordings,", sum(v["frames"] for v in index.values()) * 512 / 44100 / 3600, "hours")
