"""Cache Vocos-format log-mel features (24 kHz, n_fft 1024, hop 256, 100 mels) for every unique recording."""
import glob, hashlib, os, sys
import numpy as np, torch, torchaudio, soundfile as sf
from vocos.feature_extractors import MelSpectrogramFeatures

B = os.path.expanduser("~/music-visualizer")
OUT = f"{B}/v2/feats"
fe = MelSpectrogramFeatures(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=100).cuda()

def load24(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = torch.from_numpy(x.mean(1))
    return torchaudio.functional.resample(x, sr, 24000)

def mel(wav):
    with torch.no_grad():
        return fe(wav[None].cuda())[0].cpu().numpy().astype(np.float16)

files, seen = {}, set()
for f in sorted(glob.glob(f"{B}/cleaned/cello_dataset/*.wav")) + sorted(glob.glob(f"{B}/cello_dataset/*.wav")):
    h = hashlib.md5(open(f, "rb").read(1 << 22)).hexdigest() + str(os.path.getsize(f))
    if h not in seen:
        seen.add(h); files[os.path.basename(f)[:-4]] = f
files["yoyoma_bach1"] = f"{B}/downloads/Yo-Yo Ma Bach Cello Suite No.1 in G Major.wav"
files["maisky_bach2"] = f"{B}/oos_test/Bach - Cello Suite No. 2 [BWV 1008] - Mischa Maisky.wav"
for k, f in files.items():
    o = f"{OUT}/{k}.npy"
    if os.path.exists(o): continue
    w = load24(f)
    np.save(o, mel(w))
    if k in ("yoyoma_bach1", "maisky_bach2"):
        torchaudio.save(f"{B}/v2/{k}_24k.wav", w[None], 24000)
    print(k, len(w) / 24000 / 60, "min", flush=True)
