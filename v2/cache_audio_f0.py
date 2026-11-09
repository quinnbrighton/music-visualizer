"""Cache 24 kHz mono fp16 audio and yin f0 (100 Hz frame rate) per recording, keyed by content hash. CPU only."""
import json, os, sys, numpy as np, soundfile as sf, librosa
from multiprocessing import Pool
from scipy.signal import resample_poly
B = os.path.expanduser("~/music-visualizer/v2"); idx = json.load(open(f"{B}/mel512_index.json"))
def work(k):
    a, f = f"{B}/audio24k/{k}.npy", f"{B}/f0/{k}.npy"
    if os.path.exists(a) and os.path.exists(f): return k, "cached"
    x, sr = sf.read(idx[k]["src"], dtype="float32", always_2d=True); x = x.mean(1)
    y = resample_poly(x, 24000, sr).astype(np.float32) if sr != 24000 else x
    f0 = librosa.yin(y, fmin=60, fmax=1100, sr=24000, frame_length=2048, hop_length=240)        # 100 Hz frames
    np.save(a, y.astype(np.float16)); np.save(f, f0.astype(np.float32)); return k, f"{len(y)/24000/60:.1f} min"
if __name__ == "__main__":
    with Pool(2, maxtasksperchild=1) as p:
        for k, s in p.imap_unordered(work, sorted(idx)): print(k[:8], s, flush=True)
    print("done")
