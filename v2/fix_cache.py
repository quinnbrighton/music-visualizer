"""Finish missing cache entries with chunked yin (low memory). Appends 'done' to cache2.log when complete."""
import json, os, numpy as np, soundfile as sf, librosa
from scipy.signal import resample_poly
B = os.path.expanduser("~/music-visualizer/v2"); idx = json.load(open(f"{B}/mel512_index.json"))
for k in sorted(idx):
    a, f = f"{B}/audio24k/{k}.npy", f"{B}/f0/{k}.npy"
    ok = os.path.exists(a) and os.path.exists(f)
    if ok:
        try: np.load(a, mmap_mode="r"); np.load(f)
        except Exception: ok = False
    if ok: continue
    x, sr = sf.read(idx[k]["src"], dtype="float32", always_2d=True); x = x.mean(1); y = resample_poly(x, 24000, sr).astype(np.float32) if sr != 24000 else x; del x
    CH = 24000 * 120; out = []
    for s in range(0, len(y), CH):
        seg = y[max(0, s - 2400): s + CH + 2400]; off = (s - max(0, s - 2400)) // 240 if s else 0
        fr = librosa.yin(seg, fmin=60, fmax=1100, sr=24000, frame_length=2048, hop_length=240)
        out.append(fr[off: off + CH // 240])
    f0 = np.concatenate(out)[: 1 + len(y) // 240]
    np.save(a, y.astype(np.float16)); np.save(f, f0.astype(np.float32)); print("fixed", k[:8], len(y) / 24000 / 60, "min", len(f0), flush=True)
print("verified all"); open(f"{B}/cache2.log", "a").write("done\n")
