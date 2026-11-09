"""Brightness (log spectral centroid) and noisiness (spectral flatness) per 25 Hz token frame -> TOK_DIR/controls2/{k}.npy (T,2).
usage: python extra_controls.py AUDIO_DIR TOK_DIR"""
import os, sys, numpy as np
AD, TOK = sys.argv[1], sys.argv[2]; os.makedirs(f"{TOK}/controls2", exist_ok=True); SR, H, NF = 24000, 960, 2048
fr = np.fft.rfftfreq(NF, 1 / SR); band = (fr > 60) & (fr < 8000); win = np.hanning(NF).astype(np.float32)
for fn in sorted(os.listdir(f"{TOK}/tokens")):
    T = len(np.load(f"{TOK}/tokens/{fn}")); a = np.load(f"{AD}/{fn}", mmap_mode="r"); out = np.zeros((T, 2), np.float32)
    x = np.pad(np.asarray(a, np.float32), (NF // 2 - H // 2, NF))
    for s in range(0, T, 2000):
        idx = np.arange(s, min(T, s + 2000))[:, None] * H + np.arange(NF)[None]; P = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
        Pb = P[:, band] + 1e-10; cen = (Pb * fr[band]).sum(1) / Pb.sum(1); flat = np.exp(np.log(Pb).mean(1)) / Pb.mean(1)
        out[s: s + len(idx), 0] = np.log2(cen / 500.0); out[s: s + len(idx), 1] = np.log10(flat)
    np.save(f"{TOK}/controls2/{fn}", out); print(fn, T, out.mean(0).round(2), out.std(0).round(2), flush=True)
