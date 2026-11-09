"""Linear probes: how well do the first n RVQ stages encode controllable properties? Properties at 25 frames/s from the audio and controls;
each stage's token is replaced by the training-set mean of the property for that token, then a linear fit over stages (fit on train, R^2 on val).
usage: python probe_tokens.py TOK_DIR CELLO_DIR"""
import os, sys, numpy as np, torch
TOK, CD = sys.argv[1], sys.argv[2]; sp = torch.load(f"{CD}/multirate_nested.pt", map_location="cpu"); SR, FR = 24000, 960
def props(k):
    t = np.load(f"{TOK}/tokens/{k}.npy").astype(np.int64); c = np.load(f"{TOK}/controls/{k}.npy"); T = len(t)
    a = np.asarray(np.load(f"{CD}/audio24k/{k}.npy", mmap_mode="r")[: T * FR], np.float32).reshape(T, FR) * np.hanning(FR)[None]
    S = np.abs(np.fft.rfft(a, axis=1)) + 1e-8; f = np.fft.rfftfreq(FR, 1 / SR); P = S ** 2
    cent = np.log2((P * f).sum(1) / P.sum(1) + 1); flat = np.exp(np.log(P).mean(1)) / P.mean(1)
    flux = np.r_[0, np.maximum(np.diff(np.log(S), axis=0), 0).sum(1)]
    v = c[:, 1] > 0.5; cents = 1200 * np.log2(np.clip(c[:, 0], 40, 1200) / 110)
    hp = cents - np.convolve(cents, np.ones(9) / 9, "same"); vib = np.sqrt(np.convolve(hp ** 2, np.ones(15) / 15, "same"))          # pitch wobble around a 0.36 s mean, rms over 0.6 s
    X = {"pitch (cents, voiced)": np.where(v, cents, np.nan), "loudness dB": c[:, 2], "periodicity": c[:, 1], "brightness (log centroid)": cent, "noisiness (flatness)": np.log10(flat), "vibrato depth (cents, voiced)": np.where(v, vib, np.nan), "onset strength (flux)": flux}
    return t, X
tr = [props(k) for k in sp["train"] if os.path.exists(f"{TOK}/tokens/{k}.npy")]; va = [props(k) for k in sp["val"]]
NQ = tr[0][0].shape[1]; K = int(max(t.max() for t, _ in tr)) + 1; sets = [n for n in (1, 2, 4, 8, 12) if n <= NQ]
print("stages", NQ, "codebook", K, "| R^2 on validation songs for the first n stages")
print("%-30s" % "property" + "".join("%9s" % f"n={n}" for n in sets))
for name in tr[0][1]:
    Ttr = np.concatenate([t for t, _ in tr]); ytr = np.concatenate([x[name] for _, x in tr]); Tva = np.concatenate([t for t, _ in va]); yva = np.concatenate([x[name] for _, x in va])
    mtr, mva = ~np.isnan(ytr), ~np.isnan(yva); Ttr, ytr, Tva, yva = Ttr[mtr], ytr[mtr], Tva[mva], yva[mva]; row = []
    means = [np.bincount(Ttr[:, q], weights=ytr, minlength=K) / np.maximum(np.bincount(Ttr[:, q], minlength=K), 1) + (np.bincount(Ttr[:, q], minlength=K) == 0) * ytr.mean() for q in range(NQ)]
    for n in sets:
        Ftr = np.stack([means[q][Ttr[:, q]] for q in range(n)] + [np.ones(len(Ttr))], 1); Fva = np.stack([means[q][Tva[:, q]] for q in range(n)] + [np.ones(len(Tva))], 1)
        w = np.linalg.lstsq(Ftr, ytr, rcond=None)[0]; p = Fva @ w; row.append(1 - ((yva - p) ** 2).mean() / yva.var())
    print("%-30s" % name + "".join("%9.3f" % r for r in row))
