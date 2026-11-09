"""Tokenize every recording with a trained RVQ checkpoint: 8 token streams at 25 frames/s, plus control curves (pitch Hz, periodicity,
loudness dB) at the same rate; then token statistics and a round-trip check. env: CELLO_DIR, RVQ_RESUME (checkpoint), TOK_DIR (output)"""
import os, sys
sys.argv = ["x", "tok", "1"]
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import soundfile as sf
OUTT = os.environ["TOK_DIR"]; [os.makedirs(f"{OUTT}/{d}", exist_ok=True) for d in ("tokens", "controls")]; model.eval(); DS_ = HOP * DSR; keys = train + val + test
with torch.no_grad():
    for k in keys:
        a = A[k]; N = len(a) // DS_ * DS_; CH, CT = 20 * SR, SR; parts = []
        for c in range(0, N, CH):
            s, e = max(0, c - CT), min(N, c + CH + CT); x = torch.from_numpy(np.asarray(a[s:e], dtype=np.float32))[None].to(dev)
            _, cd, _ = quant(to25(feats(x, None), (e - s) // HOP), NQ); lo = (c - s) // DS_; parts.append(cd[0, lo: lo + min(CH, N - c) // DS_].cpu())
        C = torch.cat(parts).numpy().astype(np.int16); T25 = len(C)
        f0 = np.asarray(F0[k][: T25 * DSR], np.float32); f0 = np.pad(f0, (0, T25 * DSR - len(f0)), mode="edge").reshape(T25, DSR); per = np.asarray(PER[k][: T25 * DSR], np.float32); per = np.pad(per, (0, T25 * DSR - len(per)), mode="edge").reshape(T25, DSR)
        aa = np.asarray(a[: T25 * DS_], np.float32).reshape(T25, DS_); loud = 20 * np.log10(np.sqrt((aa ** 2).mean(1)) + 1e-5)
        np.save(f"{OUTT}/tokens/{k}.npy", C); np.save(f"{OUTT}/controls/{k}.npy", np.stack([np.median(f0, 1), per.mean(1), loud], 1).astype(np.float32))
        print(k, "frames", T25, flush=True)
# statistics: real information per token
tr = np.concatenate([np.load(f"{OUTT}/tokens/{k}.npy") for k in train]).astype(np.int64); va = np.concatenate([np.load(f"{OUTT}/tokens/{k}.npy") for k in val]).astype(np.int64)
def H(p): p = p[p > 0]; return float(-(p * np.log2(p)).sum())
print("frames: train %d (%.1f h), val %d" % (len(tr), len(tr) / 25 / 3600, len(va)))
tot_m = tot_b = 0
for i in range(NQ):
    c = np.bincount(tr[:, i], minlength=KC) + 0.5; pm = c / c.sum(); ce_m = float(-np.log2(pm[va[:, i]]).mean())
    big = np.full((KC, KC), 0.5); np.add.at(big, (tr[:-1, i], tr[1:, i]), 1); big /= big.sum(1, keepdims=True); ce_b = float(-np.log2(big[va[:-1, i], va[1:, i]]).mean())
    tot_m += ce_m; tot_b += ce_b; print("stage %d: used %4d/1024, train entropy %.2f bits, val bits/token: unigram %.2f, given previous token %.2f" % (i + 1, (c > 0.5).sum(), H(pm), ce_m, ce_b))
print("bits/s at %d stages: flat %d | unigram coding %.0f | previous-token coding %.0f" % (NQ, 25 * NQ * int(np.log2(KC)), 25 * tot_m, 25 * tot_b))
# round trip: decode the Maisky window from saved tokens
k = test[0]; s0 = 5572800; n_ = 15 * SR; C = torch.from_numpy(np.load(f"{OUTT}/tokens/{k}.npy").astype(np.int64)).to(dev); f0_ = s0 // DS_; ct = C[f0_: f0_ + n_ // DS_]
with torch.no_grad():
    zq = sum(rvq.cb[i][ct[:, i]] for i in range(NQ)).t()[None]; T = n_ // HOP; hh = dec_feats(zq, T); y = model.synth(hh, model.pitch_f0(hh), T)[0].cpu().numpy()
x = np.asarray(A[k][s0:s0 + n_], np.float32); y = y * np.sqrt((x ** 2).mean() / (y ** 2).mean()); sf.write(f"{OUTT}/roundtrip_maisky_q8.wav", y, SR)
print("round trip written; spectral loss vs original %.4f" % MSS(torch.from_numpy(y)[None].to(dev), torch.from_numpy(x)[None].to(dev)).item())
