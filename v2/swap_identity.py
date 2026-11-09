import json, sys, numpy as np, torch, librosa, soundfile as sf
from model import AE, read
name = sys.argv[1]; ck = torch.load(f"{name}.pt"); m = AE(*ck["args"]).cuda().eval(); m.load_state_dict(ck["model"])
mean, std = ck["mean"], ck["std"]; keys = ck["val"] + ck["test"]
X = {k: ((read(k) - mean[:, None]) / std[:, None]).cuda() for k in keys}
LG = 860
# ---- identity test: clip-level leave-one-out nearest neighbour among HELD-OUT recordings
codes, fmean, lab = [], [], []
with torch.no_grad():
    for ci, k in enumerate(keys):
        T = X[k].shape[1]; starts = np.linspace(0, T - LG, min(8, T // LG)).astype(int)
        for s in starts:
            x = X[k][None, :, s:s + LG]
            codes.append(m.global_from_feat(m.enc_global_feat(x))[0][0].cpu().numpy()); fmean.append(m.enc_frame(x)[0][0].mean(1).cpu().numpy()); lab.append(ci)
lab = np.array(lab)
def knn_acc(Z):
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6); Z /= np.linalg.norm(Z, axis=1, keepdims=True); S = Z @ Z.T; np.fill_diagonal(S, -9)
    return float((lab[S.argmax(1)] == lab).mean())
chance = float(np.mean([(lab == c).mean() for c in lab]))   # rough majority-weighted chance
res = {"n_clips": int(len(lab)), "n_recordings": len(keys), "global_code_knn_acc": knn_acc(np.array(codes)), "per_frame_mean_knn_acc": knn_acc(np.array(fmean)),
       "chance_approx": chance}
print(json.dumps(res, indent=1))
# ---- swap test
A = [k for k in ck["test"]][0]
prof = {k: X[k].mean(1) for k in keys}
Bk = max([k for k in ck["val"]], key=lambda k: float((prof[k] - prof[A]).norm()))
def g_of(k):
    with torch.no_grad():
        T = X[k].shape[1]; f = torch.stack([m.enc_global_feat(X[k][None, :, i:i + LG])[0] for i in range(0, T - LG, LG)]).mean(0, keepdim=True)
        return m.global_from_feat(f)[0]
gA, gB = g_of(A), g_of(Bk); s0 = 20000; W = 1300
with torch.no_grad():
    xa = X[A][None, :, s0:s0 + W]; mu = m.enc_frame(xa)[0]
    yAA, yAB = m.decode(mu, gA)[0], m.decode(mu, gB)[0]
    # also: zero-effect control (global of a random other held-out recording)
d = lambda y, k: float(((y.mean(1) - prof[k]) * std.cuda()).abs().mean())
res2 = {"A": A, "B": Bk, "dist_mean_spectrum_to_A": {"A-notes+A-global": d(yAA, A), "A-notes+B-global": d(yAB, A)},
        "dist_mean_spectrum_to_B": {"A-notes+A-global": d(yAA, Bk), "A-notes+B-global": d(yAB, Bk)},
        "output_change_swap_vs_own": float(((yAB - yAA) * std.cuda()[:, None]).abs().mean()), "A_vs_B_mean_spectrum_gap": d(prof[A][:, None].expand(-1, 1)[:, 0], Bk) if False else float(((prof[A] - prof[Bk]) * std.cuda()).abs().mean())}
print(json.dumps(res2, indent=1)); json.dump({"identity": res, "swap": res2}, open(f"{name}_tests.json", "w"), indent=1)
mb = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=512); pinv = np.linalg.pinv(mb)
def wav(y, k):
    L = (y.cpu() * std[:, None] + mean[:, None]).numpy(); S = np.maximum(pinv @ np.exp(L), 0)
    return librosa.griffinlim(S, n_iter=100, hop_length=512, n_fft=4096)
for tag, y in (("own_global", yAA), ("swapped_global", yAB)): sf.write(f"swap_{tag}.wav", wav(y, A), 44100)
sf.write("swap_B_source.wav", wav(X[Bk][:, s0 % (X[Bk].shape[1] - W):][:, :W].cpu() if False else X[Bk][:, :W].cpu(), Bk), 44100)
