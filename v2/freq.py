"""Temporal frequency of latent dimensions: per-dim spectral centroid (Welch PSD) and slow feature analysis (SFA)."""
import json, sys, numpy as np, torch
from scipy.signal import welch
from scipy.linalg import eigh

def analyze(Z, fs, label):
    Z = Z - Z.mean(0); n, D = Z.shape
    cen = []; slow = []
    for d in range(D):
        f, p = welch(Z[:, d], fs, nperseg=min(2048, n // 4)); cen.append(float((f * p).sum() / p.sum()))
        ac = np.correlate(Z[:5000, d], Z[:5000, d], "full")[4999:]; ac = ac / ac[0]
        slow.append(float(np.argmax(ac < 0.5) / fs * 1000))     # ms until autocorrelation drops below 0.5
    cov = np.cov(Z.T); dcov = np.cov(np.diff(Z, axis=0).T) * fs ** 2
    w, V = eigh(dcov, cov + 1e-6 * np.eye(D))                   # generalized eigenproblem: smallest = slowest
    print(f"\n{label}: {D} dims at {fs:.1f} frames/s")
    print("per-dim centroid Hz:", np.round(sorted(cen), 2))
    print(f"slowest dim {min(cen):.2f} Hz, fastest {max(cen):.2f} Hz; half-decay of autocorrelation {min(slow):.0f}-{max(slow):.0f} ms")
    S = Z @ V; sc = [float((lambda f_, p_: (f_ * p_).sum() / p_.sum())(*welch(S[:, i], fs, nperseg=min(2048, n // 4)))) for i in range(D)]
    print("SFA components, slowest -> fastest, centroid Hz:", np.round(sc, 2))
    print("variance share of the 3 slowest SFA comps:", round(float(S[:, :3].var(0).sum() / S.var(0).sum()), 3))
    return cen

d = json.load(open("/home/qbrighto/music-visualizer/ae3d_bach_12d.json"))
F = np.array(d["frames"], float)[:, 4:16]
analyze(F, 1.0 / (d["frame_dur"] * d["step"]) if d["step"] else 1.0 / d["frame_dur"], f"original 12-D Bach (frame_dur {d['frame_dur']:.4f}s, step {d['step']})")
from model import AE, read
for name in sys.argv[1:]:
    ck = torch.load(f"{name}.pt"); m = AE(*ck["args"], rates=ck.get("rates")).cuda().eval(); m.load_state_dict(ck["model"])
    Z = []
    with torch.no_grad():
        for k in ck["val"][:3] + ck["test"]:
            x = ((read(k) - ck["mean"][:, None]) / ck["std"][:, None]).cuda()[None]
            Z.append(m.enc_frame(x[:, :, :12000])[0][0].T.cpu().numpy())
    Zc = np.concatenate(Z); keep = Zc.var(0) > 1e-3
    analyze(Zc[:, keep], 44100 / 512, f"{name} (per-frame latent, {int(keep.sum())} active dims)")
