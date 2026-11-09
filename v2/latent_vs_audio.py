"""Is the latent changing more than the audio? (1) share of variance above 5 / 10 Hz: latent channels vs log-mel bands of the same recording; (2) decode the 15 s clip after low-passing the latent means in time. env: INIT INZD ZDIM KEY MU OUT"""
import sys, os
sys.argv = ["x", "lva", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn6.py")).read(); exec(src[: src.index("log = {")])
import soundfile as sf; from scipy.signal import welch, butter, filtfilt
key, out = os.environ["KEY"], os.environ["OUT"]; Z = np.load(os.environ["MU"])                                              # (T,16) latent means at 100 Hz
a = A[key]; T = len(Z); L = []
with torch.no_grad():
    for c in range(0, T, 6000):
        n = min(6000, T - c) * HOP; x = torch.from_numpy(np.asarray(a[c * HOP: c * HOP + n], dtype=np.float32))[None].to(dev); L.append(torch.log(_ml(x) + 1e-5)[0, :, : n // HOP].cpu())
M = torch.cat(L, 1).numpy().T[:T]; print("mel", M.shape, "latent", Z.shape)
def hf(X):
    X = (X - X.mean(0)) / (X.std(0) + 1e-6); f, p = welch(X, fs=100, nperseg=2048, axis=0); p = p / p.sum(0); return [round(float(p[f > c].sum(0).mean()), 4) for c in (2, 5, 10, 20)]
print("share of variance above 2/5/10/20 Hz  latent channels:", hf(Z), "| log-mel bands:", hf(M), "| mel 0-60 (the notes' region):", hf(M[:, :60]))
def step(X): X = (X - X.mean(0)) / (X.std(0) + 1e-6); return np.sqrt((np.diff(X, axis=0) ** 2).mean(1))
sz, sm = step(Z), step(M); print("typical frame-to-frame step (in std units): latent %.3f  mel %.3f | correlation of the two step sizes %.3f" % (np.median(sz), np.median(sm), np.corrcoef(sz, sm)[0, 1]))
# decode test on the 15 s clip at 0:10
s0 = 10 * SR; n = 15 * SR; Tn = n // HOP; x = torch.from_numpy(np.asarray(a[s0: s0 + n], dtype=np.float32))[None].to(dev); mu0 = torch.from_numpy(Z[1000: 1000 + Tn].T.copy())[None].float().to(dev); r = x[0].pow(2).mean().sqrt().item(); model.eval()
for cut in (None, 20, 10, 5, 3, 1.5):
    z = mu0 if cut is None else torch.from_numpy(filtfilt(*butter(4, cut / 50), mu0.cpu().numpy(), axis=2).copy()).float().to(dev)
    with torch.no_grad(): torch.manual_seed(0); y = model.decode(z, None, Tn)[0]
    g = y * r / y.pow(2).mean().sqrt(); print("lowpass", cut, "Hz  mss %.4f  logmel %.4f" % (MSS(y[None], x).item(), logmel_dist(y.cpu(), x[0].cpu())["lm"])); sf.write(f"{out}_lp{cut}.wav", g.cpu().numpy(), SR)
