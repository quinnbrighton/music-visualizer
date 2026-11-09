"""Render a whole recording through the pitch-in-code model (sampled latents), chunked with context, harmonic phase aligned at the seams. env: INIT INZD ZDIM KEY OUT -> OUT_orig.wav, OUT_rec.wav (24 kHz mono, rec level-matched to orig)"""
import sys, os
sys.argv = ["x", "full", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn6.py")).read(); exec(src[: src.index("log = {")])
import soundfile as sf
PHI0 = 0.0
@torch.no_grad()
def excitation(f0, n):                                                   # same as train_harm.excitation + a start-phase offset so chunks stay phase-locked
    f = F.interpolate(f0[:, None].clamp(40, 1200), size=n, mode="linear", align_corners=False)[:, 0]
    phi = torch.remainder(torch.cumsum(2 * math.pi * f.double() / SR, 1) + PHI0, 2 * math.pi).float()
    k = torch.arange(1, K + 1, device=f0.device).float(); ok = (f[..., None] * k < SR / 2 * 0.95).float()
    return (torch.sin(phi[..., None] * k) * ok / k).sum(-1) + 0.02 * torch.randn_like(f)
def cumphase(f0, n):
    f = F.interpolate(f0[:, None].clamp(40, 1200), size=n, mode="linear", align_corners=False)[:, 0]; return torch.cumsum(2 * math.pi * f.double() / SR, 1)[0]
key, out = os.environ["KEY"], os.environ["OUT"]; model.eval(); a = A[key]; N = len(a) // 960 * 960; Tt = N // HOP; CH, CT = 20 * SR, SR
mus, lvs = [], []
with torch.no_grad():
    for c in range(0, N, CH):
        s, e = max(0, c - CT), min(N, c + CH + CT); e -= (e - s) % 960
        x = torch.from_numpy(np.asarray(a[s:e], dtype=np.float32))[None].to(dev); mu, lv = model.encode(feats(x, None), (e - s) // HOP)
        lo = (c - s) // HOP; hi = lo + min(CH, N - c) // HOP; mus.append(mu[0, :, lo:hi].cpu()); lvs.append(lv[0, :, lo:hi].cpu())
mu = torch.cat(mus, 1); lv = torch.cat(lvs, 1); torch.manual_seed(0); zf = mu + torch.randn_like(mu) * (0.5 * lv).exp(); print("encoded", tuple(mu.shape), flush=True)
CORE, CTX, XF = 3000, 100, 30; y_out = np.zeros(Tt * HOP, np.float32); tail = None; pb = None; seams = []
with torch.no_grad():
    for c0 in range(0, Tt, CORE):
        c1 = min(Tt, c0 + CORE); a0 = max(0, c0 - CTX); a1 = min(Tt, c1 + CTX + XF); T = a1 - a0
        hh = model.up(zf[:, a0:a1][None].to(dev)); f0p = model.pitch_f0(hh); Phi = cumphase(f0p, T * HOP)
        PHI0 = 0.0 if pb is None else float((pb - Phi[(c0 - a0) * HOP].item()) % (2 * math.pi))
        y = model.synth(hh, f0p, T)[0].cpu().numpy(); seg = y[(c0 - a0) * HOP: (min(Tt, c1 + XF) - a0) * HOP]
        if tail is not None:
            m = min(len(tail), len(seg)); w = np.linspace(0, 1, m, dtype=np.float32); seg[:m] = tail[:m] * (1 - w) + seg[:m] * w
        core = (c1 - c0) * HOP; y_out[c0 * HOP: c0 * HOP + min(core, len(seg))] = seg[:core]; tail = seg[core:] if len(seg) > core else None
        if c1 < Tt: pb = PHI0 + Phi[(c1 - a0) * HOP].item(); seams.append(c1 * HOP)
        print("chunk", c0, "-", c1, "phi0 %.2f" % PHI0, flush=True)
xo = np.asarray(a[:N], dtype=np.float32); ro = float(np.sqrt((xo ** 2).mean())); rr = float(np.sqrt((y_out ** 2).mean())); y_out *= ro / rr
sf.write(out + "_orig.wav", xo, SR, subtype="PCM_16"); sf.write(out + "_rec.wav", np.clip(y_out, -1, 1), SR, subtype="PCM_16")
d = np.abs(np.diff(y_out)); r = [d[b - 1] / (np.median(d[b - 300: b + 300]) + 1e-9) for b in seams]; print("seam click ratio (sample step at the seam / local median step): mean %.2f max %.2f over %d seams" % (np.mean(r), np.max(r), len(r)))
ms, ls = [], []
for q in range(10):
    s0 = int((q + 0.5) / 10 * (N - 20 * SR)) // 960 * 960; x = torch.from_numpy(xo[s0: s0 + 20 * SR])[None].to(dev); y = torch.from_numpy(y_out[s0: s0 + 20 * SR])[None].to(dev); ms.append(MSS(y, x).item()); ls.append(logmel_dist(y[0].cpu(), x[0].cpu())["lm"])
print("10 windows of 20 s: mss %.4f lm %.4f" % (np.mean(ms), np.mean(ls)), "seconds", N / SR)
