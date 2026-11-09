"""Decode the Maisky test clip from latent means with (a) the code low-passed in time, (b) cycle-level pitch jitter in the excitation, (c) jitter plus amplitude shimmer on the output. No retraining. env: INIT INZD ZDIM RATESSPEC OUT"""
import sys, os
sys.argv = ["x", "cyc", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn10.py")).read(); exec(src[: src.index("log = {")])
import soundfile as sf, torchaudio.functional as AF; from scipy.signal import butter, filtfilt
JIT = 0.0
def bandnoise(shape, seed, dev_):                                        # unit-variance noise varying about once per vibration cycle (low-passed at 300 Hz)
    g = torch.Generator(device=dev_).manual_seed(seed); w = torch.randn(shape, device=dev_, generator=g); w = AF.lowpass_biquad(w, SR, 300.0); return w / w.std()
@torch.no_grad()
def excitation(f0, n):                                                   # train_harm.excitation, plus optional cycle-level jitter of the instantaneous frequency
    f = F.interpolate(f0[:, None].clamp(40, 1200), size=n, mode="linear", align_corners=False)[:, 0]
    if JIT: f = f * (1 + JIT * bandnoise(f.shape, 1, f.device))
    phi = torch.remainder(torch.cumsum(2 * math.pi * f.double() / SR, 1), 2 * math.pi).float()
    k = torch.arange(1, K + 1, device=f0.device).float(); ok = (f[..., None] * k < SR / 2 * 0.95).float()
    return (torch.sin(phi[..., None] * k) * ok / k).sum(-1) + 0.02 * torch.randn_like(f)
key = test[0]; s0 = 5572800; n = 15 * SR; T = n // HOP; x = torch.from_numpy(np.asarray(A[key][s0:s0 + n], dtype=np.float32))[None].to(dev); r = x.pow(2).mean().sqrt().item(); model.eval(); out = os.environ["OUT"]
with torch.no_grad():
    mu, lv = model.encode(feats(x, None), T)
    for tag, jit, shim, lp in (("mean_ref", 0, 0, None), ("mean_lp8", 0, 0, 8), ("mean_jit0.3", 0.003, 0, None), ("mean_jit1", 0.01, 0, None), ("mean_jit1_shim5", 0.01, 0.05, None)):
        JIT = jit; z = mu if lp is None else torch.from_numpy(filtfilt(*butter(4, lp / 50), mu.cpu().numpy(), axis=2).copy()).float().to(dev)
        torch.manual_seed(0); y = model.decode(z, None, T)[0]
        if shim: y = y * (1 + shim * bandnoise(y.shape, 2, y.device))
        print(tag, "mss %.4f lm %.4f" % (MSS(y[None], x).item(), logmel_dist(y.cpu(), x[0].cpu())["lm"])); sf.write(f"{out}_{tag}.wav", (y * r / y.pow(2).mean().sqrt()).cpu().numpy(), SR)
