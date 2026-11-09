"""Reconstruct a window of any dataset recording with the pitch-in-code model. env: INIT, KEY, START (s), DUR (s), OUT (prefix); writes OUT_orig.wav, OUT_samp.wav, OUT_mean.wav (level-matched to the original)"""
import sys, os
sys.argv = ["x", "recon", "0", "1"]; src = open(os.path.expanduser("~/music-visualizer/v2/train_bn6.py")).read(); exec(src[: src.index("log = {")])
import soundfile as sf
key, s0, dur, out = os.environ["KEY"], int(float(os.environ.get("START", 10)) * SR) // 960 * 960, float(os.environ.get("DUR", 15)), os.environ["OUT"]
model.eval(); n = int(dur * SR) // 960 * 960; T = n // HOP; x = torch.from_numpy(np.asarray(A[key][s0:s0 + n], dtype=np.float32))[None].to(dev)
with torch.no_grad():
    mu, lv = model.encode(feats(x, None), T); kd = kl_vec(mu, lv, 0.0); print("rate bits/s", round(kd.sum().item() * 100 / math.log(2), 1))
    r = x[0].pow(2).mean().sqrt().item(); sf.write(f"{out}_orig.wav", x[0].cpu().numpy(), SR)
    for tag, z in (("samp", mu + torch.randn_like(mu) * (0.5 * lv).exp()), ("mean", mu)):
        torch.manual_seed(0); y = model.decode(z, None, T)[0]; y = y * r / y.pow(2).mean().sqrt(); sf.write(f"{out}_{tag}.wav", y.cpu().numpy(), SR)
        print(tag, "mss", round(MSS(y[None], x).item(), 4), "lm", round(logmel_dist(y.cpu(), x[0].cpu())["lm"], 4))
