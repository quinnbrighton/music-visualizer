"""Decode z = mu + tau * std * eps for tau in 0..1: spectral loss on held-out audio, plus the share of energy in the between-harmonic 'noise' floor and the fast-modulation table."""
import sys, numpy as np, torch, torchaudio, soundfile as sf
from voc_model import *
from metrics import logmel_dist
B = "/home/qbrighto/music-visualizer/v2"; name = sys.argv[1]; dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); val, test = ck["val"], ck["test"]
CK = torch.load(f"{B}/{name}.pt", map_location=dev); EXPL = CK["args"].get("EXPL", 0); MELENC = CK["args"].get("MELENC", 0); F0DIR = CK["args"].get("F0DIR", "f0clean")
m = VocModel(extra=2 * EXPL, mel_in=bool(MELENC)).to(dev).eval();
_mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
def logmel(x): return ((torch.log(_mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP]
m.load_state_dict(CK["model"])
def ctrl(f0, loud): return torch.stack([torch.log2(f0.clamp(40, 1200) / 110), loud], 1)
torch.manual_seed(0)
mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0)
def modtab(y, ref):
    E = mel(y)[:, :1500]; R = mel(ref)[:, :1500]; keep = R.mean(1) > 0.05 * R.mean(1).max(); e = E[keep]; e = e / (e.mean(1, keepdim=True) + 1e-9) - 1
    F_ = torch.fft.rfft(e, dim=1) / e.shape[1]; f = torch.fft.rfftfreq(e.shape[1], 1 / 100.0)
    return [round(float((2 * F_[:, (f >= lo) & (f < hi)].abs() ** 2).sum(1).mean()), 4) for lo, hi in ((0.5, 4), (4, 10), (10, 43))]
taus = [0.0, 0.25, 0.5, 0.75, 1.0]; res = {t: [] for t in taus}; mods = {t: [] for t in taus}; S0, N = 5572800, 15 * SR
with torch.no_grad():
    for k in val + test:
        a = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); Fe = np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r"); n = 20 * SR; T = n // HOP
        s0 = 960 * (min(len(a) // 3, 30 * SR) // 960); x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); f = torch.from_numpy(np.asarray(Fe[s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev)
        if MELENC: f = logmel(x)
        f0t = torch.from_numpy(np.load(f"{B}/{F0DIR}/{k}.npy")[s0 // 240: s0 // 240 + T])[None].to(dev); ex = ctrl(f0t, loudness(x)) if EXPL else None
        mu, lv = m.encode(f, T); zg = m.encode_global(f, T)[0]; eps = torch.randn_like(mu)
        for t in taus:
            y, _ = m.decode(mu + t * eps * (0.5 * lv).exp(), zg, ex); res[t].append((k in test, mss_loss(y, x).item(), logmel_dist(y[0].cpu(), x[0].cpu())["lm"]))
            if k in test: mods[t] = modtab(y[0, : 15 * SR].cpu(), x[0, : 15 * SR].cpu()); mods["orig"] = modtab(x[0, : 15 * SR].cpu(), x[0, : 15 * SR].cpu())
print("tau | val mss | val lm | Maisky mss | Maisky lm | Maisky modulation var 0.5-4 / 4-10 / 10-43 Hz   (original:", mods["orig"], ")")
for t in taus:
    v = [r for r in res[t] if not r[0]]; te = [r for r in res[t] if r[0]]
    print(f"{t:.2f} | {np.mean([r[1] for r in v]):.3f} | {np.mean([r[2] for r in v]):.3f} | {te[0][1]:.3f} | {te[0][2]:.3f} | {mods[t]}")
# audio for listening: Maisky window @232.2 s at tau 0 and 1
k = test[0]; a = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); Fe = np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r"); T = N // HOP
f = torch.from_numpy(np.asarray(Fe[S0 // 320: S0 // 320 + N // 320], dtype=np.float32))[None].to(dev)
with torch.no_grad():
    xw = torch.from_numpy(np.asarray(a[S0:S0 + N], dtype=np.float32))[None].to(dev); f0w = torch.from_numpy(np.load(f"{B}/{F0DIR}/{k}.npy")[S0 // 240: S0 // 240 + T])[None].to(dev); exw = ctrl(f0w, loudness(xw)) if EXPL else None
    if MELENC: f = logmel(xw)
    mu, lv = m.encode(f, T); zg = m.encode_global(f, T)[0]; eps = torch.randn_like(mu)
    for t in (0.0, 1.0): sf.write(f"{B}/{name}_tau{int(t)}.wav", m.decode(mu + t * eps * (0.5 * lv).exp(), zg, exw)[0][0].cpu().numpy(), SR)
