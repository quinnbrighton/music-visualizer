"""Score every system on the SAME 15 s held-out Maisky window, in the audio domain (24 kHz)."""
import json, numpy as np, soundfile as sf, torch, torchaudio, sys
from ddsp_model import *
from metrics import logmel_dist
B = "/home/qbrighto/music-visualizer/v2"; ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); k = ck["test"][0]
s0 = 5572800; n = 15 * SR          # = 232.2 s exactly (20000 frames x 512 / 44100), multiple of 240
x = torch.from_numpy(np.asarray(np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r")[s0:s0 + n], dtype=np.float32))
def load44(p): w, sr = sf.read(p, dtype="float32"); return torchaudio.functional.resample(torch.from_numpy(w if w.ndim == 1 else w.mean(1)), sr, SR)[:n]
mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0)
def modtable(y, ref):
    E = mel(y)[:, :1500]; R = mel(ref)[:, :1500]; keep = R.mean(1) > 0.05 * R.mean(1).max(); e = E[keep]; e = e / (e.mean(1, keepdim=True) + 1e-9) - 1
    F_ = torch.fft.rfft(e, dim=1) / e.shape[1]; f = torch.fft.rfftfreq(e.shape[1], 1 / 100.0)
    return [round(float((2 * F_[:, (f >= lo) & (f < hi)].abs() ** 2).sum(1).mean()), 4) for lo, hi in ((0.5, 4), (4, 10), (10, 43))]
systems = {"original": x, "GL(true 512-mel)": load44(f"{B}/mn_2_true_spectrogram_griffinlim.wav"), "nested spectrogram AE + GL": load44(f"{B}/mn_3_multirate_reconstruction.wav")}
for name in sys.argv[1:]:                                                    # DDSP checkpoints
    c = torch.load(f"{B}/{name}.pt", map_location="cuda"); a = c["args"]; m = DDSPAE(a["ZD"], a["GD"], use_global=bool(a["UG"]), f0res=bool(a.get("F0RES", 0))).cuda().eval(); m.load_state_dict(c["model"])
    f0 = torch.from_numpy(np.load(f"{B}/{a.get('F0DIR', 'f0')}/{k}.npy")[s0 // 240: s0 // 240 + n // 240])[None].cuda(); xx = x[None].cuda()
    with torch.no_grad():
        mu, lv = m.encode(xx); zg = m.encode_global(xx)[0] if a["UG"] else None; y, parts = m.decode(f0, loudness(xx), mu, zg, n)
        kl = kl_vec(mu, lv, 0.0)
    systems[name] = y[0].cpu(); print(name, "bits/s", round(float(kl.sum()) * 100 / np.log(2)), "| KL by group (nats):", [round(float(v), 2) for v in torch.split(kl, [g for g, _ in RATES])[0:0]] or [round(float(kl[i:j].sum()), 2) for i, j in zip(np.cumsum([0] + [g for g, _ in RATES])[:-1], np.cumsum([g for g, _ in RATES]))])
    sf.write(f"{B}/cmp_{name}.wav", (y[0] / y[0].abs().max() * 0.9).cpu().numpy(), SR)
print("\nsystem | logmel L1 | spectral conv. | modulation variance 0.5-4 / 4-10 / 10-43 Hz  (original shows the target)")
res = {}
for nme, y in systems.items():
    y = y[:n] * (x.pow(2).mean().sqrt() / (y[:n].pow(2).mean().sqrt() + 1e-9))      # match RMS level to the original
    d = logmel_dist(y, x) if nme != "original" else {"lm": 0.0, "sc": 0.0}; res[nme] = {**d, "mod": modtable(y, x)}; print(f"{nme:28s} {d['lm']:.3f} {d['sc']:.3f} {res[nme]['mod']}")
json.dump(res, open(f"{B}/compare_systems.json", "w"), indent=1)
