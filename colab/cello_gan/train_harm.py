"""Harmonic-excitation vocoder, no bottleneck: wide layer-fusion encoder + a phase-coherent sine-bank excitation built from f0 (as in DDSP / neural source-filter vocoders).
The network predicts the STFT magnitude and a phase OFFSET relative to the excitation's own phase, so harmonics stay phase-locked by construction.
usage: train_harm.py NAME STEPS     env: MEL (default 1) ATTN (default 0) F0DIR (default f0clean) NHARM (default 64) BS LRS SEED"""
import json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
from vocos import Vocos
from vocos.models import VocosBackbone
from ddsp_model import SR, HOP, mss_loss
from enc_layers import LayerFusionEncoder
from metrics import logmel_dist
from voc_model import MRD
B = os.environ.get("CELLO_DIR", os.path.expanduser("~/music-visualizer/v2")); name, S1 = sys.argv[1], int(sys.argv[2]); S2 = int(sys.argv[3]) if len(sys.argv) > 3 else 0; STEPS = S1 + S2
MEL, ATTN, F0DIR, K = int(os.environ.get("MEL", 1)), int(os.environ.get("ATTN", 0)), os.environ.get("F0DIR", "f0clean"), int(os.environ.get("NHARM", 64))
BS, LRS, SEED, D = int(os.environ.get("BS", 12)), float(os.environ.get("LRS", 1.0)), int(os.environ.get("SEED", 0)), 256
NF, MELRES, BIGMSS = int(os.environ.get("NFV", 1024)), int(os.environ.get("MELRES", 0)), int(os.environ.get("BIGMSS", 0))
NMEL, NFFT = (256, 2048) if MELRES else (128, 1024); SIZES = (4096, 2048, 1024, 512, 256, 128) if BIGMSS else (2048, 1024, 512, 256, 128)
MELW = float(os.environ.get("MELW", 0))
def MSS(y, x): return mss_loss(y, x, SIZES)
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED); dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
A = {k: np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r") for k in train + val + test}; Fe = {k: (np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r") if os.path.exists(f"{B}/mertfeat/{k}.npy") else np.zeros((10 ** 7, 0), np.float32)) for k in train + val + test}      # MERT features are unused when MEL=1; the bundle leaves them out
F0 = {k: np.load(f"{B}/{F0DIR}/{k}.npy") for k in train + val + test}
SEG, GLEN = 96000, 240000; train = [k for k in train if len(A[k]) > GLEN + SEG + 2000]; w = np.array([len(A[k]) for k in train], float); w /= w.sum()
_mel = torchaudio.transforms.MelSpectrogram(SR, NFFT, NFFT, HOP, n_mels=NMEL, power=1.0).to(dev); WIN = torch.hann_window(NF, device=dev); _ml = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
def melL1(y, x): return (torch.log(_ml(y) + 1e-5) - torch.log(_ml(x) + 1e-5)).abs().mean()
def lmel(x): return ((torch.log(_mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP].transpose(1, 2)[:, :, None]      # (B,T,1,128)


@torch.no_grad()
def excitation(f0, n):                                    # f0 (B,T) Hz at 100 Hz -> (B,n) phase-coherent harmonic stack (1/k amplitudes) + a little noise
    f = F.interpolate(f0[:, None].clamp(40, 1200), size=n, mode="linear", align_corners=False)[:, 0]
    phi = torch.remainder(torch.cumsum(2 * math.pi * f.double() / SR, 1), 2 * math.pi).float()
    k = torch.arange(1, K + 1, device=f0.device).float(); ok = (f[..., None] * k < SR / 2 * 0.95).float()
    return (torch.sin(phi[..., None] * k) * ok / k).sum(-1) + 0.02 * torch.randn_like(f)


class Harm(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LayerFusionEncoder(1 if MEL else 4, NMEL if MEL else 768, D, attn=ATTN)
        self.exp = nn.Sequential(nn.Conv1d(NF // 2 + 1, 128, 1), nn.GroupNorm(8, 128), nn.GELU())
        pre = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.backbone = VocosBackbone(D + 128, 512, 1536, 8)
        miss = self.backbone.load_state_dict({k: v for k, v in pre.backbone.state_dict().items() if not k.startswith("embed.")}, strict=False)
        assert all(k.startswith("embed.") for k in miss.missing_keys) and not miss.unexpected_keys, miss
        self.out = nn.Linear(512, NF + 2)                                                                  # [log-magnitude (513) | phase offset (513)]
        with torch.no_grad():
            if NF == 1024: self.out.weight.copy_(pre.head.out.weight); self.out.bias.copy_(pre.head.out.bias)            # magnitude half starts from Vocos
            self.out.weight[NF // 2 + 1:] = 0; self.out.bias[NF // 2 + 1:] = 0                              # phase offset starts at 0 = follow the excitation

    def forward(self, f, f0, T):
        n = T * HOP; E = torch.stft(excitation(f0, n), NF, HOP, window=WIN, return_complex=True)[:, :, :T]   # (B,513,T) complex
        h, g = self.enc(f, T); e = self.exp(torch.log(E.abs() + 1e-3))
        o = self.out(self.backbone(torch.cat([h, e], 1))).transpose(1, 2)
        mag, dph = torch.exp(o[:, : NF // 2 + 1]).clamp(max=1e2), math.pi * torch.tanh(o[:, NF // 2 + 1:])
        S = mag * torch.exp(1j * (torch.angle(E) + dph)); S = torch.cat([S, S[:, :, -1:]], 2)
        return torch.istft(S, NF, HOP, window=WIN, length=n), g


INIT, ADV = os.environ.get("INIT", ""), int(os.environ.get("ADV", 0))
model = Harm().to(dev);
if INIT: model.load_state_dict(torch.load(f"{B}/{INIT}.pt", map_location=dev)["model"]); print("initialised from", INIT, flush=True)
Disc = MRD().to(dev); optd = torch.optim.AdamW(Disc.parameters(), 2e-4, betas=(0.8, 0.95)); print(len(train), "train recordings;", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params; MEL", MEL, "ATTN", ATTN, "K", K, flush=True)
opt = torch.optim.AdamW(model.parameters(), 4e-4 * LRS, betas=(0.8, 0.95), weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4 * LRS, total_steps=STEPS, pct_start=0.04)


def batch():
    ks = np.random.choice(len(train), BS, p=w); X, Fs, Ft = [], [], []
    for i in ks:
        k = train[i]; a = A[k]; s = 960 * random.randrange(0, (len(a) - SEG - 960) // 960); X.append(np.asarray(a[s:s + SEG], dtype=np.float32))
        Fs.append(np.asarray(Fe[k][s // 320: s // 320 + 300], dtype=np.float32)); Ft.append(F0[k][s // 240: s // 240 + 400])
    return [torch.from_numpy(np.stack(v)).to(dev) for v in (X, Fs, Ft)]


def feats(x, f): return lmel(x) if MEL else f


@torch.no_grad()
def evaluate(final=False, step=0):
    model.eval(); out = {}
    for k in val + test:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); T = n // HOP
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
        y, _ = model(feats(x, f), f0, T); d = logmel_dist(y[0].cpu(), x[0].cpu()); out[k[:8]] = {"mss": MSS(y, x).item(), "lm": d["lm"], "sc": d["sc"]}
    if final:
        import soundfile as sf; k = test[0]; s0 = 5572800; n = 15 * SR; T = n // HOP
        x = torch.from_numpy(np.asarray(A[k][s0:s0 + n], dtype=np.float32))[None].to(dev); f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev)
        f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev); y, _ = model(feats(x, f), f0, T); sf.write(f"{B}/{name}_cmp.wav", y[0].cpu().numpy(), SR); sf.write(f"{B}/{name}_s{step}_cmp.wav", y[0].cpu().numpy(), SR)
    model.train(); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ("mss", "lm", "sc")}; return out


log = {"args": dict(NF=NF, MELRES=MELRES, BIGMSS=BIGMSS, MEL=MEL, ATTN=ATTN, K=K, F0DIR=F0DIR, STEPS=STEPS, BS=BS), "evals": []}; t0 = time.time(); model.train()
for s in range(1, STEPS + 1):
    x, f, f0 = batch(); y, _ = model(feats(x, f), f0, 400); adv = ADV or (S2 > 0 and s > S1); loss = MSS(y, x) + (MELW * melL1(y, x) if MELW else 0)
    if adv:
        o = 240 * random.randrange(0, (SEG - 24000) // 240); xs, ys = x[:6, o:o + 24000], y[:6, o:o + 24000]       # discriminator sees a random 1 s slice (memory)
        with torch.no_grad(): dr = Disc(xs)
        df = Disc(ys); gadv = sum(F.relu(1 - q).mean() for q, _ in df) / len(df); fm = sum(F.l1_loss(a, b.detach()) for (_, fa), (_, fb) in zip(df, dr) for a, b in zip(fa, fb)) / (len(df) * 5)
        loss = loss + 0.2 * gadv + 0.4 * fm
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if adv:
        dr = Disc(xs); df = Disc(ys.detach()); ld = sum(F.relu(1 - q).mean() + F.relu(1 + p).mean() for (q, _), (p, _) in zip(dr, df)) / len(dr)
        optd.zero_grad(); ld.backward(); torch.nn.utils.clip_grad_norm_(Disc.parameters(), 5.0); optd.step()
    if s % 250 == 0: print(s, "mss", round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
    if s % 2000 == 0 or s == STEPS:
        ev = evaluate(final=True, step=s); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m: round(v, 4) for m, v in ev["val_mean"].items()}, "test", {m: round(ev[test[0][:8]][m], 4) for m in ("mss", "lm")}, flush=True)
        torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}.pt"); torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}_s{s}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
print("training done")
