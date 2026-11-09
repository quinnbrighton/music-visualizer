"""No-bottleneck test: cached MERT layers -> LayerFusionEncoder (wide, 256 ch @100 Hz, no KL, no global code) -> Vocos decoder. Spectral loss only.
usage: train_wide.py NAME STEPS     env: ATTN (attention blocks, default 2) PAST (default 300) FUTURE (default 0) BS LRS SEED"""
import json, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from vocos import Vocos
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from ddsp_model import SR, HOP, mss_loss
from enc_layers import LayerFusionEncoder
from metrics import logmel_dist
B = os.path.expanduser("~/music-visualizer/v2"); name, STEPS = sys.argv[1], int(sys.argv[2])
ATTN, PAST, FUT = int(os.environ.get("ATTN", 2)), int(os.environ.get("PAST", 300)), int(os.environ.get("FUTURE", 0))
MEL = int(os.environ.get("MEL", 0))
BS, LRS, SEED, D = int(os.environ.get("BS", 12)), float(os.environ.get("LRS", 1.0)), int(os.environ.get("SEED", 0)), 256
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED); dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
A = {k: np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r") for k in train + val + test}; Fe = {k: np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r") for k in train + val + test}
SEG, GLEN = 96000, 240000; train = [k for k in train if len(A[k]) > GLEN + SEG + 2000]; w = np.array([len(A[k]) for k in train], float); w /= w.sum()


class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LayerFusionEncoder(1 if MEL else 4, 128 if MEL else 768, D, attn=ATTN, past=PAST, future=FUT)
        pre = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.backbone = VocosBackbone(D, 512, 1536, 8)
        miss = self.backbone.load_state_dict({k: v for k, v in pre.backbone.state_dict().items() if not k.startswith("embed.")}, strict=False)
        assert all(k.startswith("embed.") for k in miss.missing_keys) and not miss.unexpected_keys, miss
        self.head = ISTFTHead(512, 1024, HOP, padding="same"); self.head.out.load_state_dict(pre.head.out.state_dict())

    def forward(self, f, T):
        h, g = self.enc(f, T); return self.head(self.backbone(h)), g


import torchaudio
_mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
def lmel(x): return ((torch.log(_mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP].transpose(1, 2)[:, :, None]      # (B,T,1,128)
model = Wide().to(dev); print(len(train), "train recordings;", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params; attn", ATTN, "past", PAST, "future", FUT, flush=True)
opt = torch.optim.AdamW(model.parameters(), 4e-4 * LRS, betas=(0.8, 0.95), weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4 * LRS, total_steps=STEPS, pct_start=0.04)


def batch():
    ks = np.random.choice(len(train), BS, p=w); X, Fs = [], []
    for i in ks:
        k = train[i]; a = A[k]; s = 960 * random.randrange(0, (len(a) - SEG - 960) // 960); X.append(np.asarray(a[s:s + SEG], dtype=np.float32))
        Fs.append(np.asarray(Fe[k][s // 320: s // 320 + 300], dtype=np.float32))
    return [torch.from_numpy(np.stack(v)).to(dev) for v in (X, Fs)]


@torch.no_grad()
def evaluate(final=False):
    model.eval(); out = {}; gs = []
    for k in val + test:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev)
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); T = n // HOP
        f = lmel(x) if MEL else f
        y, g = model(f, T); d = logmel_dist(y[0].cpu(), x[0].cpu()); out[k[:8]] = {"mss": mss_loss(y, x).item(), "lm": d["lm"], "sc": d["sc"]}; gs.append(g.mean((0, 1)).cpu().numpy())
    if final:
        import soundfile as sf; k = test[0]; s0 = 5572800; n = 15 * SR
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev)
        xf = torch.from_numpy(np.asarray(A[k][s0:s0 + n], dtype=np.float32))[None].to(dev); f = lmel(xf) if MEL else f
        y, _ = model(f, n // HOP); sf.write(f"{B}/{name}_cmp.wav", y[0].cpu().numpy(), SR)
    model.train(); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ("mss", "lm", "sc")}; out["layer_gate_mean"] = np.mean(gs, 0).round(3).tolist(); return out


log = {"args": dict(ATTN=ATTN, PAST=PAST, FUT=FUT, STEPS=STEPS, BS=BS), "evals": []}; t0 = time.time(); model.train()
for s in range(1, STEPS + 1):
    x, f = batch(); f = lmel(x) if MEL else f; y, _ = model(f, 400); loss = mss_loss(y, x)
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if s % 250 == 0: print(s, "mss", round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
    if s % 2000 == 0 or s == STEPS:
        ev = evaluate(final=(s == STEPS)); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m: round(v, 4) for m, v in ev["val_mean"].items()}, "test", {m: round(ev[test[0][:8]][m], 4) for m in ("mss", "lm")}, "layer gates (1,4,8,12)", ev["layer_gate_mean"], flush=True)
        torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
print("training done")
