"""v2: no hard voiced/unvoiced flag. Periodicity is a soft regression target; pitch is trained on every LOUD frame with trust weights
(periodicity, and whether CREPE and yin agree within 50 cents). Eval on loud frames, split by CREPE/yin consensus.
Learn f0 from the encoder: layer-fusion encoder (log-mel or cached MERT layers) -> CREPE-style pitch-bin classification (25-cent bins) + voicing.
Teacher: f0crepe/ + per/ (periodicity). usage: train_pitch.py NAME STEPS     env: MEL (1=log-mel, 0=MERT layers), BS"""
import json, math, os, random, sys, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio
from ddsp_model import SR, HOP, loudness
from enc_layers import LayerFusionEncoder
B = os.path.expanduser("~/music-visualizer/v2"); name, STEPS = sys.argv[1], int(sys.argv[2])
MEL, BS, SEED, D = int(os.environ.get("MEL", 1)), int(os.environ.get("BS", 16)), int(os.environ.get("SEED", 0)), 256
FMIN, CENT, NB = 60.0, 25.0, 209                      # bin i = FMIN * 2**(i*CENT/1200); 60 Hz .. ~1150 Hz
random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED); dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
A = {k: np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r") for k in train + val + test}; Fe = {k: np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r") for k in train + val + test}
F0 = {k: np.load(f"{B}/f0crepe/{k}.npy") for k in train + val + test}; YIN = {k: np.load(f"{B}/f0clean/{k}.npy") for k in train + val + test}; PER = {k: np.load(f"{B}/per/{k}.npy").astype(np.float32) for k in train + val + test}
SEG = 96000; train = [k for k in train if len(A[k]) > 240000 + SEG + 2000]; w = np.array([len(A[k]) for k in train], float); w /= w.sum()
_mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
def lmel(x): return ((torch.log(_mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP].transpose(1, 2)[:, :, None]
centers = torch.arange(NB, device=dev).float() * CENT                                                      # cents above FMIN
def to_cents(f0): return 1200 * torch.log2(f0.clamp(FMIN, 1200) / FMIN)


class PitchNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LayerFusionEncoder(1 if MEL else 4, 128 if MEL else 768, D)
        self.pitch, self.voice = nn.Conv1d(D, NB, 1), nn.Conv1d(D, 1, 1)

    def forward(self, f, T):
        h, _ = self.enc(f, T); return self.pitch(h), self.voice(h)[:, 0]

    @staticmethod
    def decode(lg):                                                                                       # CREPE-style: local weighted mean (in cents) around the argmax
        p = torch.softmax(lg, 1); i = p.argmax(1, keepdim=True); m = (torch.arange(NB, device=lg.device)[None, :, None] - i).abs() <= 2
        c = (p * m * centers[None, :, None]).sum(1) / (p * m).sum(1).clamp_min(1e-6); return FMIN * 2 ** (c / 1200)


model = PitchNet().to(dev); print(len(train), "train recordings;", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params; MEL", MEL, flush=True)
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS, pct_start=0.05)


def batch():
    ks = np.random.choice(len(train), BS, p=w); X, Fs, T0, V, Yn = [], [], [], [], []
    for i in ks:
        k = train[i]; a = A[k]; s = 960 * random.randrange(0, (len(a) - SEG - 960) // 960); X.append(np.asarray(a[s:s + SEG], dtype=np.float32))
        Fs.append(np.asarray(Fe[k][s // 320: s // 320 + 300], dtype=np.float32)); T0.append(F0[k][s // 240: s // 240 + 400]); V.append(PER[k][s // 240: s // 240 + 400]); Yn.append(YIN[k][s // 240: s // 240 + 400])
    return [torch.from_numpy(np.stack(v)).to(dev) for v in (X, Fs, T0, V, Yn)]


def feats(x, f): return lmel(x) if MEL else f


@torch.no_grad()
def evaluate():
    model.eval(); out = {}
    for k in val + test:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; T = n // HOP
        x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev)
        lg, vo = model(feats(x, f), T); c = to_cents(PitchNet.decode(lg)[0]); tc = to_cents(torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T]).to(dev)); ty = to_cents(torch.from_numpy(YIN[k][s0 // 240: s0 // 240 + T]).to(dev))
        pr = torch.from_numpy(PER[k][s0 // 240: s0 // 240 + T]).to(dev); loud = loudness(x)[0, :T] > 1/3; ec, ey = (c - tc).abs(), (c - ty).abs(); cons = loud & ((tc - ty).abs() < 50)
        out[k[:8]] = {"cons_med_cents": ec[cons].median().item(), "cons_within50": (ec[cons] < 50).float().mean().item(), "cons_octave": ((ec[cons] > 1100) & (ec[cons] < 1300)).float().mean().item(),
                      "loud_within50_either": (torch.minimum(ec, ey)[loud] < 50).float().mean().item(), "loud_within50_crepe": (ec[loud] < 50).float().mean().item(), "periodicity_mae": (torch.sigmoid(vo[0]) - pr).abs()[loud].mean().item()}
    model.train(); ms = tuple(next(iter(out.values())).keys()); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ms}; return out


log = {"args": dict(MEL=MEL, STEPS=STEPS), "evals": []}; t0 = time.time(); model.train()
for s in range(1, STEPS + 1):
    x, f, t, v, yn = batch(); lg, vo = model(feats(x, f), 400); c = to_cents(t)
    tgt = torch.exp(-0.5 * ((centers[None, :, None] - c[:, None]) / CENT / 1.0) ** 2); tgt = tgt / tgt.sum(1, keepdim=True)        # gaussian soft label, sigma = 25 cents
    loud = (loudness(x)[:, :400] > 1/3).float(); agree = ((1200 * torch.log2(t / yn).abs()) < 50).float(); m = loud * (0.2 + 0.8 * v.clamp(0, 1)) * (0.3 + 0.7 * agree)
    ce = -(tgt * F.log_softmax(lg, 1)).sum(1); loss = (ce * m).sum() / m.sum().clamp_min(1e-3) + 0.5 * F.binary_cross_entropy_with_logits(vo, v.clamp(0, 1))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if s % 250 == 0: print(s, "loss", round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
    if s % 1000 == 0 or s == STEPS:
        ev = evaluate(); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m_: round(v_, 4) for m_, v_ in ev["val_mean"].items()}, "test", {m_: round(ev[test[0][:8]][m_], 4) for m_ in ("cons_med_cents", "cons_within50", "loud_within50_either")}, flush=True)
        torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
print("training done")
