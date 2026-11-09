"""usage: train_ddsp.py NAME STEPS   env: ZDIM GDIM USE_GLOBAL BETA FASTSCALE"""
import json, math, os, random, sys, time
import numpy as np, torch, torchaudio, torch.nn.functional as F
from mert_ddsp import *
from metrics import logmel_dist
B = os.path.expanduser("~/music-visualizer/v2"); name, STEPS = sys.argv[1], int(sys.argv[2])
ZD, GD = int(os.environ.get("ZDIM", 24)), int(os.environ.get("GDIM", 8)); UG = int(os.environ.get("USE_GLOBAL", 1)); BETA = float(os.environ.get("BETA", 0.002)); FS_ = float(os.environ.get("FASTSCALE", 1)); AUXF = float(os.environ.get("AUXF", 0.05)); AUXL = float(os.environ.get("AUXL", 1.0)); F0DIR = os.environ.get("F0DIR", "f0"); F0RES = int(os.environ.get("F0RES", 0)); BG = float(os.environ.get("BG", 1)); NEST = float(os.environ.get("NEST", 0))
random.seed(0); torch.manual_seed(0); dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
A = {k: np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r") for k in train + val + test}; F0 = {k: np.load(f"{B}/{F0DIR}/{k}.npy") for k in train + val + test}
SEG, GLEN = 96000, 240000
train = [k for k in train if len(A[k]) > GLEN + SEG]; w = np.array([len(A[k]) for k in train], float); w /= w.sum()
print(len(train), "train recordings >= 14 s", flush=True)
model = MertDDSP(ZD, GD).to(dev).train(); print(sum(p.numel() for p in model.parameters()) / 1e6, "M params", flush=True)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], 5e-4, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 5e-4, total_steps=STEPS, pct_start=0.05)
def batch(bs=8):
    ks = np.random.choice(len(train), bs, p=w); X, G, Fz = [], [], []
    for i in ks:
        k = train[i]; a = A[k]; s = 240 * random.randrange(0, (len(a) - SEG - 1) // 240); X.append(np.asarray(a[s:s + SEG], dtype=np.float32)); Fz.append(F0[k][s // 240: s // 240 + SEG // 240])
        g = 240 * random.randrange(0, (len(a) - GLEN - 1) // 240); G.append(np.asarray(a[g:g + GLEN], dtype=np.float32))
    return [torch.from_numpy(np.stack(v)).to(dev) for v in (X, G, Fz)]
_mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
def model_logmel(x): return ((torch.log(_mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP]
def beta(s): return BETA * min(1.0, s / (0.3 * STEPS))
@torch.no_grad()
def evaluate(final=False):
    model.eval(); out = {}
    for k in val + test:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); n = 20 * SR; s0 -= s0 % 240; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev)
        f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + n // 240])[None].to(dev); mu, lv = model.encode(x); zg = model.encode_global(x)[0]
        y, parts = model.decode(mu, zg, n); kl = kl_vec(mu, lv, 0.0).sum().item()
        d = logmel_dist(y[0].cpu(), x[0].cpu()); f0err = float((12 * torch.log2(parts["f0"] / f0.clamp(40, 1200)).abs().clamp(max=12)).median()); out[k[:8]] = {"f0_med_err_semitones": f0err, "mss": mss_loss(y, x).item(), "lm": d["lm"], "sc": d["sc"], "kl_nats_per_frame": kl, "bits_per_s": kl * 100 / math.log(2)}
        if final and k in test: import soundfile as sf; sf.write(f"{B}/{name}_test_recon.wav", y[0, : 15 * SR].cpu().numpy() / max(1e-6, y[0].abs().max().item()) * 0.9, SR)
    model.train(); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ("mss", "lm", "sc", "bits_per_s")}; return out
log = {"args": dict(MERT=1, AUXF=AUXF, AUXL=AUXL, F0RES=F0RES, F0DIR=F0DIR, ZD=ZD, GD=GD, UG=UG, BETA=BETA, FS=FS_, steps=STEPS), "evals": []}; t0 = time.time(); model.train()
for s in range(1, STEPS + 1):
    x, g, f0 = batch(); loud = loudness(x)
    mu, lv = model.encode(x); z = mu + torch.randn_like(mu) * (0.5 * lv).exp(); zg = None; klg = torch.zeros((), device=dev)
    if UG:
        gm, glv = model.encode_global(g); zg = gm + torch.randn_like(gm) * (0.5 * glv).exp(); klg = kl_vec(gm, glv, 0.02).sum()
    y, pr = model.decode(z, zg, SEG)
    aux = AUXF * F.huber_loss(12 * torch.log2(pr['f0'] / f0.clamp(40, 1200)), torch.zeros_like(f0), delta=3.0) + AUXL * F.l1_loss(pr['loud'], loud)
    rec = mss_loss(y, x); kv = kl_vec(mu, lv); kw = torch.ones_like(kv); kw[: RATES[0][0] + RATES[1][0]] = FS_
    loss = rec + aux + beta(s) * ((kv * kw).sum() + BG * klg)
    if NEST:      # one random level per step: keep only groups >= j (faster groups zeroed), must rebuild the box-smoothed log-mel at that group's window
        j = random.randrange(1, len(RATES)); wj = RATES[j][1]; b = int(sum(g for g, _ in RATES[:j])); keep = torch.zeros(1, ZD, 1, device=dev); keep[:, b:] = 1
        yj, _ = model.decode(z * keep, zg, SEG); tgt = F.avg_pool1d(F.pad(model_logmel(x), (wj // 2, wj // 2), mode="replicate"), wj, stride=1)
        loss = loss + NEST * (model_logmel(yj) - tgt).abs().mean()
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if s % 250 == 0: print(s, "mss", round(rec.item(), 4), "aux", round(aux.item(), 3), "kl", round(kv.sum().item(), 2), "klg", round(klg.item(), 2), round(time.time() - t0), "s", flush=True)
    if s % 2500 == 0 or s == STEPS:
        ev = evaluate(final=(s == STEPS)); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m: round(v, 4) for m, v in ev["val_mean"].items()}, "test", {m: round(ev[test[0][:8]][m], 4) for m in ("mss", "lm", "bits_per_s")}, flush=True)
        torch.save({"model": {k: v for k, v in model.state_dict().items() if not k.startswith('mert.')}, "args": log["args"]}, f"{B}/{name}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
