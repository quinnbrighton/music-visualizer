"""usage: train_voc.py NAME STAGE1_STEPS STAGE2_STEPS   (stage 1: spectral loss only; stage 2: + multi-resolution discriminator)
env: BETA BG AUXF AUXL F0DIR ZDIM GDIM"""
import json, math, os, random, sys, time
import numpy as np, torch, torchaudio, torch.nn.functional as F
from voc_model import *
from metrics import logmel_dist
B = os.path.expanduser("~/music-visualizer/v2"); name, S1, S2 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); STEPS = S1 + S2
BETA = float(os.environ.get("BETA", 0.002)); BG = float(os.environ.get("BG", 0.05)); AUXF = float(os.environ.get("AUXF", 0.05)); AUXL = float(os.environ.get("AUXL", 1.0)); F0DIR = os.environ.get("F0DIR", "f0clean")
ZD, GD = int(os.environ.get("ZDIM", 24)), int(os.environ.get("GDIM", 8)); BS = int(os.environ.get("BS", 12)); INIT = os.environ.get("INIT", ""); EXPL = int(os.environ.get("EXPL", 0)); MELENC = int(os.environ.get("MELENC", 0)); LRS = float(os.environ.get("LRS", 1.0))
SEED = int(os.environ.get("SEED", 0)); random.seed(SEED); torch.manual_seed(SEED); np.random.seed(SEED); dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
A = {k: np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r") for k in train + val + test}; Fe = {k: np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r") for k in train + val + test}
F0 = {k: np.load(f"{B}/{F0DIR}/{k}.npy") for k in train + val + test}
SEG, GLEN = 96000, 240000; train = [k for k in train if len(A[k]) > GLEN + SEG + 2000]; w = np.array([len(A[k]) for k in train], float); w /= w.sum()
model = VocModel(ZD, GD, extra=2 * EXPL, mel_in=bool(MELENC)).to(dev)
_mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0).to(dev)
def logmel(x): return ((torch.log(_mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP]
D = MRD().to(dev); print(len(train), "train recordings;", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M generator params", flush=True)
if INIT: model.load_state_dict(torch.load(f'{B}/{INIT}.pt', map_location=dev)['model']); print('initialised from', INIT, flush=True)
opt = torch.optim.AdamW(model.parameters(), 4e-4 * LRS, betas=(0.8, 0.95), weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-4 * LRS, total_steps=STEPS, pct_start=0.04)
optd = torch.optim.AdamW(D.parameters(), 2e-4, betas=(0.8, 0.95))
def batch():
    ks = np.random.choice(len(train), BS, p=w); X, Fs, G, Ft, Ga = [], [], [], [], []
    for i in ks:
        k = train[i]; a = A[k]; s = 960 * random.randrange(0, (len(a) - SEG - 960) // 960); X.append(np.asarray(a[s:s + SEG], dtype=np.float32))
        Fs.append(np.asarray(Fe[k][s // 320: s // 320 + 300], dtype=np.float32)); Ft.append(F0[k][s // 240: s // 240 + 400])
        g = 960 * random.randrange(0, (len(a) - GLEN - 960) // 960); G.append(np.asarray(Fe[k][g // 320: g // 320 + 750], dtype=np.float32)); Ga.append(np.asarray(a[g:g + GLEN], dtype=np.float32))
    return [torch.from_numpy(np.stack(v)).to(dev) for v in (X, Fs, G, Ft, Ga)]
def ctrl(f0, loud): return torch.stack([torch.log2(f0.clamp(40, 1200) / 110), loud], 1)
def beta(s): return BETA * min(1.0, s / (0.3 * STEPS))
@torch.no_grad()
def evaluate(final=False):
    model.eval(); out = {}
    for k in val + test:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev)
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); T = n // HOP
        f0t = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
        if MELENC: f = logmel(x)
        mu, lv = model.encode(f, T); zg = model.encode_global(f, T)[0]; y, pr = model.decode(mu, zg, ctrl(f0t, loudness(x)) if EXPL else None); kl = kl_vec(mu, lv, 0.0).sum().item()
        d = logmel_dist(y[0].cpu(), x[0].cpu())
        out[k[:8]] = {"mss": mss_loss(y, x).item(), "lm": d["lm"], "sc": d["sc"], "bits_per_s": kl * 100 / math.log(2), "f0_probe_med_err_semitones": float((12 * torch.log2(pr["f0"] / f0t.clamp(40, 1200)).abs().clamp(max=12)).median())}
    if final:
        import soundfile as sf; k = test[0]; s0 = 5572800; n = 15 * SR; a = A[k]
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); T = n // HOP
        f0f = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev); xf = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev)
        f = logmel(xf) if MELENC else f
        mu, _ = model.encode(f, T); zg = model.encode_global(f, T)[0]; y, _ = model.decode(mu, zg, ctrl(f0f, loudness(xf)) if EXPL else None); sf.write(f"{B}/{name}_cmp.wav", y[0].cpu().numpy(), SR)
    model.train(); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ("mss", "lm", "sc", "bits_per_s", "f0_probe_med_err_semitones")}; return out
log = {"args": dict(MELENC=MELENC, EXPL=EXPL, S1=S1, S2=S2, BETA=BETA, BG=BG, F0DIR=F0DIR, ZD=ZD, GD=GD), "evals": []}; t0 = time.time(); model.train()
for s in range(1, STEPS + 1):
    x, f, fg, f0, ga = batch(); loud = loudness(x)
    if MELENC: f, fg = logmel(x), logmel(ga); adv = s > S1
    mu, lv = model.encode(f, 400); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    gm, glv = model.encode_global(fg, 1000); zg = gm + torch.randn_like(gm) * (0.5 * glv).exp(); klg = kl_vec(gm, glv, 0.02).sum()
    y, pr = model.decode(z, zg, ctrl(f0, loud) if EXPL else None)
    rec = mss_loss(y, x); kv = kl_vec(mu, lv)
    aux = AUXF * F.huber_loss(12 * torch.log2(pr["f0"] / f0.clamp(40, 1200)), torch.zeros_like(f0), delta=3.0) + AUXL * F.l1_loss(pr["loud"], loud)
    loss = rec + aux + beta(s) * (kv.sum() + BG * klg); gadv = fm = torch.zeros((), device=dev)
    if adv:
        o = 240 * random.randrange(0, (SEG - 24000) // 240); xs, ys = x[:6, o:o + 24000], y[:6, o:o + 24000]      # discriminator sees a random 1 s slice (memory)
        with torch.no_grad(): dr = D(xs)
        df = D(ys)
        gadv = sum(F.relu(1 - o).mean() for o, _ in df) / len(df); fm = sum(F.l1_loss(a, b.detach()) for (_, fa), (_, fb) in zip(df, dr) for a, b in zip(fa, fb)) / (len(df) * 5)
        loss = loss + 0.2 * gadv + 0.4 * fm
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if adv:
        dr = D(xs); df = D(ys.detach()); ld = sum(F.relu(1 - o).mean() + F.relu(1 + p).mean() for (o, _), (p, _) in zip(dr, df)) / len(dr)
        optd.zero_grad(); ld.backward(); torch.nn.utils.clip_grad_norm_(D.parameters(), 5.0); optd.step()
    if s % 250 == 0: print(s, "mss", round(rec.item(), 4), "aux", round(aux.item(), 3), "kl", round(kv.sum().item(), 2), "klg", round(klg.item(), 2), "adv", round(gadv.item(), 3), "fm", round(fm.item(), 3), round(time.time() - t0), "s", flush=True)
    if s % 2000 == 0 or s == STEPS:
        ev = evaluate(final=(s == STEPS)); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m: round(v, 4) for m, v in ev["val_mean"].items()}, "test", {m: round(ev[test[0][:8]][m], 4) for m in ("mss", "lm", "bits_per_s")}, flush=True)
        torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
print("training done")
