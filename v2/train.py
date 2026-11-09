"""Train the v2 autoencoder. usage: train.py NAME FRAME_DIM GLOBAL_DIM STEPS"""
import json, random, sys, time
import numpy as np, torch, torch.nn.functional as F
from model import AE, kl, load_data, read, norm_stats, B

import os
CONTRAST = float(os.environ.get('CONTRAST', 0)); BG = float(os.environ.get('BETA_G', 5)); BF = float(os.environ.get('BETA_F', 0.002))
name, FD, GD, STEPS = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
random.seed(0); torch.manual_seed(0)
idx, train, val, test = load_data()
mean, std = norm_stats(train)
data = {k: ((read(k) - mean[:, None]) / std[:, None]).half() for k in train + val + test}
dev = "cuda"
# duplicate check: val/test recordings whose mean spectrum is nearly identical to a train recording
prof = {k: data[k].float().mean(1) for k in data}
dups = []
for k in val + test:
    c = max(float(torch.cosine_similarity(prof[k] - prof[k].mean(), prof[t] - prof[t].mean(), 0)) for t in train)
    print("max profile cosine to train:", k[:8], round(c, 4), flush=True)
    if c > 0.99 and k in val: dups.append(k)
val = [k for k in val if k not in dups]; print("dropped near-duplicate val files:", [d[:8] for d in dups], flush=True)
LW, LG = int(os.environ.get('LW', 256)), 860
BS = int(os.environ.get('BS', 16))
TDIFF = float(os.environ.get('TDIFF', 0)); FASTSCALE = float(os.environ.get('FASTSCALE', 1))   # temporal-difference loss weight; KL multiplier for the 2 fastest groups
NESTED = float(os.environ.get('NESTED', 0))   # weight of the partial-level (slow-groups-only) reconstruction losses
RATES = [tuple(map(int, r.split(':'))) for r in os.environ['RATES'].split(',')] if os.environ.get('RATES') else None                                       # target window, global-clip length (frames)
model = AE(FD, GD, rates=RATES).to(dev)
print(name, sum(p.numel() for p in model.parameters()) / 1e6, "M params; train", len(train), "val", len(val), "test", len(test), flush=True)
opt = torch.optim.AdamW(model.parameters(), 1e-3, weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, 1e-3, total_steps=STEPS, pct_start=0.05)
weights = np.array([data[k].shape[1] for k in train], float); weights /= weights.sum()


def crop(k, L):
    x = data[k]; i = random.randrange(0, x.shape[1] - L); return x[:, i:i + L]


def batch(bs=None):
    bs = bs or BS
    ks = np.random.choice(len(train), bs, replace=False, p=weights)
    tg = torch.stack([crop(train[i], LW) for i in ks]).float().to(dev)
    gc = torch.stack([crop(train[i], LG) for i in ks]).float().to(dev) if GD else None   # same recording, different time
    gc2 = torch.stack([crop(train[i], LG) for i in ks]).float().to(dev) if GD and CONTRAST else None
    return tg, gc, gc2


def beta(s):
    return BF * min(1.0, s / (0.3 * STEPS))


@torch.no_grad()
def recon_eval(k, nchunk=60):
    """Out-of-sample recon error on a held-out recording. Global code from the whole recording (chunked mean pool)."""
    model.eval(); x = data[k].float().to(dev); T = x.shape[1]
    zg = None
    if GD:
        f = torch.stack([model.enc_global_feat(x[None, :, i:i + LG])[0] for i in range(0, T - LG, LG)]).mean(0, keepdim=True)
        zg = model.global_from_feat(f)[0]
    errs, base = [], []
    for i in range(0, T - LW, (T - LW) // nchunk):
        c = x[None, :, i:i + LW]; mu, _ = model.enc_frame(c); y = model.decode(mu, zg)
        errs.append(((y - c) * std.to(dev)[None, :, None]).abs().mean().item())      # natural-log mel units
        base.append(((c.mean(2, keepdim=True) * 0 - c) * std.to(dev)[None, :, None]).abs().mean().item())
    model.train()
    return float(np.mean(errs)), float(np.mean(base))


log = {"args": [name, FD, GD, STEPS], "curve": []}
t0 = time.time(); model.train()
for s in range(1, STEPS + 1):
    tg, gc, gc2 = batch()
    mu, lv = model.enc_frame(tg); zf = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    zg = None; klg = torch.zeros(())
    if GD:
        gm, glv = model.global_from_feat(model.enc_global_feat(gc)); zg = gm + torch.randn_like(gm) * (0.5 * glv).exp()
        klg, _ = kl(gm, glv, 0.02)
        if CONTRAST:
            g2 = model.global_from_feat(model.enc_global_feat(gc2))[0]
            a, b = F.normalize(gm, dim=1), F.normalize(g2, dim=1); lg = a @ b.T / 0.1; lab = torch.arange(len(a), device=dev)
            con = (F.cross_entropy(lg, lab) + F.cross_entropy(lg.T, lab)) / 2
    y = model.decode(zf, zg)
    rec = F.l1_loss(y, tg) + 0.5 * F.mse_loss(y, tg)
    nest = torch.zeros((), device=dev)
    if NESTED and RATES:
        # groups are ordered fast -> slow. Level j keeps groups j..end (zeros the faster ones = prior mean) and must rebuild the
        # spectrogram box-smoothed at group j's window. The fastest group (j=0) is the plain full reconstruction above: no smoothing, no penalty.
        bounds = np.cumsum([0] + [n for n, _ in RATES])
        for j in range(1, len(RATES)):
            w = RATES[j][1]; keep = torch.zeros(1, zf.shape[1], 1, device=dev); keep[:, bounds[j]:] = 1
            tgt = F.avg_pool1d(F.pad(tg, (w // 2, w // 2), mode="replicate"), w, stride=1)
            yj = model.decode(zf * keep, zg); nest = nest + F.l1_loss(yj, tgt) + 0.5 * F.mse_loss(yj, tgt)
        nest = nest / (len(RATES) - 1)
    kvec = (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).mean((0, 2)).clamp_min(0.02)
    KW = torch.ones_like(kvec)
    if RATES: KW[: RATES[0][0] + RATES[1][0]] = FASTSCALE
    klf = (kvec * KW).sum()
    if TDIFF:
        td = sum(F.l1_loss(y[:, :, l:] - y[:, :, :-l], tg[:, :, l:] - tg[:, :, :-l]) for l in (1, 3)) / 2
        rec = rec + TDIFF * td
    loss = rec + NESTED * nest + beta(s) * klf + beta(s) * BG * klg + (CONTRAST * con if (GD and CONTRAST) else 0)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step(); sch.step()
    if s % 500 == 0:
        ev = {k[:8]: recon_eval(k)[0] for k in val[:3]}
        print(s, round(rec.item(), 4), "klf", round(klf.item(), 2), "nest", round(float(nest), 3), "klg", round(float(klg), 2), "con", round(float(con) if (GD and CONTRAST) else 0, 3), "val", {a: round(b, 4) for a, b in ev.items()}, round(time.time() - t0), "s", flush=True)
        log["curve"].append({"step": s, "rec": rec.item(), "val": ev})
torch.save({"model": model.state_dict(), "mean": mean, "std": std, "args": [FD, GD], "rates": RATES, "val": val, "test": test, "train": train}, f"{B}/{name}.pt")
res = {"val": {k: recon_eval(k) for k in val}, "test": {k: recon_eval(k) for k in test}}
res["val_mean_l1"] = float(np.mean([v[0] for v in res["val"].values()])); res["test_mean_l1"] = float(np.mean([v[0] for v in res["test"].values()]))
log["final"] = res; json.dump(log, open(f"{B}/{name}.json", "w"), indent=1); print(json.dumps({k: res[k] for k in ("val_mean_l1", "test_mean_l1")}))
