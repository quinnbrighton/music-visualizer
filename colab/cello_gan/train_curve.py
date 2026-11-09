"""Curve model: unconditional flow matching over the 6 performance curves (pitch, voicing, periodicity, loudness, brightness, noisiness at 25 Hz),
same non-autoregressive transformer as the latent model (train_lat.LatFlow) with no control input. A random known prefix trains continuation, so
long pieces are built window by window and a real phrase can be given as a prompt. Its output is rendered to audio by the latent model (gen_free_lat.py).
In-training eval compares generated curves to real ones: voiced fraction, note changes per second, pitch spread, distance from the semitone grid.
usage: python train_curve.py NAME STEPS     env: TOK_DIR, CELLO_DIR, OUT, BS (32) L (500 = 20 s) D (256) LAYERS (6) HEADS (4) LR (3e-4) PCTX (0.5) EVERY (1000)"""
import os, sys, json, time, random
import numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _argv = sys.argv; sys.argv = ["x", "curve", "1"]
import train_lat as TL                                                      # LatFlow, sample
T3 = TL.T3; sys.argv = _argv; dev = "cuda"
L, D, LAYERS, HEADS = int(os.environ.get("L", 500)), int(os.environ.get("D", 256)), int(os.environ.get("LAYERS", 6)), int(os.environ.get("HEADS", 4))
NC = T3.NC; NOTES = int(os.environ.get("NOTES", 0)); NN = 4 if NOTES else 0; RESID = int(os.environ.get("RESID", 0))   # RESID=1 (needs NOTES): pitch channel = deviation from the written note   # NOTES=1: condition on guessed notes (guess_notes.py), 4 features per frame


def note_feats(n):                                                          # (T,4) [in note, MIDI, onset, velocity dB] -> model features
    on = n[:, 0]; return np.stack([on, (n[:, 1] - 48) / 12 * on, n[:, 2], (n[:, 3] + 40) / 20 * on], 1).astype(np.float32)


def note_lp(nf):                                                           # written note pitch in the curve's units (log2 of f/110) from note features; works on numpy or torch, (...,4)
    return (nf[..., 1] - 0.75) * nf[..., 0]


def to_resid(f, nf):                                                        # absolute -> note-relative pitch (0 where voiced but outside a note)
    f = f.copy(); on = (nf[:, 0] > 0) & (f[:, 1] > 0.5); f[:, 0] = np.where(on, f[:, 0] - note_lp(nf), 0.0); return f


def to_abs(c, nf):                                                          # note-relative -> absolute pitch (torch), c (...,6) cleaned, nf (...,4)
    c = c.clone(); c[..., 0] = (c[..., 0] + note_lp(nf)) * c[..., 1] * (nf[..., 0] > 0); return c


def make_net(): return TL.LatFlow(ZD=NC, NC=NN, D=D, LAYERS=LAYERS, HEADS=HEADS).to(dev)


def clean(c):                                                               # generated curves (n,6) in feature units -> the form the latent model was trained on
    c = c.clone(); v = (c[:, 1] > 0.5).float(); c[:, 1] = v; c[:, 0] *= v; c[:, 2] = c[:, 2].clamp(0, 1)
    on = (c[:, 3] > -1).float(); c[:, 4] *= on; c[:, 5] *= on; return c


def stats(c):                                                               # c (B,n,6) cleaned feature curves -> numbers to compare with real playing
    v = c[..., 1] > 0.5; cents = 1200 * c[..., 0] + 1200 * np.log2(110 / 440); dv = (cents[:, 1:] - cents[:, :-1]).abs(); both = v[:, 1:] & v[:, :-1]
    off = (cents - torch.round(cents / 100) * 100).abs()[v]
    return dict(voiced=round(float(v.float().mean()), 3), notes_per_s=round(float(((dv > 60) & both).float().sum() / c.shape[0] / (c.shape[1] / 25)), 2),
                pitch_sd_semitones=round(float(cents[v].std() / 100), 1), off_grid_cents=round(float(off.median()), 1), loud=round(float(c[..., 3].mean()), 2))


if __name__ == "__main__":
    name, STEPS = sys.argv[1], int(sys.argv[2]); TOK = os.environ["TOK_DIR"]; OUT = os.environ.get("OUT", TOK); os.makedirs(OUT, exist_ok=True)
    BS, LR, PCTX, EVERY = int(os.environ.get("BS", 32)), float(os.environ.get("LR", 3e-4)), float(os.environ.get("PCTX", 0.5)), int(os.environ.get("EVERY", 1000))
    sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu")
    keys = lambda ks: [k for k in ks if os.path.exists(f"{TOK}/controls2/{k}.npy")]; train, val = keys(sp["train"]), keys(sp["val"])

    def load(ks):
        out = {}
        for k in ks:
            c = np.load(f"{TOK}/controls/{k}.npy"); c2 = np.load(f"{TOK}/controls2/{k}.npy"); n = min(len(c), len(c2)); f = T3.ctrl_feats(c[:n], c2[:n])
            if NOTES: nf = note_feats(np.load(f"{TOK}/notes/{k}.npy")[:n]); out[k] = np.concatenate([to_resid(f, nf) if RESID else f, nf], 1)
            else: out[k] = f   # notes ride along as extra columns
        return out

    TR, VA = load(train), load(val); allc = np.concatenate([v[:, :NC] for v in TR.values()]); MU, SD = torch.tensor(allc.mean(0), device=dev), torch.tensor(allc.std(0) + 1e-4, device=dev)
    w = np.array([len(TR[k]) for k in train], float); w /= w.sum()

    def batch(data, ks, ws, n):
        X = []
        for _ in range(n):
            k = ks[np.random.choice(len(ks), p=ws)]; s = random.randrange(0, len(data[k]) - L); X.append(data[k][s:s + L])
        return torch.from_numpy(np.stack(X)).to(dev)

    net = make_net(); print(len(TR), "train recordings, %.1f h | params %.1fM" % (len(allc) / 25 / 3600, sum(p.numel() for p in net.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(net.parameters(), LR, betas=(0.9, 0.99), weight_decay=0.01); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.03)
    if os.environ.get("INITC"):                                           # bootstrap from another curve-model checkpoint (weights + EMA)
        ci = torch.load(os.environ["INITC"], map_location=dev); net.load_state_dict(ci["model"]); print("initialised from", os.environ["INITC"], "step", ci["step"], flush=True)
    ema = [p.detach().clone() for p in net.parameters()]
    if os.environ.get("INITC"): ema = [ci["ema"][n].detach().clone() for n, _ in net.named_parameters()]
    noc = torch.zeros(BS, L, 0, device=dev)
    vkeys = list(VA); vw = np.array([len(VA[k]) for k in vkeys], float); vw /= vw.sum(); random.seed(0); np.random.seed(0); VB = [batch(VA, vkeys, vw, 16) for _ in range(4)]
    vb = torch.cat(VB); rv = clean(vb[..., :NC].reshape(-1, NC)).reshape(64, L, NC); rv = to_abs(rv, vb[..., NC:]) if RESID else rv
    real = stats(rv); print("real val curves", real, flush=True)
    CDROP = float(os.environ.get("CDROP", 0.1)); TMIX = float(os.environ.get("TMIX", 0))   # TMIX: share of flow times drawn from a density rising toward t=1 (fine detail is learned there)
    log = {"args": dict(BS=BS, L=L, D=D, LAYERS=LAYERS, HEADS=HEADS, LR=LR, PCTX=PCTX, NOTES=NOTES, CDROP=CDROP, TMIX=TMIX, RESID=RESID, ADV=float(os.environ.get('ADV', 0))), "norm": {"mu": MU.tolist(), "sd": SD.tolist()}, "real": real, "evals": []}

    ADV = float(os.environ.get("ADV", 0))                                 # adversarial weight on curve detail (flow + GAN hybrid); 0 = off
    if ADV:
        sn = nn.utils.spectral_norm; CI = [0, 1, 3]                         # judged: pitch (note-relative when RESID), voicing, loudness - conditioned on the note features
        Dn = nn.Sequential(sn(nn.Conv1d(len(CI) + NN, 128, 5, padding=2)), nn.LeakyReLU(0.2), sn(nn.Conv1d(128, 128, 5, padding=4, dilation=2)), nn.LeakyReLU(0.2),
                           sn(nn.Conv1d(128, 128, 5, padding=8, dilation=4)), nn.LeakyReLU(0.2), sn(nn.Conv1d(128, 1, 3, padding=1))).to(dev)
        optD = torch.optim.AdamW(Dn.parameters(), 2e-4, betas=(0.5, 0.9))

    def adv_terms(cn):                                                      # one-step clean estimate at late flow time (t in [0.8, 0.98]) -> LSGAN on random 64-frame crops (2.5 s)
        c, nf = cn[..., :NC], cn[..., NC:]; x1 = (c - MU) / SD; B_ = x1.shape[0]; e = torch.randn_like(x1); t = 0.8 + 0.18 * torch.rand(B_, device=dev)
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * e; v = net(xt, t, nf, torch.zeros_like(x1), torch.zeros(B_, L, device=dev), torch.zeros(B_, device=dev))
        xh = xt + (1 - t[:, None, None]) * v; o = random.randrange(0, L - 64)
        real = torch.cat([x1[:, o:o + 64, CI], nf[:, o:o + 64]], 2).transpose(1, 2); fake = torch.cat([xh[:, o:o + 64, CI], nf[:, o:o + 64]], 2).transpose(1, 2)
        return real, fake

    def fm_loss(cn, train_mode=True):
        c, nf = cn[..., :NC], cn[..., NC:]; x1 = (c - MU) / SD; B_ = x1.shape[0]; e = torch.randn_like(x1); t = torch.sigmoid(torch.randn(B_, device=dev)); t = torch.where(torch.rand(B_, device=dev) < TMIX, torch.rand(B_, device=dev) ** 0.5, t); xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * e
        kl = torch.where(torch.rand(B_, device=dev) < PCTX, torch.randint(1, L // 2, (B_,), device=dev), torch.zeros(B_, dtype=torch.long, device=dev)) if train_mode else torch.zeros(B_, dtype=torch.long, device=dev)
        km = (torch.arange(L, device=dev)[None] < kl[:, None]).float()
        drop = (torch.rand(B_, device=dev) < CDROP).float() if (train_mode and NOTES) else torch.zeros(B_, device=dev)
        v = net(xt, t, nf, x1, km, drop); err = ((v - (x1 - e)) ** 2).mean(-1); m = 1 - km; return (err * m).sum() / m.sum()

    @torch.no_grad()
    def evaluate():
        bk = [p.detach().clone() for p in net.parameters()]
        for p, e in zip(net.parameters(), ema): p.copy_(e)
        net.eval(); vl = float(np.mean([fm_loss(c, False).item() for c in VB])); g = TL.sample(net, VB[0][..., NC:], gs=float(os.environ.get("EVALGS", 1))) * SD + MU
        g = torch.stack([clean(x) for x in g]); g = to_abs(g, VB[0][..., NC:]) if RESID else g; r = stats(g)
        if NOTES:                                                           # how well the performed pitch follows the real performance of the same notes
            rc = to_abs(VB[0][..., :NC].clone(), VB[0][..., NC:]) if RESID else VB[0][..., :NC]; vv = (g[..., 1] > 0.5) & (rc[..., 1] > 0.5); e = (1200 * (g[..., 0] - rc[..., 0])).abs()[vv]
            r.update(pitch_vs_real_cents=round(float(e.median()), 1), within50=round(float((e < 50).float().mean()), 3), voicing_agree=round(float(((g[..., 1] > 0.5) == (rc[..., 1] > 0.5)).float().mean()), 3))
        for p, b in zip(net.parameters(), bk): p.copy_(b)
        net.train(); return vl, r

    t0 = time.time(); net.train()
    for s in range(1, STEPS + 1):
        cb = batch(TR, train, w, BS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = fm_loss(cb)
            if ADV: real, fake = adv_terms(cb); gl = ((Dn(fake.float()) - 1) ** 2).mean(); loss = loss + ADV * gl
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if ADV:
            dl = ((Dn(real.float()) - 1) ** 2).mean() + (Dn(fake.detach().float()) ** 2).mean(); optD.zero_grad(); dl.backward(); optD.step()
            if s % 250 == 0: print(s, "adv g %.3f d %.3f" % (gl.item(), dl.item()), flush=True)
        with torch.no_grad():
            for p, e in zip(net.parameters(), ema): e.mul_(0.999).add_(p.detach(), alpha=0.001)
        if s % 250 == 0: print(s, "flow loss %.4f" % loss.item(), round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            vl, r = evaluate(); log["evals"].append({"step": s, "val_loss": vl, **r}); print("EVAL", s, "val flow loss %.4f" % vl, "| generated", r, flush=True)
            ck = {"model": net.state_dict(), "ema": {n: e for (n, _), e in zip(net.named_parameters(), ema)}, "args": log["args"], "norm": log["norm"], "step": s}
            torch.save(ck, f"{OUT}/{name}_last.pt"); json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)
            if s % 5000 == 0: torch.save(ck, f"{OUT}/{name}_s{s}.pt")
    print("training done")
