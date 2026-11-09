"""Regularized continuous codec: fine-tune the 64x12 RVQ codec's encoder/decoder WITHOUT quantization so its 25 Hz 24-channel latent becomes
(1) decorrelated and unit-scale per channel (interpretable, comparable channels), (2) perceptually uniform: a step of fixed size in any latent
direction changes the sound (log-mel) and the pitch head's estimate by about the same amount (path-length regularization, decoder noise fixed so
only the latent change is measured), and (3) robust to small latent errors (noise added before decoding). Reconstruction, pitch, texture and
adversarial losses continue as in train_rvq.py. (4) optional temporal predictability: a small causal conv net, trained jointly, predicts each
frame from the previous ~1.8 s; the codec is penalized by the fraction of (per-channel standardized) variance it cannot predict (PREDW). Eval reports reconstruction plus the uniformity numbers from latent_sensitivity.py.
usage: python train_reg.py NAME STEPS     env: as train_rvq.py, plus REG_INIT (codec checkpoint with model [+ disc]), REG_RESUME (continue),
COVW (5) SCALEW (1) PLW (0.5) PITCHW (1) PLEPS (0.5 latent SD) PLB (8 clips) NOISE (0.1 latent SD) RAMP (1000 steps to full weight)
PREDW (0 = off) PREDEXEMPT (0) RAW (0) REG_DISC (discriminators from another checkpoint when REG_INIT has none)"""
import os, sys
_name, _steps = sys.argv[1], int(sys.argv[2]); sys.argv = ["x", _name, str(_steps)]; os.environ.setdefault("NINIT", "1"); os.environ.setdefault("NQ", "12"); os.environ.setdefault("K", "64")
_s = open(os.path.join(os.environ.get("CELLO_DIR", "."), "train_rvq.py")).read(); exec(_s[: _s.index("\nOPT_PARAMS")])   # data, model, discriminators, losses, to25, dec_feats
COVW, SCALEW, PLW, PITCHW = (float(os.environ.get(k, d)) for k, d in (("COVW", 5), ("SCALEW", 1), ("PLW", 0.5), ("PITCHW", 1)))
PLEPS, PLB, NOISE, RAMP = float(os.environ.get("PLEPS", 0.5)), int(os.environ.get("PLB", 8)), float(os.environ.get("NOISE", 0.1)), int(os.environ.get("RAMP", 1000))
MEL = _MS[5]                                                                # 1024-sample window, 160 mel bands
RAW = int(os.environ.get("RAW", 0))    # 1 = sound-space mode: path-length directions, latent noise, sensitivity nudges and the predictability penalty in RAW latent units (one global SD), only a global size term
def scl(sd): return sd.pow(2).mean().sqrt().expand_as(sd) if RAW else sd


PREDW = float(os.environ.get("PREDW", 0)); PREDEXEMPT = int(os.environ.get("PREDEXEMPT", 0))   # 1 = note changes exempt from the predictability penalty


class Pred(nn.Module):                                                      # causal: frame t predicted from frames < t (dilated convs, receptive field 46 frames = 1.8 s)
    def __init__(self, d=256):
        super().__init__(); self.inp = nn.Conv1d(ZD, d, 1); self.cs = nn.ModuleList([nn.Conv1d(d, d, 4, dilation=r) for r in (1, 2, 4, 8)]); self.out = nn.Conv1d(d, ZD, 1)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)

    def forward(self, z):                                                   # z (B,ZD,t) standardized -> prediction of z[:, :, t] from z[:, :, :t]
        h = self.inp(F.pad(z, (1, 0))[:, :, :-1])
        for c_ in self.cs: h = h + F.gelu(c_(F.pad(h, (3 * c_.dilation[0], 0))))
        return self.out(h)


pred = Pred().to(dev)


def pred_loss(z, mu, sd, w=None):                                           # fraction of standardized variance the causal predictor cannot explain (skip the first 8 frames: no context)
    zs = (z.float() - mu[None, :, None]) / sd[None, :, None]; e = (pred(zs) - zs).pow(2).mean(1)[:, 8:]
    return e.mean() if w is None else (e * w[:, 8:]).sum() / w[:, 8:].sum().clamp_min(1)


def onset_free(f0, pr, t):                                                  # 25 Hz weight: 0 within +-2 frames of a note change (pitch jump > 50 cents or voicing change), else 1
    c = to_cents(f0.clamp_min(30)); v = (pr > 0.5).float(); ch = torch.zeros_like(v); ch[:, 1:] = (((c[:, 1:] - c[:, :-1]).abs() > 50) & (v[:, 1:] * v[:, :-1] > 0)).float() + (v[:, 1:] != v[:, :-1]).float()
    ch = F.max_pool1d(ch[:, None], DSR, DSR)[:, 0][:, :t]; return 1 - F.max_pool1d(ch[:, None], 5, 1, 2)[:, 0].clamp(max=1)


OPT_PARAMS = [p for n_, p in model.named_parameters() if not n_.startswith("pr")] + list(pred.parameters())
opt2 = torch.optim.AdamW(OPT_PARAMS, GLR2, betas=(0.8, 0.99), weight_decay=1e-4); sch2 = torch.optim.lr_scheduler.ExponentialLR(opt2, 0.999996)
rlog = {"args": dict(RAW=RAW, PREDW=PREDW, COVW=COVW, SCALEW=SCALEW, PLW=PLW, PITCHW=PITCHW, PLEPS=PLEPS, PLB=PLB, NOISE=NOISE, RAMP=RAMP, GLR=GLR2, BS=BS, init=os.environ.get("REG_INIT")), "evals": []}
START2 = 1; pl_a = None
if os.environ.get("REG_RESUME") and os.path.exists(os.environ["REG_RESUME"]):
    c = torch.load(os.environ["REG_RESUME"], map_location=dev, weights_only=False); model.load_state_dict(c["model"]); [D_.load_state_dict(sd) for D_, sd in zip(DS, c["disc"])]
    opt2.load_state_dict(c["opt"]); optd.load_state_dict(c["optd"]); sch2.load_state_dict(c["sch"]); schd.load_state_dict(c["schd"]); START2 = c["step"] + 1; pl_a = c.get("pl_a")
    if "pred" in c: pred.load_state_dict(c["pred"])
    if os.path.exists(f"{OUT}/{name}.json"): rlog = json.load(open(f"{OUT}/{name}.json"))
    print("resumed regularized run at step", START2, flush=True)
else:
    c = torch.load(os.environ["REG_INIT"], map_location=dev, weights_only=False); model.load_state_dict(c["model"])
    dsrc = c if "disc" in c else (torch.load(os.environ["REG_DISC"], map_location=dev, weights_only=False) if os.environ.get("REG_DISC") else None)
    if dsrc is not None: [D_.load_state_dict(sd) for D_, sd in zip(DS, dsrc["disc"])]
    c = dict(c, disc=1) if dsrc is not None else c
    rvq.load_state_dict(c["rvq"]); print("initialised from", os.environ["REG_INIT"], "| discriminators", "loaded" if "disc" in c else "fresh", flush=True)


def logmel(y): return torch.log(MEL(y.float()) + 1e-5)
def soft_cents(hh): return (torch.softmax(model.ph(hh)[:, :NB].float(), 1) * CENTERS[None, :, None]).sum(1)     # differentiable pitch estimate (cents)


def decode_fixed(z, T, seed):                                               # decoder with its noise path seeded, so two calls differ only by the latent
    with torch.random.fork_rng(devices=[torch.device(dev)]):
        torch.manual_seed(seed); hh = dec_feats(z, T); y = model.synth(hh, model.pitch_f0(hh).detach(), T)
    return hh, y


def path_change(z, u, T, seed, vm):                                         # per-clip change in log-mel + PITCHW * pitch (semitones) per unit latent step along u
    hh0, y0 = decode_fixed(z, T, seed); hh1, y1 = decode_fixed(z + u, T, seed); du = u.pow(2).mean((1, 2)).sqrt()
    dm = (logmel(y1) - logmel(y0)).abs().mean((1, 2)); dp = ((soft_cents(hh1) - soft_cents(hh0)).abs() / 100 * vm).sum(1) / vm.sum(1).clamp_min(1)
    return (dm + PITCHW * dp) / du, dm / du, dp / du


def latent_stats(z):                                                        # z (B,24,t) -> mean |off-diagonal correlation|, per-channel sd, standardized z
    zf = z.float().transpose(1, 2).reshape(-1, z.shape[1]); mu, sd = zf.mean(0), zf.std(0) + 1e-5; zs = (zf - mu) / sd; C = zs.t() @ zs / len(zs)
    off = C - torch.diag(torch.diag(C)); return off, sd


@torch.no_grad()
def eval_reg(step):
    model.eval(); r = {m: [] for m in ("mss", "lm", "tex", "cents", "w50")}; Zs = []; clips = []
    for k in val:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n_ = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n_], dtype=np.float32))[None].to(dev); T = n_ // HOP
        f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev); vm = torch.from_numpy(PER[k][s0 // 240: s0 // 240 + T]).to(dev)[None] > 0.5; z25 = to25(feats(x, None), T)
        hh = dec_feats(z25, T); fp = model.pitch_f0(hh); y = model.synth(hh, fp, T); Zs.append(z25[0]); clips.append((z25[:, :, :200], vm[:, :800].float()))
        r["mss"].append(MSS(y, x).item()); r["lm"].append(logmel_dist(y[0].cpu(), x[0].cpu())["lm"]); r["tex"].append(float(np.mean(texparts(y, x, 2000))))
        e = (to_cents(fp) - to_cents(f0))[vm].abs(); r["cents"].append(e.median().item()); r["w50"].append((e < 50).float().mean().item())
    out = {m: float(np.mean(v)) for m, v in r.items()}; Z = torch.cat(Zs, 1)[None]; off, sd = latent_stats(Z)
    zc = (Z[0] - Z[0].mean(1, keepdim=True)) / sd[:, None]; ev = torch.linalg.eigvalsh(zc @ zc.t() / zc.shape[1]).flip(0)
    zm, zsd = Z[0].mean(1), scl(Z[0].std(1) + 1e-5); out["pred_unexplained"] = float(pred_loss(Z, zm, zsd))
    out.update(mean_abs_corr=float(off.abs().sum() / (ZD * (ZD - 1))), max_abs_corr=float(off.abs().max()), sd_min=float(sd.min()), sd_max=float(sd.max()), pca_top1=float(ev[0] / ev.sum()))
    sm, sp = [], []                                                         # per-channel sensitivity: +-PLEPS sd along each axis, fixed decoder noise, first two val clips (8 s)
    for i in range(ZD):
        u = torch.zeros(1, ZD, 1, device=dev); u[0, i] = PLEPS * scl(sd)[i]; m_, p_ = [], []
        for zc8, vm8 in clips[:2]:
            for sgn in (1, -1): _, dm, dp = path_change(zc8, sgn * u, 800, 1234, vm8); m_.append(dm.item()); p_.append(dp.item())
        sm.append(np.mean(m_) * PLEPS); sp.append(np.mean(p_) * PLEPS * 100)
    out.update(sens_mel=[round(v, 4) for v in sm], sens_cents=[round(v, 2) for v in sp], sens_mel_spread=float(max(sm) / min(sm)), sens_cents_spread=float(max(sp) / max(min(sp), 1e-3)))
    import soundfile as sf; k = test[0]; s0 = 5572800; n_ = 15 * SR; T = n_ // HOP; x = torch.from_numpy(np.asarray(A[k][s0:s0 + n_], dtype=np.float32))[None].to(dev); z25 = to25(feats(x, None), T)
    hh = dec_feats(z25, T); sf.write(f"{OUT}/{name}_s{step}.wav", model.synth(hh, model.pitch_f0(hh), T)[0].cpu().numpy(), SR)
    set_train(); return out


def save(step):
    st = {"model": model.state_dict(), "pred": pred.state_dict(), "rvq": rvq.state_dict(), "disc": [D_.state_dict() for D_ in DS], "opt": opt2.state_dict(), "optd": optd.state_dict(), "sch": sch2.state_dict(), "schd": schd.state_dict(), "step": step, "args": rlog["args"], "pl_a": pl_a}
    torch.save(st, f"{OUT}/{name}_latest.pt")
    if step % 2000 == 0 or step == STEPS: torch.save({k_: st[k_] for k_ in ("model", "pred", "rvq", "step", "args")}, f"{OUT}/{name}_s{step}.pt")
    json.dump(rlog, open(f"{OUT}/{name}.json", "w"), indent=1)
def show(step, ev): print("EVAL", step, "mss %.4f lm %.4f tex %.4f cents %.1f w50 %.3f | corr mean %.3f max %.3f | sd %.2f-%.2f | pca top1 %.3f | sensitivity spread mel %.2fx pitch %.2fx | unpredictable %.3f" % tuple(ev[k] for k in ("mss", "lm", "tex", "cents", "w50", "mean_abs_corr", "max_abs_corr", "sd_min", "sd_max", "pca_top1", "sens_mel_spread", "sens_cents_spread", "pred_unexplained")), flush=True)


if START2 == 1: ev = eval_reg(0); rlog["evals"].append({"step": 0, **ev}); show(0, ev); json.dump(rlog, open(f"{OUT}/{name}.json", "w"), indent=1)
set_train(); t0 = time.time()
for s in range(START2, STEPS + 1):
    ramp = min(1.0, s / RAMP); x, f, f0, pr = batch(); ff = feats(x, f); T = 400
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMPG):
        z25 = to25(ff, T); off, sd = latent_stats(z25); covl = off.pow(2).sum() / (ZD * (ZD - 1)); scalel = torch.log(scl(sd)).pow(2).mean(); predl = pred_loss(z25, z25.float().mean((0, 2)).detach(), scl(sd).detach(), onset_free(f0, pr, z25.shape[2]) if PREDEXEMPT else None)   # every frame, note changes included, unless PREDEXEMPT
        sdd = scl(sd).detach()[None, :, None]; zin = z25 + torch.rand(z25.shape[0], 1, 1, device=dev) * NOISE * sdd * torch.randn_like(z25)
        hh = dec_feats(zin, T)
        with torch.no_grad(): ht, _ = teacher(ff, T)
        lg = model.ph(hh); pl, pv = lg[:, :NB].float(), lg[:, NB].float(); fp = pdecode(pl); vm = (pr > 0.5).float()
        tg = torch.exp(-0.5 * ((CENTERS[None, :, None] - to_cents(f0)[:, None]) / CENT) ** 2); tg = tg / tg.sum(1, keepdim=True)
        pce = ((-(tg * F.log_softmax(pl, 1)).sum(1)) * vm).sum() / vm.sum().clamp_min(1); pper = F.binary_cross_entropy_with_logits(pv, pr.clamp(0, 1)); pcents = (to_cents(fp) - to_cents(f0)).abs()[vm > 0].median()
        y = model.synth(hh, fp.detach(), T)
        u = torch.randn(PLB, ZD, 1, device=dev); u = u / u.pow(2).mean(1, keepdim=True).sqrt() * PLEPS * sdd     # random direction, PLEPS latent-sd per channel (rms)
        g, _, _ = path_change(z25[:PLB], u, T, s, vm[:PLB, :T])
    with torch.no_grad(): pl_a = g.mean().item() if pl_a is None else 0.99 * pl_a + 0.01 * g.mean().item()
    pll = ((g.float() / pl_a - 1) ** 2).mean()
    y, hh, ht = y.float(), hh.float(), ht.float()
    dist = F.mse_loss(hh, ht) / ht.var().clamp_min(1e-6); rec = MSS(y, x) + MELW * melL1(y, x)
    with torch.no_grad(): cx = contrast(x, f0)
    cy = contrast(y, f0); cgap = (cy - cx).abs().mean(); tex = texloss(y, x); msm = msmel(y, x)
    loss = rec + CW * cgap + TW * tex + PW * (pce + 0.3 * pper) + DIST * 0.2 * dist + MSW * msm + ramp * (COVW * covl + SCALEW * scalel + PLW * pll + PREDW * predl) + pred_loss(z25.detach(), z25.float().mean((0, 2)).detach(), scl(sd).detach())   # last term trains the predictor alone
    o_ = random.randrange(0, x.shape[1] - CROP); xc, yc = x[:, o_:o_ + CROP], y[:, o_:o_ + CROP]; gadv = gfm = torch.zeros((), device=dev)
    for D_ in DS:
        with torch.no_grad(): dr = dcall(D_, xc)
        for (sf_, ff_), (sr_, fr_) in zip(dcall(D_, yc), dr): gadv = gadv + ((1 - sf_) ** 2).mean(); gfm = gfm + sum(F.l1_loss(a, b) for a, b in zip(ff_, fr_))
    loss = loss + ADV * gadv + FMW * gfm
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt2.zero_grad(); continue
    opt2.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(OPT_PARAMS, 1e3); opt2.step(); sch2.step()
    for p in Dp: p.requires_grad_(True)
    ld = torch.zeros((), device=dev); yd = yc.detach()
    for D_ in DS:
        for (sr_, _), (sf_, _) in zip(dcall(D_, xc), dcall(D_, yd)): ld = ld + ((1 - sr_) ** 2).mean() + (sf_ ** 2).mean()
    optd.zero_grad(); ld.backward(); torch.nn.utils.clip_grad_norm_(Dp, 10.0); optd.step(); schd.step()
    for p in Dp: p.requires_grad_(False)
    if s % 250 == 0: print(s, "rec", round(rec.item(), 3), "msmel", round(msm.item(), 3), "tex", round(tex.item(), 3), "pitch_cents", round(pcents.item(), 1), "cov", round(covl.item(), 4), "scale", round(scalel.item(), 3),
                           "path", round(pll.item(), 3), "unpred", round(predl.item(), 3), "path_mean", round(pl_a, 3), "g_adv", round(gadv.item(), 3), "d_loss", round(ld.item(), 3), round(time.time() - t0), "s", flush=True)
    if s % EVERY == 0 or s == STEPS:
        ev = eval_reg(s); rlog["evals"].append({"step": s, **ev}); show(s, ev); save(s)
print("training done")
