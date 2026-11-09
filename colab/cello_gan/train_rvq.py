"""Residual-VQ stage, branched from the adversarial checkpoint.

The 24-channel code is averaged to 100/DS frames per second, quantised by an NQ-stage residual VQ (KC entries per stage, codebooks
started from k-means on real code values, EMA updates, dead entries reset from the batch), linearly upsampled back to 100 Hz and decoded.
Quantizer dropout trains the first n stages at random, so one model runs at any 1..NQ stages (bit rate = frames/s * n * log2(KC)).
Stage 1 (FREEZE steps): decoder frozen, encoder and codebooks adapt. Then everything trains. Adversarial losses continue throughout.
Evaluates at step 0 (plain k-means) and every EVERY steps, at several numbers of stages; writes clips, a JSON log and checkpoints to OUT_DIR.
usage: python train_rvq.py NAME STEPS     env: as train_gan.py, plus RESUME (adversarial *_latest.pt to branch from), RVQ_RESUME (continue
this run), NQ (8) K (1024) DS (4) FREEZE (2000) EVERY (1000) QDROP (0.5) CB_DECAY (0.99) COMMIT (0.25) NINIT (20) DEAD (300) GLR (5e-5)"""
import os, sys
_name, _steps = sys.argv[1], int(sys.argv[2]); sys.argv = ["x", _name, "0", str(_steps)]
_dir = os.environ.get("CELLO_DIR", os.path.expanduser("~/music-visualizer/v2"))
_src = open(os.path.join(_dir, "train_gan.py")).read(); exec(_src[: _src.index("log = {")])          # data, model (+ RESUME weights), discriminators, losses
import json
NQ, KC, DSR = int(os.environ.get("NQ", 8)), int(os.environ.get("K", 1024)), int(os.environ.get("DS", 4))
FREEZE, EVERY, QDROP = int(os.environ.get("FREEZE", 2000)), int(os.environ.get("EVERY", 1000)), float(os.environ.get("QDROP", 0.5))
DECAY, COMMIT, NINIT, DEAD = float(os.environ.get("CB_DECAY", 0.99)), float(os.environ.get("COMMIT", 0.25)), int(os.environ.get("NINIT", 20)), int(os.environ.get("DEAD", 300))
GLR2 = float(os.environ.get("GLR", 5e-5)); EVAL_N = sorted({int(x) for x in os.environ.get("EVALN", "1,2,4,6,8").split(",") if int(x) <= NQ} | {NQ})


class RVQ(nn.Module):
    def __init__(self, nq, k, d):
        super().__init__(); self.nq, self.k = nq, k
        for nm, sh in (("cb", (nq, k, d)), ("ema_w", (nq, k, d)), ("ema_n", (nq, k)), ("age", (nq, k))): self.register_buffer(nm, torch.zeros(sh))

    @staticmethod
    def nearest(r, c):
        return torch.cat([(rr.pow(2).sum(1, keepdim=True) - 2 * rr @ c.t() + c.pow(2).sum(1)[None]).argmin(1) for rr in r.split(65536)])

    @torch.no_grad()
    def kmeans_init(self, x, iters=25):                                    # stage by stage on the residual left by the stages before
        r = x.clone()
        for i in range(self.nq):
            pick = torch.randperm(len(r), device=r.device)[: self.k] if len(r) >= self.k else torch.randint(0, len(r), (self.k,), device=r.device); c = r[pick].clone()
            for _ in range(iters):
                idx = self.nearest(r, c); cnt = torch.bincount(idx, minlength=self.k).float(); s = torch.zeros_like(c).index_add_(0, idx, r); full = cnt > 0
                c[full] = s[full] / cnt[full, None]; c[~full] = r[torch.randint(0, len(r), (int((~full).sum()),), device=r.device)]
            self.cb[i] = c; self.ema_w[i] = c; self.ema_n[i] = 1.0; r = r - c[self.nearest(r, c)]
        return r

    def forward(self, x, n, update=False):                                 # x (N,d) -> straight-through quantised x, codes (N,n), commitment loss
        r = x.detach(); q = torch.zeros_like(r); codes = []
        for i in range(n):
            idx = self.nearest(r, self.cb[i]); qi = self.cb[i][idx]; codes.append(idx)
            if update: self._update(i, r, idx)
            q = q + qi; r = r - qi
        return x + (q - x).detach(), torch.stack(codes, 1), F.mse_loss(x, q)

    @torch.no_grad()
    def _update(self, i, r, idx):
        cnt = torch.bincount(idx, minlength=self.k).float(); s = torch.zeros_like(self.cb[i]).index_add_(0, idx, r)
        self.ema_n[i].mul_(DECAY).add_(cnt, alpha=1 - DECAY); self.ema_w[i].mul_(DECAY).add_(s, alpha=1 - DECAY)
        tot = self.ema_n[i].sum(); ns = (self.ema_n[i] + 1e-5) / (tot + self.k * 1e-5) * tot; self.cb[i] = self.ema_w[i] / ns[:, None]
        self.age[i] += 1; self.age[i][cnt > 0] = 0; dead = self.age[i] > DEAD
        if dead.any():
            v = r[torch.randint(0, len(r), (int(dead.sum()),), device=r.device)]; self.cb[i][dead] = v; self.ema_w[i][dead] = v; self.ema_n[i][dead] = 1.0; self.age[i][dead] = 0


rvq = RVQ(NQ, KC, ZD).to(dev)
def to25(ff, T):                                                           # encoder -> 24-channel code -> averaged to 100/DS frames per second
    h, _ = model.enc(ff, T); mu, _lv = model.down(h).chunk(2, 1); return F.avg_pool1d(model.smooth(mu), DSR, DSR)
def quant(z25, n, update=False):
    b_, c_, t_ = z25.shape
    with torch.autocast("cuda", enabled=False): zq, codes, commit = rvq(z25.float().transpose(1, 2).reshape(-1, c_), n, update)
    return zq.reshape(b_, t_, c_).transpose(1, 2), codes.reshape(b_, t_, n), commit
def dec_feats(zq, T): return model.up(F.interpolate(zq, size=T, mode="linear", align_corners=False))

OPT_PARAMS = [p for n_, p in model.named_parameters() if not n_.startswith("pr")]                                        # the temporal prior is not used with discrete codes
DEC = [p for n_, p in model.named_parameters() if n_.split(".")[0] in ("up", "exp", "backbone", "out", "nz", "ph")]
opt2 = torch.optim.AdamW(OPT_PARAMS, GLR2, betas=(0.8, 0.99), weight_decay=1e-4); sch2 = torch.optim.lr_scheduler.ExponentialLR(opt2, 0.999996)
rlog = {"args": dict(NQ=NQ, K=KC, DS=DSR, FREEZE=FREEZE, QDROP=QDROP, GLR=GLR2, BS=BS, COMMIT=COMMIT, branch=os.environ.get("RESUME", INITW)), "evals": []}; START2 = 1
if os.environ.get("RVQ_RESUME"):
    c = torch.load(os.environ["RVQ_RESUME"], map_location=dev, weights_only=False); model.load_state_dict(c["model"]); rvq.load_state_dict(c["rvq"]); [D_.load_state_dict(sd) for D_, sd in zip(DS, c["disc"])]
    opt2.load_state_dict(c["opt"]); optd.load_state_dict(c["optd"]); sch2.load_state_dict(c["sch"]); schd.load_state_dict(c["schd"]); START2 = c["step"] + 1
    if os.path.exists(f"{OUT}/{name}.json"): rlog = json.load(open(f"{OUT}/{name}.json"))
    print("resumed RVQ run at step", START2, flush=True)
else:
    model.eval()
    with torch.no_grad(): X = torch.cat([to25(feats(xb, fb), 400).float().transpose(1, 2).reshape(-1, ZD) for xb, fb, _, _ in (batch() for _ in range(NINIT))])
    res = rvq.kmeans_init(X); print("init: k-means on %d code frames; residual energy left after %d stages: %.4f of the code variance" % (len(X), NQ, (res.pow(2).sum(1).mean() / (X - X.mean(0)).pow(2).sum(1).mean()).item()), flush=True)


@torch.no_grad()
def eval_rvq(step):
    model.eval(); acc = {n: {m: [] for m in ("mss", "lm", "tex", "cents", "w50")} for n in EVAL_N}; allc = []
    for k in val:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n_ = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n_], dtype=np.float32))[None].to(dev); T = n_ // HOP
        f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev); vm = torch.from_numpy(PER[k][s0 // 240: s0 // 240 + T]).to(dev)[None] > 0.5; z25 = to25(feats(x, None), T)
        for nq in EVAL_N:
            zq, codes, _ = quant(z25, nq); hh = dec_feats(zq, T); fp = model.pitch_f0(hh); y = model.synth(hh, fp, T); r = acc[nq]
            r["mss"].append(MSS(y, x).item()); r["lm"].append(logmel_dist(y[0].cpu(), x[0].cpu())["lm"]); r["tex"].append(float(np.mean(texparts(y, x, 2000))))
            e = (to_cents(fp) - to_cents(f0))[vm].abs(); r["cents"].append(e.median().item()); r["w50"].append((e < 50).float().mean().item())
            if nq == NQ: allc.append(codes[0])
    out = {str(n): dict({m: float(np.mean(v)) for m, v in d.items()}, bits_per_s=100 / DSR * n * math.log2(KC)) for n, d in acc.items()}
    C = torch.cat(allc); out["usage"] = []
    for i in range(NQ):
        cnt = torch.bincount(C[:, i], minlength=KC).float(); p = cnt[cnt > 0] / cnt.sum(); out["usage"].append({"used": int((cnt > 0).sum()), "perplexity": float(torch.exp(-(p * p.log()).sum()))})
    import soundfile as sf; k = test[0]; s0 = 5572800; n_ = 15 * SR; T = n_ // HOP; x = torch.from_numpy(np.asarray(A[k][s0:s0 + n_], dtype=np.float32))[None].to(dev); z25 = to25(feats(x, None), T)
    for nq in sorted({NQ, 4, 2} & set(EVAL_N)):
        zq, _, _ = quant(z25, nq); hh = dec_feats(zq, T); sf.write(f"{OUT}/{name}_q{nq}_s{step}.wav", model.synth(hh, model.pitch_f0(hh), T)[0].cpu().numpy(), SR)
    set_train(); return out


def save(step):
    st = {"model": model.state_dict(), "rvq": rvq.state_dict(), "disc": [D_.state_dict() for D_ in DS], "opt": opt2.state_dict(), "optd": optd.state_dict(), "sch": sch2.state_dict(), "schd": schd.state_dict(), "step": step, "args": rlog["args"]}
    torch.save(st, f"{OUT}/{name}_latest.pt")
    if step % 5000 == 0: torch.save({k_: st[k_] for k_ in ("model", "rvq", "disc", "step", "args")}, f"{OUT}/{name}_s{step}.pt")
    json.dump(rlog, open(f"{OUT}/{name}.json", "w"), indent=1)
def show(step, ev): print("EVAL", step, " | ".join("q%s %d b/s mss %.4f lm %.4f tex %.4f cents %.1f" % (n, ev[n]["bits_per_s"], ev[n]["mss"], ev[n]["lm"], ev[n]["tex"], ev[n]["cents"]) for n in map(str, EVAL_N)), "| codes used per stage", [u["used"] for u in ev["usage"]], flush=True)


if START2 == 1: ev = eval_rvq(0); rlog["evals"].append({"step": 0, **ev}); show(0, ev); json.dump(rlog, open(f"{OUT}/{name}.json", "w"), indent=1)
set_train(); t0 = time.time()
for s in range(START2, STEPS + 1):
    frozen = s <= FREEZE
    for p in DEC: p.requires_grad_(not frozen)
    nq = NQ if random.random() >= QDROP else random.randint(1, NQ)                                  # quantizer dropout
    x, f, f0, pr = batch(); ff = feats(x, f); T = 400
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMPG):
        z25 = to25(ff, T); zq, codes, commit = quant(z25, nq, update=True); hh = dec_feats(zq, T)
        with torch.no_grad(): ht, _ = teacher(ff, T)
        lg = model.ph(hh); pl, pv = lg[:, :NB].float(), lg[:, NB].float(); fp = pdecode(pl); vm = (pr > 0.5).float()
        tg = torch.exp(-0.5 * ((CENTERS[None, :, None] - to_cents(f0)[:, None]) / CENT) ** 2); tg = tg / tg.sum(1, keepdim=True)
        pce = ((-(tg * F.log_softmax(pl, 1)).sum(1)) * vm).sum() / vm.sum().clamp_min(1); pper = F.binary_cross_entropy_with_logits(pv, pr.clamp(0, 1)); pcents = (to_cents(fp) - to_cents(f0)).abs()[vm > 0].median()
        y = model.synth(hh, fp.detach(), T)
    y, hh, ht = y.float(), hh.float(), ht.float()
    dist = F.mse_loss(hh, ht) / ht.var().clamp_min(1e-6); rec = MSS(y, x) + MELW * melL1(y, x)
    with torch.no_grad(): cx = contrast(x, f0)
    cy = contrast(y, f0); cgap = (cy - cx).abs().mean(); tex = texloss(y, x); msm = msmel(y, x)
    loss = rec + CW * cgap + TW * tex + PW * (pce + 0.3 * pper) + DIST * 0.2 * dist + COMMIT * commit + MSW * msm
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
    if s % 250 == 0: print(s, "frozen" if frozen else "all", "nq", nq, "rec", round(rec.item(), 3), "msmel", round(msm.item(), 3), "tex", round(tex.item(), 3), "pitch_cents", round(pcents.item(), 1), "commit", round(commit.item(), 4), "g_adv", round(gadv.item(), 3), "g_fm", round(gfm.item(), 3), "d_loss", round(ld.item(), 3), "idle_codes_stage1", int((rvq.age[0] > 100).sum()), round(time.time() - t0), "s", flush=True)
    if s % EVERY == 0 or s == STEPS:
        ev = eval_rvq(s); rlog["evals"].append({"step": s, **ev}); show(s, ev); save(s)
print("training done")
