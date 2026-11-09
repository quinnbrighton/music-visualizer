"""[v9: temporal prior: the rate is charged against a learned causal predictor of each frame from the frames before it (surprise), not against a fixed N(0,1) per frame; logs the old per-frame rate as iid for comparison]
[v6: texture distance (sd / band-correlation / roughness parts) is tracked on the full 20 s validation clips at every evaluation; texstats takes a length]
[v5: pitch is predicted from the quantised code (a CREPE-style bin head on the decoder's latent features) and drives the excitation; trained with teacher f0 mixed in (scheduled sampling over PRAMP steps). env PW (pitch loss weight), PRAMP, INZD (channels of the init checkpoint)]
[v4: ZDIM < 24 keeps the first ZDIM latent channels (whole multi-rate groups) and slices the 24-D weights it starts from]
[v3: + texture loss (TW): per-second band-level spread, band-to-band correlation and temporal roughness of the log-mel must match the real cello, without frame alignment]
[v2: learned noise path + harmonic peak/valley contrast loss; start from a trained 24-D model via INIT; env CW = contrast weight]
24-D multi-timescale bottleneck bootstrapped from a trained no-bottleneck harmonic vocoder (harm_long step 8000).
Stage A (A steps): only the new down/up layers train (distilled to the frozen teacher's 256-ch features + spectral loss). Stage B: all weights, low LR, KL ramps in.
usage: train_bn.py NAME A_STEPS B_STEPS     env: INIT (default harm_long_s8000) BETA (default 0.002) ZDIM (24) MELW (default 1) DISTILL (default 5)"""
import copy, sys
_name, _A, _Bst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = ["x", _name, "1"]
import os; os.environ.setdefault("MEL", "1"); os.environ.setdefault("ATTN", "0")
_src = open(os.path.expanduser("~/music-visualizer/v2/train_harm.py")).read(); exec(_src[: _src.index("model = Harm().to(dev);")]); exec(_src[_src.index("def batch():"): _src.index("@torch.no_grad()\ndef evaluate")])      # data, excitation, Harm, MSS, melL1, batch, feats
from ddsp_model import RATES, kl_vec
STEPS = _A + _Bst; ZD0 = int(os.environ.get("INZD", 24)); INITW = os.environ.get("INIT", "harm_long_s8000"); BETA = float(os.environ.get("BETA", 0.002)); ZD = int(os.environ.get("ZDIM", 24)); DIST = float(os.environ.get("DISTILL", 5)); MELW = float(os.environ.get("MELW", 1))

_acc, _keep = 0, []
for _n, _w in RATES:
    if _acc + _n <= ZD: _keep.append((_n, _w)); _acc += _n
assert _acc == ZD, f"ZDIM {ZD} must be a sum of whole groups {RATES}"; RATES = _keep

PER = {k: np.load(f"{B}/per/{k}.npy").astype(np.float32) for k in train + val + test}; PW = float(os.environ.get("PW", 1.0)); PRAMP = int(os.environ.get("PRAMP", 1500))
FMIN, CENT, NB = 60.0, 25.0, 209; CENTERS = torch.arange(NB, device=dev).float() * CENT
def to_cents(f0): return 1200 * torch.log2(f0.clamp(FMIN, 1200) / FMIN)
def pdecode(lg):                                                                                          # CREPE-style: local weighted mean (cents) around the argmax -> Hz
    p = torch.softmax(lg, 1); i = p.argmax(1, keepdim=True); m = (torch.arange(NB, device=lg.device)[None, :, None] - i).abs() <= 2
    c = (p * m * CENTERS[None, :, None]).sum(1) / (p * m).sum(1).clamp_min(1e-6); return FMIN * 2 ** (c / 1200)
def batch():
    ks = np.random.choice(len(train), BS, p=w); X, Fs, Ft, Pr = [], [], [], []
    for i in ks:
        k = train[i]; a = A[k]; s = 960 * random.randrange(0, (len(a) - SEG - 960) // 960); X.append(np.asarray(a[s:s + SEG], dtype=np.float32))
        Fs.append(np.asarray(Fe[k][s // 320: s // 320 + 300], dtype=np.float32)); Ft.append(F0[k][s // 240: s // 240 + 400]); Pr.append(PER[k][s // 240: s // 240 + 400])
    return [torch.from_numpy(np.stack(v)).to(dev) for v in (X, Fs, Ft, Pr)]

def kl_ar(mu, lv, m, lp, free=0.02): return (0.5 * (lp - lv + (lv.exp() + (mu - m) ** 2) / lp.exp() - 1)).mean((0, 2)).clamp_min(free)      # KL to the predicted Gaussian, per channel


class HarmBN(Harm):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv1d(D, 2 * ZD, 1); self.up = nn.Sequential(nn.Conv1d(ZD, D, 3, padding=1), nn.GELU(), nn.Conv1d(D, D, 1))
        self.ph = nn.Sequential(nn.Conv1d(D, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, NB + 1, 1)); self.nz = nn.Linear(512, NF // 2 + 1); nn.init.zeros_(self.nz.weight); nn.init.constant_(self.nz.bias, -3.0); self.pr1 = nn.Conv1d(ZD, 128, 5); self.pr2 = nn.Conv1d(128, 128, 5, dilation=2); self.pr3 = nn.Conv1d(128, 2 * ZD, 1); nn.init.zeros_(self.pr3.weight); nn.init.zeros_(self.pr3.bias)      # per-bin noise gain relative to the local spectral level

    def smooth(self, z):
        out, i = [], 0
        for n, w in RATES:
            g = z[:, i:i + n]; i += n
            out.append(g if w == 1 else F.avg_pool1d(F.pad(g, (w // 2, w // 2), mode="replicate"), w, stride=1))
        return torch.cat(out, 1)

    def encode(self, f, T):
        h, _ = self.enc(f, T); mu, lv = self.down(h).chunk(2, 1); return self.smooth(mu), self.smooth(lv).clamp(-8, 4)

    def synth(self, h, f0, T):
        n = T * HOP; E = torch.stft(excitation(f0, n), NF, HOP, window=WIN, return_complex=True)[:, :, :T]; e = self.exp(torch.log(E.abs() + 1e-3))
        bb = self.backbone(torch.cat([h, e], 1)); o = self.out(bb).transpose(1, 2); zn = self.nz(bb).transpose(1, 2).clamp(max=2.0)
        mag, dph = torch.exp(o[:, : NF // 2 + 1]).clamp(max=1e2), math.pi * torch.tanh(o[:, NF // 2 + 1:])
        S = mag * torch.exp(1j * (torch.angle(E) + dph))
        env = F.avg_pool1d(mag.transpose(1, 2), 41, 1, 20, count_include_pad=False).transpose(1, 2)                  # local spectral level (~500 Hz)
        with torch.no_grad(): Nn = torch.stft(torch.randn(h.shape[0], n, device=h.device), NF, HOP, window=WIN, return_complex=True)[:, :, :T]; Nn = Nn / Nn.abs().pow(2).mean().sqrt()   # noise made in the time domain so it survives the inverse STFT
        S = S + torch.exp(zn) * env * Nn; S = torch.cat([S, S[:, :, -1:]], 2)
        return torch.istft(S, NF, HOP, window=WIN, length=n)

    def prior_params(self, z):                                          # causal: frame t is predicted only from frames before t (about 13 frames of context); starts as N(0, 1)
        h = F.pad(z, (1, 0))[:, :, :-1]; h = F.gelu(self.pr1(F.pad(h, (4, 0)))); h = F.gelu(self.pr2(F.pad(h, (8, 0)))); m, lp = self.pr3(h).chunk(2, 1); return m, lp.clamp(-8, 4)

    def pitch_f0(self, hh): return pdecode(self.ph(hh)[:, :NB])

    def decode(self, z, f0, T):
        hh = self.up(z); return self.synth(hh, self.pitch_f0(hh) if f0 is None else f0, T)


model = HarmBN().to(dev); ck0 = torch.load(f"{B}/{INITW}.pt", map_location=dev)["model"]
if ZD < ZD0: W, bi = ck0["down.weight"], ck0["down.bias"]; ck0["down.weight"] = torch.cat([W[:ZD], W[ZD0:ZD0 + ZD]]); ck0["down.bias"] = torch.cat([bi[:ZD], bi[ZD0:ZD0 + ZD]]); ck0["up.0.weight"] = ck0["up.0.weight"][:, :ZD]
print("loaded", INITW, model.load_state_dict(ck0, strict=False), flush=True)
teacher = copy.deepcopy(model.enc).eval()
for p in teacher.parameters(): p.requires_grad_(False)
new = list(model.down.parameters()) + list(model.up.parameters()) + list(model.nz.parameters()) + list(model.ph.parameters()) + [p for n_, p in model.named_parameters() if n_.startswith("pr")]; old = [p for n_, p in model.named_parameters() if not (n_.startswith("down") or n_.startswith("up") or n_.startswith("nz") or n_.startswith("ph") or n_.startswith("pr"))]
print(len(train), "train recordings;", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params; ZDIM", ZD, "A", _A, "B", _Bst, flush=True)
LRM = float(os.environ.get("LRM", 1.0)); opt = torch.optim.AdamW([{"params": new, "lr": 1e-3 * LRM}, {"params": old, "lr": 1e-4 * LRM}], betas=(0.8, 0.95), weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-3 * LRM, 1e-4 * LRM], total_steps=STEPS, pct_start=0.05)
def set_train():
    model.train(); model.enc.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm1d) else None)                 # keep the encoder's input statistics fixed
def beta(s): return BETA                                                                                          # already at the target rate: no ramp
CW = float(os.environ.get("CW", 0.05)); W2K = torch.hann_window(2048, device=dev)
TW = float(os.environ.get("TW", 2.0)); LAGS = (1, 2, 4, 8, 16, 32)
def texstats(y, n=400):                                                        # statistics of 1-second windows of the log-mel (40 bands)
    L = torch.log(_ml(y) + 1e-5)[:, :, :n]; L = F.avg_pool1d(L.transpose(1, 2), 3, 3, ceil_mode=True).transpose(1, 2)[:, :40]; B_ = L.shape[0]
    W_ = n // 100; L = L.reshape(B_, 40, W_, 100); mu = L.mean(3, keepdim=True); c = L - mu; sd = c.pow(2).mean(3).add(1e-4).sqrt(); z = c / sd[..., None]
    corr = torch.einsum("bkwt,bjwt->bwkj", z, z) / 100                                                      # (B,4,40,40) how the bands move together
    ac = torch.stack([(z[..., :-l] * z[..., l:]).mean(3) for l in LAGS], -1)                               # (B,40,4,lags) temporal roughness / modulation
    return sd, corr, ac
def texparts(y, x, n=400):                                              # (sd, corr, roughness) L1 distances, 0 = identical texture statistics
    with torch.no_grad(): tx = texstats(x, n); ty = texstats(y, n)
    return [(a - b).abs().mean().item() for a, b in zip(ty, tx)]
def texloss(y, x):
    with torch.no_grad(): tx = texstats(x)
    ty = texstats(y); return sum((a - b).abs().mean() for a, b in zip(ty, tx)) / 3
def contrast(y, f0):                                                    # mean harmonic peak-to-valley ratio (dB) per clip, in 1-3 kHz and 3-6 kHz
    X = torch.stft(y, 2048, HOP, window=W2K, return_complex=True).abs().pow(2)[:, :, : f0.shape[1]]; bw = SR / 2048; h = torch.arange(1, 61, device=y.device)[None, :, None].float(); f = f0.clamp(40, 1200)[:, None]
    ix = lambda hz: (hz / bw).round().long().clamp(2, 1021); pk = torch.stack([X.gather(1, ix(h * f) + d) for d in (-1, 0, 1)]).max(0)[0]; va = X.gather(1, ix((h + 0.5) * f))
    lr = 10 * torch.log10(pk + 1e-8) - 10 * torch.log10(va + 1e-8); hz = h * f; ok = ((f0 > 60) & (f0 < 600))[:, None].float(); out = []
    for lo, hi in ((1000, 3000), (3000, 6000)):
        m = ((hz >= lo) & (hz < hi)).float() * ok; out.append((lr * m).sum((1, 2)) / m.sum((1, 2)).clamp_min(1))
    return torch.stack(out, 1)                                          # (B,2)


@torch.no_grad()
def evaluate(step):
    model.eval(); out = {}; kds = []
    for k in val + test:
        a = A[k]; s0 = min(len(a) // 3, 30 * SR); s0 -= s0 % 960; n = 20 * SR; x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); T = n // HOP
        f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
        mu, lv = model.encode(feats(x, f), T); zq_ = mu + torch.randn_like(mu) * (0.5 * lv).exp(); pm_, pl_ = model.prior_params(zq_); kd = kl_ar(mu, lv, pm_, pl_, 0.0); r = {"bits_per_s": kd.sum().item() * 100 / math.log(2), "bits_iid": kl_vec(mu, lv, 0.0).sum().item() * 100 / math.log(2)}; kds.append(kd.cpu().numpy() * 100 / math.log(2)) if k != test[0] else None
        for tag, z in (("samp", mu + torch.randn_like(mu) * (0.5 * lv).exp()), ("mean", mu)):
            y = model.decode(z, None, T); d = logmel_dist(y[0].cpu(), x[0].cpu()); r[tag + "_mss"] = MSS(y, x).item(); r[tag + "_lm"] = d["lm"]; tp = texparts(y, x, 2000); r[tag + "_tex"] = float(np.mean(tp)); r[tag + "_texsd"], r[tag + "_texcorr"], r[tag + "_texrough"] = tp; fp_ = model.pitch_f0(model.up(z)); e_ = (to_cents(fp_) - to_cents(f0))[torch.from_numpy(PER[k][s0 // 240: s0 // 240 + T]).to(dev)[None] > 0.5].abs(); r[tag + "_cents"] = e_.median().item(); r[tag + "_w50"] = (e_ < 50).float().mean().item()
        out[k[:8]] = r
    import soundfile as sf; k = test[0]; s0 = 5572800; n = 15 * SR; T = n // HOP; x = torch.from_numpy(np.asarray(A[k][s0:s0 + n], dtype=np.float32))[None].to(dev)
    f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
    mu, lv = model.encode(feats(x, f), T)
    for tag, z in (("samp", mu + torch.randn_like(mu) * (0.5 * lv).exp()), ("mean", mu)): sf.write(f"{B}/{name}_{tag}_s{step}_cmp.wav", model.decode(z, None, T)[0].cpu().numpy(), SR)
    set_train(); ms = tuple(next(iter(out.values())).keys()); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ms}; out["kl_bits_per_s_by_dim"] = np.mean(kds, 0).round(1).tolist(); return out


log = {"args": dict(INIT=INITW, A=_A, B=_Bst, BETA=BETA, ZD=ZD), "evals": []}; t0 = time.time(); set_train()
for s in range(1, STEPS + 1):
    stageA = s <= _A
    for p in old: p.requires_grad_(not stageA)
    x, f, f0, pr = batch(); ff = feats(x, f); T = 400
    h, _ = model.enc(ff, T); mu, lv = model.down(h).chunk(2, 1); mu, lv = model.smooth(mu), model.smooth(lv).clamp(-8, 4); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    hh = model.up(z)
    with torch.no_grad(): ht, _ = teacher(ff, T)
    lg = model.ph(hh); pl, pv = lg[:, :NB], lg[:, NB]; fp = pdecode(pl); vm = (pr > 0.5).float()
    tg = torch.exp(-0.5 * ((CENTERS[None, :, None] - to_cents(f0)[:, None]) / CENT) ** 2); tg = tg / tg.sum(1, keepdim=True)
    pce = ((-(tg * F.log_softmax(pl, 1)).sum(1)) * vm).sum() / vm.sum().clamp_min(1); pper = F.binary_cross_entropy_with_logits(pv, pr.clamp(0, 1)); pcents = (to_cents(fp) - to_cents(f0)).abs()[vm > 0].median()
    use = (torch.rand(f0.shape[0], 1, device=dev) < min(1.0, s / PRAMP)).float(); f0u = use * fp.detach() + (1 - use) * f0            # scheduled sampling: teacher f0 early, own pitch later
    y = model.synth(hh, f0u, T); pm_, pl_ = model.prior_params(z); kv = kl_ar(mu, lv, pm_, pl_)
    dist = F.mse_loss(hh, ht) / ht.var().clamp_min(1e-6); rec = MSS(y, x) + MELW * melL1(y, x)
    with torch.no_grad(): cx = contrast(x, f0)
    cy = contrast(y, f0); cgap = (cy - cx).abs().mean(); tex = texloss(y, x); loss = rec + CW * cgap + TW * tex + PW * (pce + 0.3 * pper) + DIST * dist * (1.0 if stageA else 0.2) + beta(s) * kv.sum()
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if s % 250 == 0: print(s, "A" if stageA else "B", "rec", round(rec.item(), 3), "dist", round(dist.item(), 3), "tex", round(tex.item(), 3), "pitchCE", round(pce.item(), 3), "pitch_cents", round(pcents.item(), 1), "cgap_dB", round(cgap.item(), 2), "cdiff_dB", [round(v, 1) for v in (cy - cx).mean(0).tolist()], "kl", round(kv.sum().item(), 2), "bits/s~", round(kv.sum().item() * 100 / math.log(2)), "iid_bits/s~", round(kl_vec(mu, lv).sum().item() * 100 / math.log(2)), round(time.time() - t0), "s", flush=True)
    if s % 2000 == 0 or s == STEPS:
        ev = evaluate(s); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m: round(v, 4) for m, v in ev["val_mean"].items()}, "test", {m: round(ev[test[0][:8]][m], 4) for m in ("samp_mss", "samp_lm", "mean_mss", "bits_per_s")}, "\n   bits/s by dim (groups 6|5|5|4|4):", ev["kl_bits_per_s_by_dim"], flush=True)
        torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}_s{s}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
print("training done")
