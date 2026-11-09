"""[v2: learned noise path + harmonic peak/valley contrast loss; start from a trained 24-D model via INIT; env CW = contrast weight]
24-D multi-timescale bottleneck bootstrapped from a trained no-bottleneck harmonic vocoder (harm_long step 8000).
Stage A (A steps): only the new down/up layers train (distilled to the frozen teacher's 256-ch features + spectral loss). Stage B: all weights, low LR, KL ramps in.
usage: train_bn.py NAME A_STEPS B_STEPS     env: INIT (default harm_long_s8000) BETA (default 0.002) ZDIM (24) MELW (default 1) DISTILL (default 5)"""
import copy, sys
_name, _A, _Bst = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); sys.argv = ["x", _name, "1"]
import os; os.environ.setdefault("MEL", "1"); os.environ.setdefault("ATTN", "0")
_src = open(os.path.expanduser("~/music-visualizer/v2/train_harm.py")).read(); exec(_src[: _src.index("model = Harm().to(dev);")]); exec(_src[_src.index("def batch():"): _src.index("@torch.no_grad()\ndef evaluate")])      # data, excitation, Harm, MSS, melL1, batch, feats
from ddsp_model import RATES, kl_vec
STEPS = _A + _Bst; INITW = os.environ.get("INIT", "harm_long_s8000"); BETA = float(os.environ.get("BETA", 0.002)); ZD = int(os.environ.get("ZDIM", 24)); DIST = float(os.environ.get("DISTILL", 5)); MELW = float(os.environ.get("MELW", 1))


class HarmBN(Harm):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv1d(D, 2 * ZD, 1); self.up = nn.Sequential(nn.Conv1d(ZD, D, 3, padding=1), nn.GELU(), nn.Conv1d(D, D, 1))
        self.nz = nn.Linear(512, NF // 2 + 1); nn.init.zeros_(self.nz.weight); nn.init.constant_(self.nz.bias, -3.0)      # per-bin noise gain relative to the local spectral level

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

    def decode(self, z, f0, T): return self.synth(self.up(z), f0, T)


model = HarmBN().to(dev); ck0 = torch.load(f"{B}/{INITW}.pt", map_location=dev)["model"]; print("loaded", INITW, model.load_state_dict(ck0, strict=False), flush=True)
teacher = copy.deepcopy(model.enc).eval()
for p in teacher.parameters(): p.requires_grad_(False)
new = list(model.down.parameters()) + list(model.up.parameters()) + list(model.nz.parameters()); old = [p for n_, p in model.named_parameters() if not (n_.startswith("down") or n_.startswith("up") or n_.startswith("nz"))]
print(len(train), "train recordings;", round(sum(p.numel() for p in model.parameters()) / 1e6, 1), "M params; ZDIM", ZD, "A", _A, "B", _Bst, flush=True)
opt = torch.optim.AdamW([{"params": new, "lr": 1e-3}, {"params": old, "lr": 1e-4}], betas=(0.8, 0.95), weight_decay=1e-4)
sch = torch.optim.lr_scheduler.OneCycleLR(opt, [1e-3, 1e-4], total_steps=STEPS, pct_start=0.05)
def set_train():
    model.train(); model.enc.apply(lambda m: m.eval() if isinstance(m, nn.BatchNorm1d) else None)                 # keep the encoder's input statistics fixed
def beta(s): return BETA                                                                                          # already at the target rate: no ramp
CW = float(os.environ.get("CW", 0.05)); W2K = torch.hann_window(2048, device=dev)
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
        mu, lv = model.encode(feats(x, f), T); kd = kl_vec(mu, lv, 0.0); r = {"bits_per_s": kd.sum().item() * 100 / math.log(2)}; kds.append(kd.cpu().numpy() * 100 / math.log(2)) if k != test[0] else None
        for tag, z in (("samp", mu + torch.randn_like(mu) * (0.5 * lv).exp()), ("mean", mu)):
            y = model.decode(z, f0, T); d = logmel_dist(y[0].cpu(), x[0].cpu()); r[tag + "_mss"] = MSS(y, x).item(); r[tag + "_lm"] = d["lm"]
        out[k[:8]] = r
    import soundfile as sf; k = test[0]; s0 = 5572800; n = 15 * SR; T = n // HOP; x = torch.from_numpy(np.asarray(A[k][s0:s0 + n], dtype=np.float32))[None].to(dev)
    f = torch.from_numpy(np.asarray(Fe[k][s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev); f0 = torch.from_numpy(F0[k][s0 // 240: s0 // 240 + T])[None].to(dev)
    mu, lv = model.encode(feats(x, f), T)
    for tag, z in (("samp", mu + torch.randn_like(mu) * (0.5 * lv).exp()), ("mean", mu)): sf.write(f"{B}/{name}_{tag}_s{step}_cmp.wav", model.decode(z, f0, T)[0].cpu().numpy(), SR)
    set_train(); ms = tuple(next(iter(out.values())).keys()); out["val_mean"] = {m: float(np.mean([out[k][m] for k in out if k != test[0][:8]])) for m in ms}; out["kl_bits_per_s_by_dim"] = np.mean(kds, 0).round(1).tolist(); return out


log = {"args": dict(INIT=INITW, A=_A, B=_Bst, BETA=BETA, ZD=ZD), "evals": []}; t0 = time.time(); set_train()
for s in range(1, STEPS + 1):
    stageA = s <= _A
    for p in old: p.requires_grad_(not stageA)
    x, f, f0 = batch(); ff = feats(x, f); T = 400
    h, _ = model.enc(ff, T); mu, lv = model.down(h).chunk(2, 1); mu, lv = model.smooth(mu), model.smooth(lv).clamp(-8, 4); z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    hh = model.up(z)
    with torch.no_grad(): ht, _ = teacher(ff, T)
    y = model.synth(hh, f0, T); kv = kl_vec(mu, lv)
    dist = F.mse_loss(hh, ht) / ht.var().clamp_min(1e-6); rec = MSS(y, x) + MELW * melL1(y, x)
    with torch.no_grad(): cx = contrast(x, f0)
    cy = contrast(y, f0); cgap = (cy - cx).abs().mean(); loss = rec + CW * cgap + DIST * dist * (1.0 if stageA else 0.2) + beta(s) * kv.sum()
    if not torch.isfinite(loss): print("non-finite loss, skipping step", s, flush=True); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
    if s % 250 == 0: print(s, "A" if stageA else "B", "rec", round(rec.item(), 3), "dist", round(dist.item(), 3), "cgap_dB", round(cgap.item(), 2), "cdiff_dB", [round(v, 1) for v in (cy - cx).mean(0).tolist()], "kl", round(kv.sum().item(), 2), "bits/s~", round(kv.sum().item() * 100 / math.log(2)), round(time.time() - t0), "s", flush=True)
    if s % 2000 == 0 or s == STEPS:
        ev = evaluate(s); log["evals"].append({"step": s, **ev}); print("EVAL", s, {m: round(v, 4) for m, v in ev["val_mean"].items()}, "test", {m: round(ev[test[0][:8]][m], 4) for m in ("samp_mss", "samp_lm", "mean_mss", "bits_per_s")}, "\n   bits/s by dim (groups 6|5|5|4|4):", ev["kl_bits_per_s_by_dim"], flush=True)
        torch.save({"model": model.state_dict(), "args": log["args"]}, f"{B}/{name}_s{s}.pt"); json.dump(log, open(f"{B}/{name}.json", "w"), indent=1)
print("training done")
