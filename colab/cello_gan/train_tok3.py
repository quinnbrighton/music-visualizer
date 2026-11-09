"""Token transformer v3 = v2 + "plan B" constraints, so the controls actually steer generation:
- pitch-match loss: the predicted token distributions are sampled (straight-through Gumbel), decoded through the FROZEN codec's latent->pitch
  head, and penalised (cross-entropy over 25-cent bins) against the requested pitch on voiced frames (PW, ramped in over PRAMP steps)
- frames near note changes (pitch jump > 50 cents, voicing flip, loudness rise > 6 dB; +-2 frames) count WON times more in the token loss
- extra controls: brightness (log spectral centroid) and noisiness (spectral flatness) from controls2/ (extra_controls.py)
- history corruption default 0
usage: python train_tok3.py NAME STEPS CODEC.pt     env: as train_tok2.py plus PW (0.5) PRAMP (1000) WON (4) PB (16 windows for the pitch loss)"""
import os, sys, math, json, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
name, STEPS = sys.argv[1], int(sys.argv[2]); TOK = os.environ["TOK_DIR"]; OUT = os.environ.get("OUT", TOK); dev = "cuda"; os.makedirs(OUT, exist_ok=True)
NQ, K = int(os.environ.get("NQ", 12)), int(os.environ.get("K", 64)); BOS = K
BS, CTX, D, LAYERS, HEADS = int(os.environ.get("BS", 64)), int(os.environ.get("CTX", 300)), int(os.environ.get("D", 256)), int(os.environ.get("LAYERS", 6)), int(os.environ.get("HEADS", 4))
LR, CDROP, CORRUPT, DROPOUT, WD = float(os.environ.get("LR", 5e-4)), float(os.environ.get("CDROP", 0.1)), float(os.environ.get("CORRUPT", 0.0)), float(os.environ.get("DROPOUT", 0.2)), float(os.environ.get("WD", 0.1))
EVERY, PATIENCE = int(os.environ.get("EVERY", 500)), int(os.environ.get("PATIENCE", 6))
PW, PRAMP, WON, PB = float(os.environ.get("PW", 0.5)), int(os.environ.get("PRAMP", 1000)), float(os.environ.get("WON", 4)), int(os.environ.get("PB", 16))
SS, SSRAMP = float(os.environ.get("SS", 0.0)), int(os.environ.get("SSRAMP", 3000))                         # scheduled sampling fraction, ramped in over SSRAMP steps
NC = 6; DSR = 4; FMIN, CENT, NB = 60.0, 25.0, 209
SW = torch.tensor([1.0 / (1 + 0.15 * q) for q in range(NQ)]); SW = (SW / SW.mean()).to(dev)


def ctrl_feats(c, c2):                                                      # (T,3) pitch Hz, periodicity, loudness dB + (T,2) log2 centroid/500, log10 flatness -> (T,6)
    v = (c[:, 1] > 0.5).astype(np.float32); lp = np.log2(np.clip(c[:, 0], 40, 1200) / 110.0) * v; on = (c[:, 2] > -60).astype(np.float32)
    return np.stack([lp, v, c[:, 1], (c[:, 2] + 40) / 20, c2[:, 0] * on, (c2[:, 1] + 3) / 0.7 * on], 1).astype(np.float32)


def onset_w(c):                                                             # per-frame loss weight: WON near note changes, else 1
    v = c[:, 1] > 0.5; ce = 1200 * np.log2(np.clip(c[:, 0], 40, 1200)); m = np.zeros(len(c), bool)
    m[1:] |= (np.abs(np.diff(ce)) > 50) & v[1:] & v[:-1]; m[1:] |= v[1:] != v[:-1]; m[1:] |= np.diff(c[:, 2]) > 6
    m = np.convolve(m.astype(float), np.ones(5), "same") > 0; return np.where(m, WON, 1.0).astype(np.float32)


def pbins(c):                                                               # requested pitch -> 25-cent bin, -100 where unvoiced (25 Hz frames)
    b = np.clip(np.round(1200 * np.log2(np.clip(c[:, 0], FMIN, 1200) / FMIN) / CENT), 0, NB - 1).astype(np.int64); b[c[:, 1] <= 0.5] = -100; return b


def load(keys):
    out = {}
    for k in keys:
        t = np.load(f"{TOK}/tokens/{k}.npy").astype(np.int64); c = np.load(f"{TOK}/controls/{k}.npy"); c2 = np.load(f"{TOK}/controls2/{k}.npy"); n = min(len(t), len(c), len(c2))
        out[k] = (t[:n], ctrl_feats(c[:n], c2[:n]), onset_w(c[:n]), pbins(c[:n]))
    return out


def delay_batch(toks):                                                      # toks (B,L,NQ) -> inputs (B,S,NQ), targets (B,S,NQ) with -100 where invalid; S = L + NQ - 1
    B_, L, _ = toks.shape; S = L + NQ - 1; inp = torch.full((B_, S, NQ), BOS, dtype=torch.long); tgt = torch.full((B_, S, NQ), -100, dtype=torch.long)
    for q in range(NQ):
        tgt[:, q:q + L, q] = toks[:, :, q]; inp[:, q + 1:q + 1 + L, q] = toks[:, :L, q][:, : S - q - 1]
    return inp, tgt


def delay_w(w):                                                             # per-frame weights (B,L) -> (B,S,NQ) aligned with the delayed targets
    B_, L = w.shape; out = torch.ones(B_, L + NQ - 1, NQ)
    for q in range(NQ): out[:, q:q + L, q] = w
    return out


def undelay(lg, L): return torch.stack([lg[:, q:q + L, q] for q in range(NQ)], 2)          # (B,S,NQ,K) -> (B,L,NQ,K): all stages of each frame
def ctrl_steps(ctrl, S): B_, L, C = ctrl.shape; return torch.cat([ctrl, ctrl[:, -1:].expand(B_, S - L, C)], 1)


def batch(data, keys, ws, n, L):
    T_, C_, W_, P_ = [], [], [], []
    for _ in range(n):
        k = keys[np.random.choice(len(keys), p=ws)]; s = random.randrange(0, len(data[k][0]) - L); d = data[k]
        T_.append(d[0][s:s + L]); C_.append(d[1][s:s + L]); W_.append(d[2][s:s + L]); P_.append(d[3][s:s + L])
    inp, tgt = delay_batch(torch.from_numpy(np.stack(T_)))
    return inp.to(dev), tgt.to(dev), ctrl_steps(torch.from_numpy(np.stack(C_)), inp.shape[1]).to(dev), delay_w(torch.from_numpy(np.stack(W_))).to(dev), torch.from_numpy(np.stack(P_)).to(dev)


class TokLM3(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(K + 1, D) for _ in range(NQ)]); self.pos = nn.Parameter(torch.randn(1, CTX + NQ + 8, D) * 0.02)
        self.cond = nn.Sequential(nn.Conv1d(NC + 1, D, 9, padding=4), nn.GELU(), nn.Conv1d(D, D, 9, padding=4))      # last channel = "controls dropped"
        self.film = nn.ModuleList([nn.Linear(D, D) for _ in range(LAYERS)])
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(D, HEADS, 4 * D, DROPOUT, activation="gelu", batch_first=True, norm_first=True) for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D); self.heads = nn.ModuleList([nn.Linear(D, K) for _ in range(NQ)])

    def forward(self, inp, ctrl, drop=None):                                # inp (B,S,NQ), ctrl (B,S,NC) -> logits (B,S,NQ,K)
        B_, S, _ = inp.shape
        if drop is None: drop = torch.zeros(B_, device=inp.device)
        c = self.cond(torch.cat([ctrl * (1 - drop)[:, None, None], drop[:, None, None].expand(B_, S, 1)], 2).transpose(1, 2)).transpose(1, 2)
        h = sum(e(inp[:, :, q]) for q, e in enumerate(self.emb)) + self.pos[:, :S] + c; mask = nn.Transformer.generate_square_subsequent_mask(S, device=inp.device)
        for f, l in zip(self.film, self.layers): h = l(h + f(c), src_mask=mask, is_causal=True)
        return torch.stack([hd(self.norm(h)) for hd in self.heads], 2)


class CodecPitch(nn.Module):                                                # frozen piece of the codec: tokens -> 24-ch latent -> decoder features -> pitch-bin logits (100 Hz)
    def __init__(self, path):
        super().__init__(); c = torch.load(path, map_location="cpu", weights_only=False); sd = c["model"]
        self.up = nn.Sequential(nn.Conv1d(24, 256, 3, padding=1), nn.GELU(), nn.Conv1d(256, 256, 1)); self.ph = nn.Sequential(nn.Conv1d(256, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, NB + 1, 1))
        self.up.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith("up.")}); self.ph.load_state_dict({k[3:]: v for k, v in sd.items() if k.startswith("ph.")})
        self.register_buffer("cb", c["rvq"]["cb"].float()); self.requires_grad_(False); self.eval()

    def forward(self, onehot):                                              # (B,L,NQ,K) one-hot / soft -> (B,NB,4L) logits
        zq = torch.einsum("blqk,qkd->bdl", onehot.float(), self.cb); zq = F.interpolate(zq, size=zq.shape[2] * DSR, mode="linear", align_corners=False)
        return self.ph(self.up(zq))[:, :NB]


def pitch_loss(cp, lg, pb, hard=True):                                      # lg (B,S,NQ,K) logits, pb (B,L) bins -> CE, fraction of voiced frames within 50 cents
    L = pb.shape[1]; u = undelay(lg, L).float(); oh = F.gumbel_softmax(u, tau=1.0, hard=hard, dim=-1); pl = cp(oh)
    t = pb.repeat_interleave(DSR, 1); ce = F.cross_entropy(pl, t, ignore_index=-100)
    with torch.no_grad(): v = t != -100; acc = ((pl.argmax(1) - t).abs() <= 2)[v].float().mean()
    return ce, float(acc)


if __name__ == "__main__":
    sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu"); train = [k for k in sp["train"] if os.path.exists(f"{TOK}/tokens/{k}.npy")]; val = sp["val"]
    TR, VA = load(train), load(val); w = np.array([len(TR[k][0]) for k in train], float); w /= w.sum(); cp = CodecPitch(sys.argv[3]).to(dev)
    model = TokLM3().to(dev)
    if os.environ.get("INITLM"): model.load_state_dict(torch.load(os.environ["INITLM"], map_location=dev)["model"]); print("initialised from", os.environ["INITLM"], flush=True)
    print(len(TR), "train recordings | NQ", NQ, "K", K, "| params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), "| PW", PW, "WON", WON, "CORRUPT", CORRUPT, flush=True)
    opt = torch.optim.AdamW(model.parameters(), LR, betas=(0.9, 0.95), weight_decay=WD); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.03)
    vkeys = list(VA); vw = np.array([len(VA[k][0]) for k in vkeys], float); vw /= vw.sum(); random.seed(0); np.random.seed(0)
    VB = [batch(VA, vkeys, vw, 16, CTX) for _ in range(8)]
    log = {"args": dict(NQ=NQ, K=K, BS=BS, CTX=CTX, D=D, LAYERS=LAYERS, HEADS=HEADS, LR=LR, CDROP=CDROP, CORRUPT=CORRUPT, DROPOUT=DROPOUT, WD=WD, PW=PW, WON=WON, NC=NC, SS=SS), "evals": []}

    def per_stage(lg, tgt, wt=None):
        l = F.cross_entropy(lg.float().reshape(-1, K), tgt.reshape(-1), ignore_index=-100, reduction="none").reshape(tgt.shape); m = (tgt != -100).float()
        if wt is not None: m = m * wt
        return (l * m).sum((0, 1)) / m.sum((0, 1)).clamp_min(1)

    @torch.no_grad()
    def evaluate():
        model.eval(); a = torch.zeros(NQ, device=dev); a0 = torch.zeros(NQ, device=dev); ao = torch.zeros(NQ, device=dev); pa, pa0 = [], []
        for inp, tgt, ctrl, wt, pb in VB:
            lg = model(inp, ctrl); lg0 = model(inp, ctrl, torch.ones(inp.shape[0], device=dev))
            a += per_stage(lg, tgt); a0 += per_stage(lg0, tgt); ao += per_stage(lg, tgt, (wt > 1).float())
            pa.append(pitch_loss(cp, lg, pb)[1]); pa0.append(pitch_loss(cp, lg0, pb)[1])
        model.train(); n = len(VB); f = lambda x: (x / n / math.log(2)).tolist()
        return f(a), f(a0), f(ao), float(np.mean(pa)), float(np.mean(pa0))

    best, bad, t0 = 1e9, 0, time.time(); model.train()
    for s in range(1, STEPS + 1):
        inp, tgt, ctrl, wt, pb = batch(TR, train, w, BS, CTX); drop = (torch.rand(BS, device=dev) < CDROP).float()
        if CORRUPT > 0: m = (torch.rand(inp.shape, device=dev) < CORRUPT) & (inp != BOS); inp = inp.masked_scatter(m, torch.randint(0, K, (int(m.sum()),), device=dev))
        ss = SS * min(1.0, s / SSRAMP)
        if ss > 0:                                                          # scheduled sampling: part of the history becomes the model's own samples (its realistic mistakes)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16): smp = torch.distributions.Categorical(logits=model(inp, ctrl, drop).float()).sample()
            own = torch.full_like(inp, BOS); own[:, 1:] = smp[:, :-1]; m = (torch.rand(inp.shape, device=dev) < ss) & (inp != BOS); inp = torch.where(m, own, inp)
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(inp, ctrl, drop)
        ps = per_stage(lg, tgt, wt); loss = (ps * SW).mean(); pr = min(1.0, s / PRAMP) * PW; pce, pacc = torch.zeros((), device=dev), 0.0
        if pr > 0:
            keep = (drop[:PB] == 0)                                         # pitch loss only where the controls were given
            if keep.any(): pce, pacc = pitch_loss(cp, lg[:PB][keep], pb[:PB][keep])
        opt.zero_grad(); (loss + pr * pce).backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if s % 250 == 0: print(s, "weighted loss %.3f pitch CE %.3f (sampled tokens within 50 cents %.2f) | bits/token" % (loss.item(), float(pce), pacc), [round(v / math.log(2), 2) for v in ps.tolist()][:6], round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            b, b0, bo, pa, pa0 = evaluate(); score = float((torch.tensor(b, device=dev) * SW).mean())
            log["evals"].append({"step": s, "val_bits_per_token": b, "val_bits_no_controls": b0, "val_bits_note_changes": bo, "pitch50": pa, "pitch50_no_controls": pa0, "score": score})
            better = score < best - 1e-3
            if better: best, bad = score, 0; torch.save({"model": model.state_dict(), "args": log["args"], "step": s, "score": score}, f"{OUT}/{name}_best.pt")
            else: bad += 1
            print("EVAL", s, "val bits/token", [round(x, 2) for x in b[:4]], "-> %.0f b/s" % (25 * sum(b)), "| no controls %.0f b/s" % (25 * sum(b0)), "| at note changes %.0f b/s" % (25 * sum(bo)),
                  "| pitch within 50c: %.2f (no controls %.2f)" % (pa, pa0), "| score %.3f%s" % (score, " (best, saved)" if better else f" (no gain {bad}/{PATIENCE})"), flush=True)
            torch.save({"model": model.state_dict(), "args": log["args"], "step": s}, f"{OUT}/{name}_last.pt"); json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)
            if s % 2500 == 0: torch.save({"model": model.state_dict(), "args": log["args"], "step": s}, f"{OUT}/{name}_s{s}.pt")
            if bad >= PATIENCE: print("early stop at", s, "| best score %.3f" % best, flush=True); break
    print("training done")
