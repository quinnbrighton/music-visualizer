"""Token transformer v2 (MusicGen-style delay pattern) over the RVQ tokens, with the fixes from v1:
controls are added before EVERY layer (not only at the input), the token history is randomly corrupted (CORRUPT) so the model must read the
controls, smaller model with more dropout and weight decay, early stages weigh more in the loss, best-checkpoint saving and early stopping.
usage: python train_tok2.py NAME STEPS     env: TOK_DIR (tokens/, controls/), OUT (default TOK_DIR), CELLO_DIR (split), NQ (8) K (1024) BS (64)
CTX (300) D (256) LAYERS (6) HEADS (4) LR (5e-4) CDROP (0.1) CORRUPT (0.2) DROPOUT (0.2) WD (0.1) EVERY (500) PATIENCE (6 evals)"""
import os, sys, math, json, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
name, STEPS = sys.argv[1], int(sys.argv[2]); TOK = os.environ["TOK_DIR"]; OUT = os.environ.get("OUT", TOK); dev = "cuda"; os.makedirs(OUT, exist_ok=True)
NQ, K = int(os.environ.get("NQ", 8)), int(os.environ.get("K", 1024)); BOS = K
BS, CTX, D, LAYERS, HEADS = int(os.environ.get("BS", 64)), int(os.environ.get("CTX", 300)), int(os.environ.get("D", 256)), int(os.environ.get("LAYERS", 6)), int(os.environ.get("HEADS", 4))
LR, CDROP, CORRUPT, DROPOUT, WD = float(os.environ.get("LR", 5e-4)), float(os.environ.get("CDROP", 0.1)), float(os.environ.get("CORRUPT", 0.2)), float(os.environ.get("DROPOUT", 0.2)), float(os.environ.get("WD", 0.1))
EVERY, PATIENCE = int(os.environ.get("EVERY", 500)), int(os.environ.get("PATIENCE", 6))
sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu"); train, val = [k for k in sp["train"] if os.path.exists(f"{TOK}/tokens/{k}.npy")], sp["val"]
SW = torch.tensor([1.0 / (1 + 0.15 * q) for q in range(NQ)]); SW = (SW / SW.mean()).to(dev)                  # early stages count more
def ctrl_feats(c):                                                          # (T,3) pitch Hz, periodicity, loudness dB -> (T,4) features
    v = (c[:, 1] > 0.5).astype(np.float32); lp = np.log2(np.clip(c[:, 0], 40, 1200) / 110.0) * v
    return np.stack([lp, v, c[:, 1], (c[:, 2] + 40) / 20], 1).astype(np.float32)
def load(keys): return {k: (np.load(f"{TOK}/tokens/{k}.npy").astype(np.int64), ctrl_feats(np.load(f"{TOK}/controls/{k}.npy"))) for k in keys}
def delay_batch(toks):                                                      # toks (B,L,NQ) -> inputs (B,S,NQ), targets (B,S,NQ) with -100 where invalid; S = L + NQ - 1
    B_, L, _ = toks.shape; S = L + NQ - 1; inp = torch.full((B_, S, NQ), BOS, dtype=torch.long); tgt = torch.full((B_, S, NQ), -100, dtype=torch.long)
    for q in range(NQ):
        tgt[:, q:q + L, q] = toks[:, :, q]; inp[:, q + 1:q + 1 + L, q] = toks[:, :L, q][:, : S - q - 1]
    return inp, tgt
def ctrl_steps(ctrl, S): B_, L, C = ctrl.shape; return torch.cat([ctrl, ctrl[:, -1:].expand(B_, S - L, C)], 1)
def batch(data, keys, ws, n, L):
    T_, C_ = [], []
    for _ in range(n):
        k = keys[np.random.choice(len(keys), p=ws)]; s = random.randrange(0, len(data[k][0]) - L); T_.append(data[k][0][s:s + L]); C_.append(data[k][1][s:s + L])
    inp, tgt = delay_batch(torch.from_numpy(np.stack(T_))); return inp.to(dev), tgt.to(dev), ctrl_steps(torch.from_numpy(np.stack(C_)), inp.shape[1]).to(dev)


class TokLM2(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(K + 1, D) for _ in range(NQ)]); self.pos = nn.Parameter(torch.randn(1, CTX + NQ + 8, D) * 0.02)
        self.cond = nn.Sequential(nn.Conv1d(5, D, 9, padding=4), nn.GELU(), nn.Conv1d(D, D, 9, padding=4))           # +-0.2 s of controls around each step; channel 5 = "controls dropped"
        self.film = nn.ModuleList([nn.Linear(D, D) for _ in range(LAYERS)])                                          # controls re-injected before every layer
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(D, HEADS, 4 * D, DROPOUT, activation="gelu", batch_first=True, norm_first=True) for _ in range(LAYERS)])
        self.norm = nn.LayerNorm(D); self.heads = nn.ModuleList([nn.Linear(D, K) for _ in range(NQ)])

    def forward(self, inp, ctrl, drop=None):                                # inp (B,S,NQ), ctrl (B,S,4) -> logits (B,S,NQ,K)
        B_, S, _ = inp.shape
        if drop is None: drop = torch.zeros(B_, device=inp.device)
        c = self.cond(torch.cat([ctrl * (1 - drop)[:, None, None], drop[:, None, None].expand(B_, S, 1)], 2).transpose(1, 2)).transpose(1, 2)
        h = sum(e(inp[:, :, q]) for q, e in enumerate(self.emb)) + self.pos[:, :S] + c; mask = nn.Transformer.generate_square_subsequent_mask(S, device=inp.device)
        for f, l in zip(self.film, self.layers): h = l(h + f(c), src_mask=mask, is_causal=True)
        return torch.stack([hd(self.norm(h)) for hd in self.heads], 2)


if __name__ == "__main__":
    TR, VA = load(train), load(val); w = np.array([len(TR[k][0]) for k in train], float); w /= w.sum()
    model = TokLM2().to(dev); print(len(TR), "train recordings,", int(sum(len(v[0]) for v in TR.values())), "frames | NQ", NQ, "K", K, "| params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), LR, betas=(0.9, 0.95), weight_decay=WD); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.03)
    vkeys = list(VA); vw = np.array([len(VA[k][0]) for k in vkeys], float); vw /= vw.sum(); random.seed(0); np.random.seed(0)
    VB = [batch(VA, vkeys, vw, 16, CTX) for _ in range(8)]; log = {"args": dict(NQ=NQ, K=K, BS=BS, CTX=CTX, D=D, LAYERS=LAYERS, HEADS=HEADS, LR=LR, CDROP=CDROP, CORRUPT=CORRUPT, DROPOUT=DROPOUT, WD=WD), "evals": []}
    def per_stage(lg, tgt):
        l = F.cross_entropy(lg.float().reshape(-1, K), tgt.reshape(-1), ignore_index=-100, reduction="none").reshape(tgt.shape); m = (tgt != -100).float(); return (l * m).sum((0, 1)) / m.sum((0, 1)).clamp_min(1)
    @torch.no_grad()
    def evaluate():
        model.eval(); a = torch.zeros(NQ, device=dev); a0 = torch.zeros(NQ, device=dev)
        for inp, tgt, ctrl in VB:
            a += per_stage(model(inp, ctrl), tgt); a0 += per_stage(model(inp, ctrl, torch.ones(inp.shape[0], device=dev)), tgt)
        model.train(); return (a / len(VB) / math.log(2)).tolist(), (a0 / len(VB) / math.log(2)).tolist()
    best, bad, t0 = 1e9, 0, time.time(); model.train()
    for s in range(1, STEPS + 1):
        inp, tgt, ctrl = batch(TR, train, w, BS, CTX); drop = (torch.rand(BS, device=dev) < CDROP).float()
        if CORRUPT > 0: m = (torch.rand(inp.shape, device=dev) < CORRUPT) & (inp != BOS); inp = inp.masked_scatter(m, torch.randint(0, K, (int(m.sum()),), device=dev))
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(inp, ctrl, drop)
        ps = per_stage(lg, tgt); loss = (ps * SW).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if s % 250 == 0: print(s, "weighted loss %.3f | bits/token by stage" % loss.item(), [round(v / math.log(2), 2) for v in ps.tolist()], round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            b, b0 = evaluate(); score = float((torch.tensor(b, device=dev) * SW).mean()); log["evals"].append({"step": s, "val_bits_per_token": b, "val_bits_no_controls": b0, "score": score})
            better = score < best - 1e-3
            if better: best, bad = score, 0; torch.save({"model": model.state_dict(), "args": log["args"], "step": s, "score": score}, f"{OUT}/{name}_best.pt")
            else: bad += 1
            print("EVAL", s, "val bits/token", [round(x, 2) for x in b], "-> %.0f b/s" % (25 * sum(b)), "| no controls %.0f b/s" % (25 * sum(b0)), "| score %.3f%s" % (score, " (best, saved)" if better else f" (no gain {bad}/{PATIENCE})"), flush=True)
            torch.save({"model": model.state_dict(), "args": log["args"], "step": s}, f"{OUT}/{name}_last.pt"); json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)
            if bad >= PATIENCE: print("early stop at", s, "| best score %.3f" % best, flush=True); break
    print("training done")
