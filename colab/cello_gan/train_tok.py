"""Control-conditioned token transformer (MusicGen-style delay pattern) over the RVQ tokens.

Step s sees, for each stage q, the token of frame s-1-q (one-step delay per stage, so finer stages see the coarser ones of the same frame)
and predicts the token of frame s-q for every stage. Controls (pitch with voiced flag, periodicity, loudness) of the surrounding frames are
added at every step through a small non-causal conv; they are dropped CDROP of the time (classifier-free guidance).
usage: python train_tok.py NAME STEPS     env: TOK_DIR (tokens/, controls/), CELLO_DIR (for the split), BS (32) CTX (300 frames) D (384) LAYERS (8)
HEADS (6) LR (3e-4) CDROP (0.1) EVERY (1000)"""
import os, sys, math, json, time, random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
name, STEPS = sys.argv[1], int(sys.argv[2]); TOK = os.environ["TOK_DIR"]; dev = "cuda"
BS, CTX, D, LAYERS, HEADS = int(os.environ.get("BS", 32)), int(os.environ.get("CTX", 300)), int(os.environ.get("D", 384)), int(os.environ.get("LAYERS", 8)), int(os.environ.get("HEADS", 6))
LR, CDROP, EVERY = float(os.environ.get("LR", 3e-4)), float(os.environ.get("CDROP", 0.1)), int(os.environ.get("EVERY", 1000))
sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu"); train, val = [k for k in sp["train"] if os.path.exists(f"{TOK}/tokens/{k}.npy")], sp["val"]
NQ, K = 8, 1024; BOS = K
def ctrl_feats(c):                                                          # (T,3) pitch Hz, periodicity, loudness dB -> (T,4) features
    v = (c[:, 1] > 0.5).astype(np.float32); lp = np.log2(np.clip(c[:, 0], 40, 1200) / 110.0) * v
    return np.stack([lp, v, c[:, 1], (c[:, 2] + 40) / 20], 1).astype(np.float32)
def load(keys): return {k: (np.load(f"{TOK}/tokens/{k}.npy").astype(np.int64), ctrl_feats(np.load(f"{TOK}/controls/{k}.npy"))) for k in keys}
TR, VA = load(train), load(val); w = np.array([len(TR[k][0]) for k in train], float); w /= w.sum()
print(len(TR), "train recordings,", int(sum(len(v[0]) for v in TR.values())), "frames; val", int(sum(len(v[0]) for v in VA.values())), flush=True)


def window(data, k, s, L):                                                  # tokens (L,NQ) and controls (L,4) for frames s..s+L-1
    t, c = data[k]; return t[s:s + L], c[s:s + L]
def delay_batch(toks):                                                      # toks (B,L,NQ) -> inputs (B,S,NQ), targets (B,S,NQ) with -100 where invalid; S = L + NQ - 1
    B_, L, _ = toks.shape; S = L + NQ - 1; inp = torch.full((B_, S, NQ), BOS, dtype=torch.long); tgt = torch.full((B_, S, NQ), -100, dtype=torch.long)
    for q in range(NQ):
        tgt[:, q:q + L, q] = toks[:, :, q]; inp[:, q + 1:q + 1 + L, q] = toks[:, :L, q][:, : S - q - 1]
    return inp, tgt
def ctrl_steps(ctrl, S):                                                    # control of frame s at step s (padded at the end)
    B_, L, C = ctrl.shape; return torch.cat([ctrl, ctrl[:, -1:].expand(B_, S - L, C)], 1)
def batch(data, keys, ws, n, L):
    T_, C_ = [], []
    for _ in range(n):
        k = keys[np.random.choice(len(keys), p=ws)]; s = random.randrange(0, len(data[k][0]) - L); t, c = window(data, k, s, L); T_.append(t); C_.append(c)
    inp, tgt = delay_batch(torch.from_numpy(np.stack(T_))); return inp.to(dev), tgt.to(dev), ctrl_steps(torch.from_numpy(np.stack(C_)), inp.shape[1]).to(dev)


class TokLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.ModuleList([nn.Embedding(K + 1, D) for _ in range(NQ)]); self.pos = nn.Parameter(torch.randn(1, CTX + NQ + 8, D) * 0.02)
        self.cond = nn.Sequential(nn.Conv1d(5, D, 9, padding=4), nn.GELU(), nn.Conv1d(D, D, 9, padding=4))           # +-0.2 s of controls around each step; channel 5 = "controls dropped"
        layer = nn.TransformerEncoderLayer(D, HEADS, 4 * D, 0.1, activation="gelu", batch_first=True, norm_first=True); self.tf = nn.TransformerEncoder(layer, LAYERS)
        self.norm = nn.LayerNorm(D); self.heads = nn.ModuleList([nn.Linear(D, K) for _ in range(NQ)])

    def forward(self, inp, ctrl, drop=None):                                # inp (B,S,NQ), ctrl (B,S,4) -> logits (B,S,NQ,K)
        B_, S, _ = inp.shape
        if drop is None: drop = torch.zeros(B_, device=inp.device)
        c = torch.cat([ctrl * (1 - drop)[:, None, None], drop[:, None, None].expand(B_, S, 1)], 2)
        h = sum(e(inp[:, :, q]) for q, e in enumerate(self.emb)) + self.pos[:, :S] + self.cond(c.transpose(1, 2)).transpose(1, 2)
        h = self.norm(self.tf(h, mask=nn.Transformer.generate_square_subsequent_mask(S, device=inp.device), is_causal=True))
        return torch.stack([hd(h) for hd in self.heads], 2)


if __name__ == "__main__":
    model = TokLM().to(dev); print("params %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6), flush=True)
    opt = torch.optim.AdamW(model.parameters(), LR, betas=(0.9, 0.95), weight_decay=0.05); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.03)
    vkeys = list(VA); vw = np.array([len(VA[k][0]) for k in vkeys], float); vw /= vw.sum(); random.seed(0); np.random.seed(0)
    VB = [batch(VA, vkeys, vw, 16, CTX) for _ in range(8)]; log = {"args": dict(BS=BS, CTX=CTX, D=D, LAYERS=LAYERS, HEADS=HEADS, LR=LR, CDROP=CDROP), "evals": []}
    @torch.no_grad()
    def evaluate():
        model.eval(); nll = torch.zeros(NQ, device=dev); cnt = torch.zeros(NQ, device=dev); nll0 = torch.zeros(NQ, device=dev)
        for inp, tgt, ctrl in VB:
            for dd, acc in ((0.0, nll), (1.0, nll0)):
                lg = model(inp, ctrl, torch.full((inp.shape[0],), dd, device=dev)); l = F.cross_entropy(lg.reshape(-1, K), tgt.reshape(-1), ignore_index=-100, reduction="none").reshape(tgt.shape)
                m = tgt != -100; acc += (l * m).sum((0, 1)); cnt += m.sum((0, 1)) * (dd == 0)
        model.train(); b = (nll / cnt / math.log(2)).tolist(); b0 = (nll0 / cnt / math.log(2)).tolist(); return b, b0
    t0 = time.time(); model.train()
    for s in range(1, STEPS + 1):
        inp, tgt, ctrl = batch(TR, train, w, BS, CTX); drop = (torch.rand(BS, device=dev) < CDROP).float()
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = model(inp, ctrl, drop)
        loss = F.cross_entropy(lg.float().reshape(-1, K), tgt.reshape(-1), ignore_index=-100)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step()
        if s % 250 == 0: print(s, "loss %.3f (%.2f bits/token)" % (loss.item(), loss.item() / math.log(2)), round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            b, b0 = evaluate(); log["evals"].append({"step": s, "val_bits_per_token": b, "val_bits_no_controls": b0})
            print("EVAL", s, "val bits/token by stage", [round(x, 2) for x in b], "-> %.0f b/s at 8 stages" % (25 * sum(b)), "| without controls %.0f b/s" % (25 * sum(b0)), flush=True)
            torch.save({"model": model.state_dict(), "args": log["args"], "step": s}, f"{TOK}/{name}.pt"); json.dump(log, open(f"{TOK}/{name}.json", "w"), indent=1)
    print("training done")
