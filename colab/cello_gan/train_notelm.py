"""Note-level music model (the "MIDI generator"): a small causal transformer over note-event tokens (note_tokens.py) with KEY tokens and a long
context. Anti-overfitting: random transposition (+-TRANS semitones, key tokens shifted to match), tempo stretch (+-STRETCH), dropout, weight decay,
small model, early stopping on held-out recordings (best val checkpoint kept).
usage: python train_notelm.py NAME STEPS     env: TOK_DIR CELLO_DIR OUT; CTX (2048) BS (12) D (320) LAYERS (6) HEADS (5) DROP (0.15) WD (0.1) LR (3e-4)
TRANS (6) STRETCH (0.1) EVERY (500)"""
import os, sys, math, time, json, random, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import note_tokens as NT
CTX, BS, D, LAYERS, HEADS = (int(os.environ.get(k, d)) for k, d in (("CTX", 2048), ("BS", 12), ("D", 320), ("LAYERS", 6), ("HEADS", 5)))
DROP, WD, LR, TRANS, STRETCH, EVERY = float(os.environ.get("DROP", 0.15)), float(os.environ.get("WD", 0.1)), float(os.environ.get("LR", 3e-4)), int(os.environ.get("TRANS", 6)), float(os.environ.get("STRETCH", 0.1)), int(os.environ.get("EVERY", 500))
dev = "cuda"


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D); self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(DROP)); self.drop = nn.Dropout(DROP)

    def forward(self, x):
        B, T, _ = x.shape; q, k, v = self.qkv(self.n1(x)).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=DROP if self.training else 0.0).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a)); return x + self.ff(self.n2(x))


class NoteLM(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(NT.VOCAB, D); self.pos = nn.Parameter(torch.randn(1, CTX, D) * 0.02); self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)]); self.nf = nn.LayerNorm(D); self.head = nn.Linear(D, NT.VOCAB, bias=False); self.head.weight = self.emb.weight

    def forward(self, t):
        x = self.drop(self.emb(t) + self.pos[:, : t.shape[1]])
        for b in self.blocks: x = b(x)
        return self.head(self.nf(x))


def window(ev, train=True):                                                 # random window of notes -> CTX+1 tokens (BOS first), augmented when training
    s0 = random.randrange(0, max(1, len(ev) - 64)); sub = ev[s0: s0 + CTX // 4 + 8]
    sh = 0; st = 1.0
    if train:
        lo, hi = int(sub[:, 1].min()), int(sub[:, 1].max()); opts = [s for s in range(-TRANS, TRANS + 1) if lo + s >= NT.PMIN and hi + s <= NT.PMAX]
        sh = random.choice(opts) if opts else 0; st = 1.0 + random.uniform(-STRETCH, STRETCH)
    t = [NT.BOS] + NT.encode(sub, stretch=st, shift=sh); t = t[: CTX + 1]
    return t + [NT.PAD] * (CTX + 1 - len(t))


def kind(t):                                                                # token type for per-field loss reporting
    return np.select([t < NT.OFF_P, t < NT.OFF_D, t < NT.OFF_V, t < NT.OFF_K, t < NT.PAD], [0, 1, 2, 3, 4], 5)


if __name__ == "__main__":
    name, STEPS = sys.argv[1], int(sys.argv[2]); TD = os.environ["TOK_DIR"]; OUT = os.environ.get("OUT", TD); os.makedirs(OUT, exist_ok=True)
    EVD = dict(np.load(f"{TD}/note_events.npz")); sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu")
    tr = [EVD[k] for k in sp["train"] if k in EVD]; va = [EVD[k] for k in sp["val"] if k in EVD]; w = np.array([len(e) for e in tr], float); w /= w.sum()
    random.seed(0); VB = torch.tensor([window(va[i % len(va)], False) for i in range(48)])
    net = NoteLM().to(dev); print(len(tr), "train /", len(va), "val recordings | params %.1fM | ctx %d tokens (~%d notes)" % (sum(p.numel() for p in net.parameters()) / 1e6, CTX, CTX // 4), flush=True)
    dec = [p for n, p in net.named_parameters() if p.dim() >= 2]; nodec = [p for n, p in net.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": WD}, {"params": nodec, "weight_decay": 0}], LR, betas=(0.9, 0.95))
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05); log = {"args": dict(CTX=CTX, D=D, LAYERS=LAYERS, HEADS=HEADS, DROP=DROP, WD=WD, TRANS=TRANS, STRETCH=STRETCH), "evals": []}
    best = 1e9; t0 = time.time()

    @torch.no_grad()
    def evaluate():
        net.eval(); tot = np.zeros(6); cnt = np.zeros(6)
        for i in range(0, len(VB), 12):
            b = VB[i: i + 12].to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16): lg = net(b[:, :-1]).float()
            l = F.cross_entropy(lg.reshape(-1, NT.VOCAB), b[:, 1:].reshape(-1), reduction="none").cpu().numpy(); kk = kind(b[:, 1:].reshape(-1).cpu().numpy())
            for j in range(5): tot[j] += l[kk == j].sum(); cnt[j] += (kk == j).sum()
        net.train(); per = tot[:5] / np.maximum(cnt[:5], 1); return float(tot[:5].sum() / cnt[:5].sum()), per

    for s in range(1, STEPS + 1):
        b = torch.tensor([window(tr[np.random.choice(len(tr), p=w)]) for _ in range(BS)]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = net(b[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, NT.VOCAB), b[:, 1:].reshape(-1), ignore_index=NT.PAD)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if s % 100 == 0: print(s, "train loss %.3f" % loss.item(), round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            vl, per = evaluate(); log["evals"].append({"step": s, "val": vl, "shift": per[0], "pitch": per[1], "dur": per[2], "vel": per[3], "key": per[4], "train": loss.item()})
            print("EVAL", s, "val loss %.3f (per note %.2f bits) | shift %.2f pitch %.2f dur %.2f vel %.2f key %.2f | train %.3f" % (vl, vl * 4 / math.log(2), *per, loss.item()), flush=True)
            ck = {"model": net.state_dict(), "args": log["args"], "step": s, "val": vl}; torch.save(ck, f"{OUT}/{name}_last.pt")
            if vl < best: best = vl; torch.save(ck, f"{OUT}/{name}_best.pt"); print("  new best", flush=True)
            json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("training done | best val %.3f" % best)
