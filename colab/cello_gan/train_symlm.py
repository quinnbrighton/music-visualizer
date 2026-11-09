"""Symbolic, meter-aware note model (sym_tokens.py): causal transformer pretrained on folk tunes + Bach chorale voices + Bach cello suites, then
optionally fine-tuned on the cello suites only (STYLES=cello, INIT=pretrained). Augmentation: random transposition within range (key token
shifted). Early stopping on held-out lines (3% of lines + 3 held-out cello movements).
usage: python train_symlm.py NAME STEPS DATA.pkl     env: OUT CTX (1024) BS (16) D (384) LAYERS (8) HEADS (6) DROP (0.1) WD (0.1) LR (3e-4)
TRANS (6) STYLES (folk,chorale,cello) CELLOW (8 = sampling weight of cello lines) INIT (checkpoint to start from) EVERY (500)"""
import os, sys, math, time, json, random, pickle, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import sym_tokens as ST
CTX, BS, D, LAYERS, HEADS = (int(os.environ.get(k, d)) for k, d in (("CTX", 1024), ("BS", 16), ("D", 384), ("LAYERS", 8), ("HEADS", 6)))
DROP, WD, LR, TRANS, EVERY = float(os.environ.get("DROP", 0.1)), float(os.environ.get("WD", 0.1)), float(os.environ.get("LR", 3e-4)), int(os.environ.get("TRANS", 6)), int(os.environ.get("EVERY", 500))
dev = "cuda"; HELD = ("bwv-1008-4.mid", "bwv-1011-6.mid", "cellosuite3_3_courante.mid")


class Block(nn.Module):
    def __init__(self):
        super().__init__(); self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D); self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(DROP)); self.drop = nn.Dropout(DROP)

    def forward(self, x):
        B, T, _ = x.shape; q, k, v = self.qkv(self.n1(x)).view(B, T, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=DROP if self.training else 0.0).transpose(1, 2).reshape(B, T, D)
        x = x + self.drop(self.proj(a)); return x + self.ff(self.n2(x))


class SymLM(nn.Module):
    def __init__(self):
        super().__init__(); self.emb = nn.Embedding(ST.VOCAB, D); self.pos = nn.Parameter(torch.randn(1, CTX, D) * 0.02); self.drop = nn.Dropout(DROP)
        self.blocks = nn.ModuleList([Block() for _ in range(LAYERS)]); self.nf = nn.LayerNorm(D); self.head = nn.Linear(D, ST.VOCAB, bias=False); self.head.weight = self.emb.weight

    def forward(self, t):
        x = self.drop(self.emb(t) + self.pos[:, : t.shape[1]])
        for b in self.blocks: x = b(x)
        return self.head(self.nf(x))


def sample_tokens(item, train=True):                                        # one line -> CTX+1 tokens: BOS + header + notes from a random bar-aligned start
    style, _, arr, key, meter = item; si = ST.STYLES.index(style)
    starts = np.where(np.diff(np.concatenate([[-1], arr[:, 3]])) != 0)[0]; s0 = int(random.choice(starts[: max(1, len(starts) - 4)])) if train and len(arr) > CTX // 3 else 0
    sub = arr[s0: s0 + CTX // 3 + 8].copy(); sh = 0
    if train:
        lo, hi = int(sub[:, 2].min()), int(sub[:, 2].max()); opts = [s for s in range(-TRANS, TRANS + 1) if lo + s >= ST.PMIN and hi + s <= ST.PMAX]; sh = random.choice(opts) if opts else 0
    ctrl = ST.attrs(sub[: CTX // 3]) if (int(os.environ.get("CTRL", 0)) and (not train or random.random() > 0.2)) else None   # style controls, dropped 20% of the time
    t = [ST.BOS] + ST.encode(sub, key, meter, si, shift=sh, ctrl=ctrl); t = t[: CTX + 1]
    return t + [ST.PAD] * (CTX + 1 - len(t))


if __name__ == "__main__":
    name, STEPS, data_path = sys.argv[1], int(sys.argv[2]), sys.argv[3]; OUT = os.environ.get("OUT", "."); os.makedirs(OUT, exist_ok=True)
    data = pickle.load(open(data_path, "rb")); styles = os.environ.get("STYLES", "folk,chorale,cello").split(","); rng = random.Random(0)
    val = [d for d in data if d[1] in HELD or (d[0] != "cello" and rng.random() < 0.03)]; vs = set(id(d) for d in val); tr = [d for d in data if id(d) not in vs and d[0] in styles]
    w = np.array([len(d[2]) * (float(os.environ.get("CELLOW", 8)) if d[0] == "cello" else 1.0) for d in tr]); w /= w.sum()
    random.seed(0); VB = {s: torch.tensor([sample_tokens(d, False) for d in [x for x in val if x[0] == s][:48]]) for s in ST.STYLES if any(x[0] == s for x in val)}
    net = SymLM().to(dev)
    if os.environ.get("INIT"):                                              # tolerate a smaller vocabulary in the source checkpoint (new control tokens get fresh rows)
        sd = torch.load(os.environ["INIT"], map_location=dev)["model"]; own = net.state_dict()
        for k_, v_ in sd.items():
            if own[k_].shape != v_.shape: nv = own[k_].clone(); nv[: v_.shape[0]] = v_; sd[k_] = nv
        net.load_state_dict(sd); print("initialised from", os.environ["INIT"], flush=True)
    print(len(tr), "train lines", {s: sum(1 for d in tr if d[0] == s) for s in styles}, "| val", {s: len(v) for s, v in VB.items()}, "| params %.1fM" % (sum(p.numel() for p in net.parameters()) / 1e6), flush=True)
    dec = [p for p in net.parameters() if p.dim() >= 2]; nodec = [p for p in net.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": dec, "weight_decay": WD}, {"params": nodec, "weight_decay": 0}], LR, betas=(0.9, 0.95)); sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=STEPS, pct_start=0.05)
    log = {"args": dict(CTX=CTX, D=D, LAYERS=LAYERS, HEADS=HEADS, DROP=DROP, WD=WD, TRANS=TRANS, STYLES=styles), "evals": []}; best = 1e9; t0 = time.time()

    @torch.no_grad()
    def evaluate():
        net.eval(); r = {}
        for s, vb in VB.items():
            ls = []
            for i in range(0, len(vb), 16):
                b = vb[i: i + 16].to(dev)
                with torch.autocast("cuda", dtype=torch.bfloat16): lg = net(b[:, :-1]).float()
                ls.append(F.cross_entropy(lg.reshape(-1, ST.VOCAB), b[:, 1:].reshape(-1), ignore_index=ST.PAD).item())
            r[s] = float(np.mean(ls))
        net.train(); return r

    for s in range(1, STEPS + 1):
        b = torch.tensor([sample_tokens(tr[np.random.choice(len(tr), p=w)]) for _ in range(BS)]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16): lg = net(b[:, :-1])
        loss = F.cross_entropy(lg.float().reshape(-1, ST.VOCAB), b[:, 1:].reshape(-1), ignore_index=ST.PAD)
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sch.step()
        if s % 100 == 0: print(s, "train loss %.3f" % loss.item(), round(time.time() - t0), "s", flush=True)
        if s % EVERY == 0 or s == STEPS:
            r = evaluate(); key = r[os.environ["SELECT"]] if os.environ.get("SELECT") in r else (r.get("cello", np.mean(list(r.values()))) if "cello" in styles and len(styles) == 1 else float(np.mean(list(r.values()))))
            log["evals"].append({"step": s, **r, "train": loss.item()}); print("EVAL", s, " | ".join("%s %.3f" % kv for kv in r.items()), "| train %.3f" % loss.item(), flush=True)
            ck = {"model": net.state_dict(), "args": log["args"], "step": s, "val": r}; torch.save(ck, f"{OUT}/{name}_last.pt")
            if key < best: best = key; torch.save(ck, f"{OUT}/{name}_best.pt"); print("  new best", flush=True)
            if s % int(os.environ.get("SNAP", 10 ** 9)) == 0: torch.save(ck, f"{OUT}/{name}_s{s}.pt")
            json.dump(log, open(f"{OUT}/{name}.json", "w"), indent=1)
    print("training done")
