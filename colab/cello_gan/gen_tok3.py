"""Generate the three test windows with the v3 (plan B) transformer and score pitch-following and continuity.
usage: python gen_tok3.py CODEC.pt TOKLM3.pt OUTDIR     env: as gen_free.py; CONFIGS "gs:temp_early:temp_late;..." (default 1:1:1;1:1:0.5)"""
import os, sys
_codec, _lm, _out = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); rvq.load_state_dict(c["rvq"]); model.eval()
import soundfile as sf, importlib, json
sys.argv = ["x", "free", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TT = importlib.import_module("train_tok3")
ck = torch.load(_lm, map_location=dev); lm = TT.TokLM3().to(dev); lm.load_state_dict(ck["model"]); lm.eval(); print("transformer step", ck.get("step"), flush=True)
L = 300; DS_ = HOP * DSR; TD = os.environ["TOK_DIR"]; os.makedirs(_out, exist_ok=True); EARLY = 2
CONFIGS = [tuple(float(v) for v in cfg.split(":")) for cfg in os.environ.get("CONFIGS", "1:1:1;1:1:0.5").split(";")]


@torch.no_grad()
def gen(ctrl, gs, te, tl, seed=0):                                          # temperature te for stages < EARLY, tl for the later stages
    torch.manual_seed(seed); S = L + TT.NQ - 1; toks = torch.zeros(L, TT.NQ, dtype=torch.long, device=dev); cs_all = TT.ctrl_steps(ctrl[None], S)
    temps = torch.tensor([te if q < EARLY else tl for q in range(TT.NQ)], device=dev)[:, None]
    for s in range(S):
        inp, _ = TT.delay_batch(toks[None].cpu()); inp = inp.to(dev)       # full length: the causal mask hides future tokens, and the control conv sees the real upcoming controls (as in training)
        lg = lm(inp, cs_all, torch.zeros(1, device=dev))[0, s].float()
        if gs != 1: lu = lm(inp, cs_all, torch.ones(1, device=dev))[0, s].float(); lg = lu + gs * (lg - lu)
        smp = torch.multinomial(torch.softmax(lg / temps, -1), 1)[:, 0]
        for q in range(TT.NQ):
            f = s - q
            if 0 <= f < L: toks[f, q] = smp[q]
    return toks


@torch.no_grad()
def decode(tk, f0req=None):
    zq = sum(rvq.cb[i][tk[:, i]] for i in range(NQ)).t()[None]; T = L * DSR; hh = dec_feats(zq, T); fp = model.pitch_f0(hh)
    f0 = fp if f0req is None else torch.from_numpy(np.repeat(f0req, DSR)[:T].astype(np.float32))[None].to(dev)
    y = model.synth(hh, f0, T)[0].cpu().numpy(); return y / (np.abs(y).max() + 1e-6) * 0.5, fp[0].cpu().numpy()


def pitch_err(fp, craw):
    req = np.repeat(craw[:, 0], DSR)[: len(fp)]; v = np.repeat(craw[:, 1] > 0.5, DSR)[: len(fp)]
    e = np.abs(1200 * np.log2(np.clip(fp, 40, 1200) / np.clip(req, 40, 1200)))[v]; return round(float(np.median(e)), 1), round(float((e < 50).mean()), 3)


def cont(tk):
    zq = sum(rvq.cb[i][tk[:, i]] for i in range(NQ)); return round((tk[1:, 0] == tk[:-1, 0]).float().mean().item(), 3), round((zq[1:] - zq[:-1]).norm(dim=1).mean().item(), 2)


rows = []
for k, f0 in [(val[0], 30 * 25), (val[2], 60 * 25), (test[0], 5572800 // DS_)]:
    tk = torch.from_numpy(np.load(f"{TD}/tokens/{k}.npy").astype(np.int64))[f0: f0 + L].to(dev)
    craw = np.load(f"{TD}/controls/{k}.npy")[f0: f0 + L]; c2 = np.load(f"{TD}/controls2/{k}.npy")[f0: f0 + L]; cf = torch.from_numpy(TT.ctrl_feats(craw, c2)).to(dev)
    rows.append(dict(win=k[:8], cfg="real tokens", pitch=pitch_err(decode(tk)[1], craw), cont=cont(tk)))
    for gs, te, tl in CONFIGS:
        g = gen(cf, gs, te, tl); y, fp = decode(g); yr, _ = decode(g, craw[:, 0]); tag = f"g{gs:g}_t{te:g}-{tl:g}"
        sf.write(f"{_out}/{k[:8]}_{tag}.wav", y, SR); sf.write(f"{_out}/{k[:8]}_{tag}_reqpitch.wav", yr, SR)
        rows.append(dict(win=k[:8], cfg=tag, pitch=pitch_err(fp, craw), cont=cont(g))); print(rows[-1], flush=True)
json.dump(rows, open(f"{_out}/results.json", "w"), indent=1)
for r in rows: print(r)
