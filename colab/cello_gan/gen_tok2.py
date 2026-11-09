"""[v2: uses train_tok2.TokLM2; each window is also decoded with the REQUESTED pitch fed straight to the decoder (tokens give tone, dynamics, texture); writes results.json]
Generate cello from control curves with the token transformer, decode with the RVQ codec, and check how well the pitch is followed.
For each test window: real tokens -> audio (codec ceiling), generated tokens from the window's own controls -> audio, at guidance scales GS.
usage: python gen_tok.py TOKCKPT OUTDIR     env: as tokenize_rvq.py (CELLO_DIR, RVQ_RESUME, TOK_DIR, ...), GS (comma list, default 1,3), TEMP (1.0), TOPK (0 = off)"""
import os, sys
_ck, _out = sys.argv[1], sys.argv[2]; sys.argv = ["x", "gen", "1"]
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])         # codec
import soundfile as sf, importlib
sys.argv = ["x", "gen", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TT = importlib.import_module("train_tok2")
lm = TT.TokLM2().to(dev); lm.load_state_dict(torch.load(_ck, map_location=dev)["model"]); lm.eval(); model.eval(); os.makedirs(_out, exist_ok=True)
if os.environ.get("CODEC"): _c = torch.load(os.environ["CODEC"], map_location=dev, weights_only=False); model.load_state_dict(_c["model"]); rvq.load_state_dict(_c["rvq"]); model.eval(); print("codec step", _c["step"], flush=True)
GS = [float(g) for g in os.environ.get("GS", "1,3").split(",")]; TEMP = float(os.environ.get("TEMP", 1.0)); TOPK = int(os.environ.get("TOPK", 0)); L = 300; DS_ = HOP * DSR


@torch.no_grad()
def generate(ctrl, gs, seed=0):                                             # ctrl (L,4) features -> tokens (L,NQ)
    torch.manual_seed(seed); S = L + TT.NQ - 1; toks = torch.zeros(L, TT.NQ, dtype=torch.long, device=dev); c = TT.ctrl_steps(ctrl[None], S)
    for s in range(S):
        inp, _ = TT.delay_batch(toks[None].cpu()); inp = inp[:, : s + 1].to(dev); cs = c[:, : s + 1]
        lg = lm(inp, cs, torch.zeros(1, device=dev))[0, -1].float()
        if gs != 1.0: lu = lm(inp, cs, torch.ones(1, device=dev))[0, -1].float(); lg = lu + gs * (lg - lu)
        for q in range(TT.NQ):
            f = s - q
            if 0 <= f < L:
                l = lg[q] / TEMP
                if TOPK: v, i = l.topk(TOPK); l = torch.full_like(l, -1e9).scatter(0, i, v)
                toks[f, q] = torch.multinomial(torch.softmax(l, 0), 1)
    return toks


@torch.no_grad()
def decode(toks, f0req=None):                                              # tokens (L,NQ) -> audio, decoder pitch (100 Hz); f0req (L,) Hz at 25 Hz overrides the pitch
    zq = sum(rvq.cb[i][toks[:, i]] for i in range(NQ)).t()[None]; T = L * DSR; hh = dec_feats(zq, T); fp = model.pitch_f0(hh)
    f0 = fp if f0req is None else torch.from_numpy(np.repeat(f0req, DSR)[:T].astype(np.float32))[None].to(dev)
    return model.synth(hh, f0, T)[0].cpu().numpy(), fp[0].cpu().numpy()


def pitch_err(fp, craw):                                                    # median cents between decoder pitch and requested pitch, voiced frames
    req = np.repeat(craw[:, 0], DSR)[: len(fp)]; v = np.repeat(craw[:, 1] > 0.5, DSR)[: len(fp)]
    e = np.abs(1200 * np.log2(np.clip(fp, 40, 1200) / np.clip(req, 40, 1200)))[v]; return float(np.median(e)), float((e < 50).mean())


wins = [(val[0], 30 * 25), (val[2], 60 * 25), (test[0], 5572800 // DS_)]; rows = []
for k, f0 in wins:
    tk = torch.from_numpy(np.load(f"{TOK_DIR}/tokens/{k}.npy").astype(np.int64))[f0: f0 + L].to(dev) if False else torch.from_numpy(np.load(os.environ["TOK_DIR"] + f"/tokens/{k}.npy").astype(np.int64))[f0: f0 + L].to(dev)
    craw = np.load(os.environ["TOK_DIR"] + f"/controls/{k}.npy")[f0: f0 + L]; cf = torch.from_numpy(TT.ctrl_feats(craw)).to(dev)
    x = np.asarray(A[k][f0 * DS_: (f0 + L) * DS_], np.float32); r = np.sqrt((x ** 2).mean()); sf.write(f"{_out}/{k[:8]}_original.wav", x, SR)
    y, fp = decode(tk); sf.write(f"{_out}/{k[:8]}_codec.wav", y * r / np.sqrt((y ** 2).mean()), SR); rows.append((k[:8], "codec (real tokens)", *pitch_err(fp, craw)))
    for gs in GS:
        g = generate(cf, gs); y, fp = decode(g); sf.write(f"{_out}/{k[:8]}_gen_g{gs:g}.wav", y * r / np.sqrt((y ** 2).mean()), SR)
        y2, _ = decode(g, craw[:, 0]); sf.write(f"{_out}/{k[:8]}_gen_g{gs:g}_reqpitch.wav", y2 * r / np.sqrt((y2 ** 2).mean()), SR)
        rows.append((k[:8], f"generated, guidance {gs:g} (token pitch error; the _reqpitch file uses the requested pitch)", *pitch_err(fp, craw), float((g[:, 0] == tk[:, 0]).float().mean())))
        print(rows[-1], flush=True)
for row in rows: print(row)
import json; json.dump([dict(zip(("window", "what", "median_cents", "within50", "stage1_token_agreement"), r)) for r in rows], open(f"{_out}/results.json", "w"), indent=1)
