"""Generate longer passages (sliding window) with the token transformer and decode them with the 64-entry RVQ codec.
Modes: free (controls dropped = unconditional) and conditioned on a real passage's pitch/loudness curves (decoded with token pitch and with
the requested pitch). usage: python gen_free.py CODEC.pt TOKLM.pt OUTDIR     env: CELLO_DIR, TOK_DIR, NQ (12) K (64) SECONDS (15) TEMP (1.0) GS (3)"""
import os, sys
_codec, _lm, _out = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])          # codec skeleton (tiny k-means, then overwritten)
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); rvq.load_state_dict(c["rvq"]); model.eval(); print("codec step", c["step"], flush=True)
import soundfile as sf, importlib
sys.argv = ["x", "free", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TT = importlib.import_module("train_tok2")
ck = torch.load(_lm, map_location=dev); lm = TT.TokLM2().to(dev); lm.load_state_dict(ck["model"]); lm.eval(); print("transformer step", ck.get("step"), flush=True)
SEC, TEMP, GS = float(os.environ.get("SECONDS", 15)), float(os.environ.get("TEMP", 1.0)), float(os.environ.get("GS", 3)); L = int(SEC * 25); W, HOLD = 300, 200; DS_ = HOP * DSR; os.makedirs(_out, exist_ok=True)


@torch.no_grad()
def sample_window(toks, P, ctrl, uncond, gs, seed):                         # fill frames P.. of toks (Lw,NQ) given the first P frames; ctrl (Lw,4)
    torch.manual_seed(seed); Lw = toks.shape[0]; S = Lw + TT.NQ - 1; cs_all = TT.ctrl_steps(ctrl[None], S); d1 = torch.ones(1, device=dev); d0 = torch.zeros(1, device=dev)
    for s in range(P, S):
        inp, _ = TT.delay_batch(toks[None].cpu()); inp = inp[:, : s + 1].to(dev); cs = cs_all[:, : s + 1]
        if uncond: lg = lm(inp, cs, d1)[0, -1].float()
        else:
            lg = lm(inp, cs, d0)[0, -1].float()
            if gs != 1: lu = lm(inp, cs, d1)[0, -1].float(); lg = lu + gs * (lg - lu)
        for q in range(TT.NQ):
            f = s - q
            if P <= f < Lw: toks[f, q] = torch.multinomial(torch.softmax(lg[q] / TEMP, 0), 1)
    return toks


def generate(ctrl, uncond, gs=1.0, seed=0):                                 # sliding windows of W frames, keeping HOLD frames of context
    out = torch.zeros(L, TT.NQ, dtype=torch.long, device=dev); w = sample_window(torch.zeros(min(W, L), TT.NQ, dtype=torch.long, device=dev), 0, ctrl[:W], uncond, gs, seed); out[: len(w)] = w; done = len(w)
    while done < L:
        n = min(W - HOLD, L - done); win = torch.zeros(HOLD + n, TT.NQ, dtype=torch.long, device=dev); win[:HOLD] = out[done - HOLD: done]
        win = sample_window(win, HOLD, ctrl[done - HOLD: done + n], uncond, gs, seed + done); out[done: done + n] = win[HOLD:]; done += n; print("  generated", done, "/", L, "frames", flush=True)
    return out


@torch.no_grad()
def decode(toks, f0req=None):
    zq = sum(rvq.cb[i][toks[:, i]] for i in range(NQ)).t()[None]; T = toks.shape[0] * DSR; hh = dec_feats(zq, T); fp = model.pitch_f0(hh)
    f0 = fp if f0req is None else torch.from_numpy(np.repeat(f0req, DSR)[:T].astype(np.float32))[None].to(dev)
    y = model.synth(hh, f0, T)[0].cpu().numpy(); return y / (np.abs(y).max() + 1e-6) * 0.5


k = test[0]; f0 = 5572800 // DS_; craw = np.load(os.environ["TOK_DIR"] + f"/controls/{k}.npy")[f0: f0 + L]; cf = torch.from_numpy(TT.ctrl_feats(craw)).to(dev)
x = np.asarray(A[k][f0 * DS_: (f0 + L) * DS_], np.float32); sf.write(f"{_out}/maisky_original_{int(SEC)}s.wav", x / (np.abs(x).max() + 1e-6) * 0.5, SR)
g = generate(torch.zeros_like(cf), uncond=True); sf.write(f"{_out}/free_unconditional_{int(SEC)}s_t{TEMP:g}.wav", decode(g), SR); print("free generation written", flush=True)
g = generate(cf, uncond=False, gs=GS); sf.write(f"{_out}/maisky_controls_{int(SEC)}s_t{TEMP:g}_g{GS:g}.wav", decode(g), SR); sf.write(f"{_out}/maisky_controls_{int(SEC)}s_t{TEMP:g}_g{GS:g}_reqpitch.wav", decode(g, craw[:, 0]), SR)
print("done", flush=True)
