"""Continuity diagnostic: how jumpy are generated tokens vs real ones, across temperature / top-p / guidance.
usage: python diag_cont.py CODEC.pt TOKLM.pt OUTDIR     env: as gen_free.py"""
import os, sys
_codec, _lm, _out = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); rvq.load_state_dict(c["rvq"]); model.eval()
import soundfile as sf, importlib
sys.argv = ["x", "free", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TT = importlib.import_module("train_tok2")
lm = TT.TokLM2().to(dev); lm.load_state_dict(torch.load(_lm, map_location=dev)["model"]); lm.eval(); L = 300; DS_ = HOP * DSR; os.makedirs(_out, exist_ok=True)


@torch.no_grad()
def gen(ctrl, gs, temp, topp, seed=0):
    torch.manual_seed(seed); S = L + TT.NQ - 1; toks = torch.zeros(L, TT.NQ, dtype=torch.long, device=dev); cs_all = TT.ctrl_steps(ctrl[None], S)
    for s in range(S):
        inp, _ = TT.delay_batch(toks[None].cpu()); inp = inp[:, : s + 1].to(dev); cs = cs_all[:, : s + 1]
        lg = lm(inp, cs, torch.zeros(1, device=dev))[0, -1].float()
        if gs != 1: lu = lm(inp, cs, torch.ones(1, device=dev))[0, -1].float(); lg = lu + gs * (lg - lu)
        for q in range(TT.NQ):
            f = s - q
            if 0 <= f < L:
                p = torch.softmax(lg[q] / temp, 0)
                if topp < 1: sp_, ix = p.sort(descending=True); keep = sp_.cumsum(0) - sp_ < topp; p = torch.zeros_like(p).scatter(0, ix[keep], sp_[keep])
                toks[f, q] = torch.multinomial(p / p.sum(), 1)
    return toks


@torch.no_grad()
def teacher_pstay(tk, ctrl):                                                # model's own probability that stage 1 repeats, on real tokens
    inp, _ = TT.delay_batch(tk[None].cpu()); lg = lm(inp.to(dev), TT.ctrl_steps(ctrl[None], inp.shape[1]))[0, :L, 0].float()
    p = torch.softmax(lg, -1); return float(p[1:].gather(1, tk[:-1, :1]).mean())


def stats(tk):
    zq = sum(rvq.cb[i][tk[:, i]] for i in range(NQ)); jump = (zq[1:] - zq[:-1]).norm(dim=1).mean().item()
    same = (tk[1:, 0] == tk[:-1, 0]).float().mean().item(); same2 = (tk[1:, 1] == tk[:-1, 1]).float().mean().item(); return round(same, 3), round(same2, 3), round(jump, 3)


@torch.no_grad()
def decode(tk, f0req=None):
    zq = sum(rvq.cb[i][tk[:, i]] for i in range(NQ)).t()[None]; T = L * DSR; hh = dec_feats(zq, T); fp = model.pitch_f0(hh)
    f0 = fp if f0req is None else torch.from_numpy(np.repeat(f0req, DSR)[:T].astype(np.float32))[None].to(dev)
    y = model.synth(hh, f0, T)[0].cpu().numpy(); return y / (np.abs(y).max() + 1e-6) * 0.5


k = test[0]; f0 = 5572800 // DS_; tk = torch.from_numpy(np.load(os.environ["TOK_DIR"] + f"/tokens/{k}.npy").astype(np.int64))[f0: f0 + L].to(dev)
craw = np.load(os.environ["TOK_DIR"] + f"/controls/{k}.npy")[f0: f0 + L]; cf = torch.from_numpy(TT.ctrl_feats(craw)).to(dev)
print("real tokens  stage1 same, stage2 same, latent jump:", stats(tk), "| model p(stage1 repeats | real history) %.3f" % teacher_pstay(tk, cf), flush=True)
for gs, temp, topp in [(1, 1.0, 1.0), (3, 1.0, 1.0), (1, 0.7, 1.0), (1, 1.0, 0.8), (1, 0.7, 0.8), (3, 0.7, 0.8)]:
    g = gen(cf, gs, temp, topp); print(f"gen g{gs} t{temp} p{topp}:", stats(g), flush=True)
    sf.write(f"{_out}/maisky_g{gs}_t{temp}_p{topp}_reqpitch.wav", decode(g, craw[:, 0]), SR)
