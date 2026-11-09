"""Where does the generated chain's spectral imbalance come from? On the Maisky test passage, compare the band balance (relative to the original)
of: the codec on the real latent; the latent model rendering the REAL curves (several guidance/sampler settings); the latent model rendering
curves written by the curve model from the guessed notes. Also compares the curves' own brightness/noisiness with the real ones.
usage: python freq_diag.py CODEC.pt LAT.pt CURVE.pt LATENT_DIR     env: as gen_lat.py"""
import os, sys
_codec, _lm, _cm, _ld = sys.argv[1:5]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval()
import importlib
sys.argv = ["x", "x", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TL = importlib.import_module("train_lat"); TC = importlib.import_module("train_curve"); T3 = TL.T3
ck = torch.load(_lm, map_location=dev); net = TL.LatFlow().to(dev); net.load_state_dict(ck["ema"]); net.eval(); MU, SD = (torch.tensor(ck["norm"][k], device=dev) for k in ("mu", "sd"))
cc = torch.load(_cm, map_location=dev); a = cc["args"]; TC.L, TC.D, TC.LAYERS, TC.HEADS, TC.NN = a["L"], a["D"], a["LAYERS"], a["HEADS"], 4
cnet = TC.make_net(); cnet.load_state_dict(cc["ema"]); cnet.eval(); CMU, CSD = (torch.tensor(cc["norm"][k], device=dev) for k in ("mu", "sd"))
TD = os.environ["TOK_DIR"]; DS_ = HOP * DSR; k = test[0]; f0 = 5572800 // DS_; n = 300
craw = np.load(f"{TD}/controls/{k}.npy")[f0: f0 + n]; c2 = np.load(f"{TD}/controls2/{k}.npy")[f0: f0 + n]; cf = torch.from_numpy(T3.ctrl_feats(craw, c2)).to(dev)
nf = torch.from_numpy(TC.note_feats(np.load(f"{TD}/notes/{k}.npy")[f0: f0 + n])).to(dev); zr = torch.from_numpy(np.load(f"{_ld}/{k}.npy").astype(np.float32))[f0: f0 + n].to(dev)
x = np.asarray(A[k][f0 * DS_: (f0 + n) * DS_], np.float32)
bands = [(40, 100), (100, 200), (200, 400), (400, 800), (800, 1600), (1600, 3200), (3200, 6400), (6400, 12000)]


def spec(y):
    N = 4096; w = np.hanning(N); S = np.mean([np.abs(np.fft.rfft(y[i:i + N] * w)) ** 2 for i in range(0, len(y) - N, N // 2)], 0); fr = np.fft.rfftfreq(N, 1 / SR)
    b = np.array([S[(fr >= lo) & (fr < hi)].sum() for lo, hi in bands]); return b / b.sum()


@torch.no_grad()
def dec(z):
    T = z.shape[0] * DSR; hh = dec_feats(z.t()[None], T); return model.synth(hh, model.pitch_f0(hh), T)[0].cpu().numpy()


b0 = spec(x); rows = [("codec on real latent", dec(zr))]
for gs in (1.0, 2.0):
    for te in ("1.0", "0.9"):
        os.environ["TEND"] = te; rows.append((f"lat on REAL curves g{gs:g} tend{te}", dec(TL.sample(net, cf[None], gs=gs)[0] * SD + MU)))
os.environ["TEND"] = "0.9"; g = TC.clean(TL.sample(cnet, nf[None])[0] * CSD + CMU); g = TC.to_abs(g, nf) if a.get("RESID") else g
os.environ["TEND"] = "1.0"; rows.append(("lat on GENERATED curves g2", dec(TL.sample(net, g[None], gs=2.0)[0] * SD + MU)))
print("                                    " + " ".join(f"{lo}-{hi}".rjust(9) for lo, hi in bands))
for name, y in rows: print(f"{name:35s} " + " ".join(f"{10 * np.log10(v):+9.1f}" for v in spec(y) / b0))
v = (cf[:, 1] > 0.5) & (g[:, 1] > 0.5)
print("curve brightness (log2 centroid/500) real %.3f generated %.3f | noisiness real %.3f generated %.3f | loudness real %.3f generated %.3f" % (
    cf[:, 4][v].mean(), g[:, 4][v].mean(), cf[:, 5][v].mean(), g[:, 5][v].mean(), cf[:, 3].mean(), g[:, 3].mean()))
