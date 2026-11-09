"""Free generation: the curve model writes pitch/voicing/periodicity/loudness/brightness/noisiness, the latent model renders them, the codec decodes.
Makes NFREE pieces from nothing (seeds 0..) and one continuation of a real Maisky phrase (first PROMPT seconds real, the rest generated), each SECONDS long,
plus a plot of the generated pitch and loudness curves.
usage: python gen_free_lat.py CODEC.pt LAT.pt CURVE.pt OUTDIR     env: as gen_lat.py; SECONDS (40) NFREE (3) PROMPT (8) GS (2)"""
import os, sys
_codec, _lm, _cm, _out = sys.argv[1:5]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval()
import soundfile as sf, importlib, json
sys.argv = ["x", "x", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TL = importlib.import_module("train_lat"); TC = importlib.import_module("train_curve"); T3 = TL.T3
ck = torch.load(_lm, map_location=dev); net = TL.LatFlow().to(dev); net.load_state_dict(ck["ema"]); net.eval(); MU, SD = (torch.tensor(ck["norm"][k], device=dev) for k in ("mu", "sd"))
cc = torch.load(_cm, map_location=dev); a = cc["args"]; TC.L, TC.D, TC.LAYERS, TC.HEADS = a["L"], a["D"], a["LAYERS"], a["HEADS"]
cnet = TC.make_net(); cnet.load_state_dict(cc["ema"]); cnet.eval(); CMU, CSD = (torch.tensor(cc["norm"][k], device=dev) for k in ("mu", "sd"))
print("latent model step", ck["step"], "| curve model step", cc["step"], flush=True)
SEC, NFREE, PROMPT, GS = float(os.environ.get("SECONDS", 40)), int(os.environ.get("NFREE", 3)), float(os.environ.get("PROMPT", 8)), float(os.environ.get("GS", 2))
n = int(SEC * 25); TD = os.environ["TOK_DIR"]; DS_ = HOP * DSR; os.makedirs(_out, exist_ok=True)


def windows(net_, ctrl, L, hold, first_known=None, seed=0, **kw):          # long sequence from overlapping windows; each keeps the last `hold` frames of the previous as known context
    zd = net_.zd; nn_ = ctrl.shape[0]; out = torch.zeros(0, zd, device=dev); done = 0
    if first_known is not None: out = first_known; done = len(first_known)
    while done < nn_:
        h = min(hold, done); m = min(L - h, nn_ - done); kn = torch.zeros(1, h + m, zd, device=dev); km = torch.zeros(1, h + m, device=dev)
        kn[0, :h] = out[done - h: done]; km[0, :h] = 1
        w = TL.sample(net_, ctrl[None, done - h: done + m], known=kn, kmask=km, seed=seed + done, **kw)[0]; out = torch.cat([out, w[h:]]); done += m
    return out


def curves(seed, prompt=None):                                              # (n,6) cleaned feature curves; prompt = real feature curves for the start
    pk = None if prompt is None else (prompt - CMU) / CSD
    g = windows(cnet, torch.zeros(n, 0, device=dev), TC.L, TC.L // 2 - 50, pk, seed) * CSD + CMU
    return TC.clean(g)


@torch.no_grad()
def render(cf, seed=0):                                                     # feature curves -> latent (lat1, guidance GS) -> audio with the codec's own pitch
    z = windows(net, cf, TL.L, 100, None, seed, gs=GS) * SD + MU; T = z.shape[0] * DSR; hh = dec_feats(z.t()[None], T); fp = model.pitch_f0(hh)
    y = model.synth(hh, fp, T)[0].cpu().numpy(); return y / (np.abs(y).max() + 1e-6) * 0.5, fp[0].cpu().numpy()


def follow(fp, cf):                                                         # how closely the rendered audio follows the generated pitch curve
    v = np.repeat(cf[:, 1].cpu().numpy() > 0.5, DSR)[: len(fp)]; req = np.repeat(110 * 2 ** cf[:, 0].cpu().numpy(), DSR)[: len(fp)]
    e = np.abs(1200 * np.log2(np.clip(fp, 40, 1200) / req))[v]; return round(float(np.median(e)), 1), round(float((e < 50).mean()), 3)


import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
pieces = [(f"free_{i}", curves(i)) for i in range(NFREE)]
k = test[0]; f0 = 5572800 // DS_; craw = np.load(f"{TD}/controls/{k}.npy")[f0: f0 + n]; c2 = np.load(f"{TD}/controls2/{k}.npy")[f0: f0 + n]
real = torch.from_numpy(T3.ctrl_feats(craw, c2)).to(dev); P = int(PROMPT * 25); pieces.append((f"maisky_prompt{int(PROMPT)}s_continued", curves(100, real[:P])))
rows = [dict(piece="real Maisky curves", **TC.stats(real[None]))]
fig, axs = plt.subplots(len(pieces), 1, figsize=(14, 2.6 * len(pieces)), squeeze=False); tt = np.arange(n) / 25
for i, (nm, cf) in enumerate(pieces):
    y, fp = render(cf, seed=i); sf.write(f"{_out}/{nm}.wav", y, SR); rows.append(dict(piece=nm, follow=follow(fp, cf), **TC.stats(cf[None]))); print(rows[-1], flush=True)
    np.save(f"{_out}/{nm}_curves.npy", cf.cpu().numpy()); c_ = cf.cpu().numpy(); midi = np.where(c_[:, 1] > 0.5, 57 + 12 * c_[:, 0], np.nan)
    ax = axs[i, 0]; ax.plot(tt, midi, lw=1); ax.set_ylabel("MIDI pitch"); ax.set_title(nm); ax2 = ax.twinx(); ax2.plot(tt, c_[:, 3] * 20 - 40, color="0.6", lw=0.6); ax2.set_ylabel("loudness dB")
    if "prompt" in nm: ax.axvline(PROMPT, color="r", ls="--")
axs[-1, 0].set_xlabel("seconds"); fig.tight_layout(); fig.savefig(f"{_out}/curves.png", dpi=90)
x = np.asarray(A[k][f0 * DS_: (f0 + n) * DS_], np.float32); sf.write(f"{_out}/maisky_original_{int(SEC)}s.wav", x / (np.abs(x).max() + 1e-6) * 0.5, SR)
json.dump(rows, open(f"{_out}/results.json", "w"), indent=1)
for r in rows: print(r)
