"""Generate audio from the continuous latent flow model: the three 12 s test windows from their own controls (guidance GS list), decoded with
the codec's own pitch head and with the requested pitch; plus a LONG passage (SECONDS) made of overlapping windows, each continuing the last.
usage: python gen_lat.py CODEC.pt LAT.pt OUTDIR     env: as gen_free.py; GS (1,2) SECONDS (30) NSTEP (32)"""
import os, sys
_codec, _lm, _out = sys.argv[1], sys.argv[2], sys.argv[3]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval()
import soundfile as sf, importlib, json
sys.argv = ["lat", "x", "1", _codec]; sys.path.insert(0, os.environ["CELLO_DIR"]); TL = importlib.import_module("train_lat"); T3 = TL.T3
ck = torch.load(_lm, map_location=dev); net = TL.LatFlow().to(dev); net.load_state_dict(ck["ema"]); net.eval(); MU, SD = torch.tensor(ck["norm"]["mu"], device=dev), torch.tensor(ck["norm"]["sd"], device=dev)
print("latent model step", ck["step"], flush=True); GS = [float(g) for g in os.environ.get("GS", "1,2").split(",")]; L = TL.L; TD = os.environ["TOK_DIR"]; DS_ = HOP * DSR; os.makedirs(_out, exist_ok=True)


def ctrl_of(k, f0, n):
    craw = np.load(f"{TD}/controls/{k}.npy")[f0: f0 + n]; c2 = np.load(f"{TD}/controls2/{k}.npy")[f0: f0 + n]; return craw, torch.from_numpy(T3.ctrl_feats(craw, c2)).to(dev)


@torch.no_grad()
def decode(z, f0req=None):                                                  # z (n,24) raw latent at 25 Hz
    T = z.shape[0] * DSR; hh = dec_feats(z.t()[None], T); fp = model.pitch_f0(hh)
    f0 = fp if f0req is None else torch.from_numpy(np.repeat(f0req, DSR)[:T].astype(np.float32))[None].to(dev)
    y = model.synth(hh, f0, T)[0].cpu().numpy(); return y / (np.abs(y).max() + 1e-6) * 0.5, fp[0].cpu().numpy()


def pitch_err(fp, craw):
    req = np.repeat(craw[:, 0], DSR)[: len(fp)]; v = np.repeat(craw[:, 1] > 0.5, DSR)[: len(fp)]
    e = np.abs(1200 * np.log2(np.clip(fp, 40, 1200) / np.clip(req, 40, 1200)))[v]; return round(float(np.median(e)), 1), round(float((e < 50).mean()), 3)


def move(z): return round(float((z[1:] - z[:-1]).norm(dim=-1).mean()), 2)


rows = []
for k, f0 in [(val[0], 30 * 25), (val[2], 60 * 25), (test[0], 5572800 // DS_)]:
    craw, cf = ctrl_of(k, f0, L); zr = torch.from_numpy(np.load(f"{TD}/latents/{k}.npy").astype(np.float32))[f0: f0 + L].to(dev)
    y, fp = decode(zr); rows.append(dict(win=k[:8], cfg="real latent", pitch=pitch_err(fp, craw), move=move(zr))); sf.write(f"{_out}/{k[:8]}_codec.wav", y, SR)
    for gs in GS:
        z = TL.sample(net, cf[None], gs=gs)[0] * SD + MU; y, fp = decode(z); yr, _ = decode(z, craw[:, 0])
        sf.write(f"{_out}/{k[:8]}_g{gs:g}.wav", y, SR); sf.write(f"{_out}/{k[:8]}_g{gs:g}_reqpitch.wav", yr, SR)
        rows.append(dict(win=k[:8], cfg=f"generated g{gs:g}", pitch=pitch_err(fp, craw), move=move(z), dist=round(float(((z - zr) / SD).pow(2).mean()), 2))); print(rows[-1], flush=True)
# long passage: Maisky controls, windows of L frames, each keeping the last HOLD frames of the previous one as known context
SEC = float(os.environ.get("SECONDS", 30)); n = int(SEC * 25); HOLD = 100; k = test[0]; f0 = 5572800 // DS_; craw, cf = ctrl_of(k, f0, n); gs = GS[-1]
out = TL.sample(net, cf[None, :L], gs=gs)[0]; done = L
while done < n:
    m = min(L - HOLD, n - done); kn = torch.zeros(1, HOLD + m, TL.ZD, device=dev); kn[0, :HOLD] = out[done - HOLD: done]; km = torch.zeros(1, HOLD + m, device=dev); km[0, :HOLD] = 1
    w = TL.sample(net, cf[None, done - HOLD: done + m], known=kn, kmask=km, gs=gs, seed=done)[0]; out = torch.cat([out, w[HOLD:]]); done += m
z = out * SD + MU; y, fp = decode(z); sf.write(f"{_out}/maisky_long_{int(SEC)}s_g{gs:g}.wav", y, SR); rows.append(dict(win="maisky long", cfg=f"g{gs:g}", pitch=pitch_err(fp, craw), move=move(z)))
x = np.asarray(A[k][f0 * DS_: (f0 + n) * DS_], np.float32); sf.write(f"{_out}/maisky_long_{int(SEC)}s_original.wav", x / (np.abs(x).max() + 1e-6) * 0.5, SR)
json.dump(rows, open(f"{_out}/results.json", "w"), indent=1)
for r in rows: print(r)
