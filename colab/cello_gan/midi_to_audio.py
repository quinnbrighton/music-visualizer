"""MIDI -> cello audio: notes -> expressive curves (note-conditioned curve model) -> codec latent (latent flow model) -> waveform (codec decoder).
usage: python midi_to_audio.py CODEC.pt LAT.pt CURVE.pt IN.mid OUT.wav     env: as gen_lat.py; GS (2, latent guidance) CGS (1, curve guidance)
TRANSPOSE (0 semitones) SECONDS (cap, default whole file) TEND (0.9: curve sampler stops there and jumps to t=1)"""
import os, sys
_codec, _lm, _cm, _mid, _wav = sys.argv[1:6]; sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval()
import soundfile as sf, importlib, pretty_midi
sys.argv = ["x", "x", "1"]; sys.path.insert(0, os.environ["CELLO_DIR"]); TL = importlib.import_module("train_lat"); TC = importlib.import_module("train_curve")
ck = torch.load(_lm, map_location=dev); net = TL.LatFlow().to(dev); net.load_state_dict(ck["ema"]); net.eval(); MU, SD = (torch.tensor(ck["norm"][k], device=dev) for k in ("mu", "sd"))
cc = torch.load(_cm, map_location=dev); a = cc["args"]; TC.L, TC.D, TC.LAYERS, TC.HEADS, TC.NN = a["L"], a["D"], a["LAYERS"], a["HEADS"], 4
cnet = TC.make_net(); cnet.load_state_dict(cc["ema"]); cnet.eval(); CMU, CSD = (torch.tensor(cc["norm"][k], device=dev) for k in ("mu", "sd"))
GS, CGS, TR = float(os.environ.get("GS", 2)), float(os.environ.get("CGS", 1)), int(os.environ.get("TRANSPOSE", 0)); os.environ.setdefault("TEND", "0.9")
print("latent model step", ck["step"], "| curve model step", cc["step"], flush=True)

pm = pretty_midi.PrettyMIDI(_mid); notes = sorted((n for i in pm.instruments if not i.is_drum for n in i.notes), key=lambda n: n.start)
end = max(n.end for n in notes); end = min(end, float(os.environ.get("SECONDS", end))); T = int(end * 25) + 25
nf = np.zeros((T, 4), np.float32)                                          # [in note, MIDI, onset, velocity dB] as guess_notes.py writes them
for n in notes:
    s, e = int(round(n.start * 25)), max(int(round(n.end * 25)), int(round(n.start * 25)) + 1)
    if s >= T: continue
    e = min(e, T); nf[s:e, 0] = 1; nf[s:e, 1] = n.pitch + TR; nf[s, 2] = 1; nf[s:e, 3] = (n.velocity - 127) / 2.5   # inverse of guess_notes.to_midi
nft = torch.from_numpy(TC.note_feats(nf)).to(dev)


def windows(net_, ctrl, L, hold, gs, seed=0):                              # overlapping windows, each keeping the last `hold` frames of the previous one as known context
    zd = net_.zd; n_ = ctrl.shape[0]; out = torch.zeros(0, zd, device=dev); done = 0
    while done < n_:
        h = min(hold, done); m = min(L - h, n_ - done); kn = torch.zeros(1, h + m, zd, device=dev); km = torch.zeros(1, h + m, device=dev); kn[0, :h] = out[done - h: done]; km[0, :h] = 1
        w = TL.sample(net_, ctrl[None, done - h: done + m], known=kn, kmask=km, gs=gs, seed=seed + done)[0]; out = torch.cat([out, w[h:]]); done += m
    return out


with torch.no_grad():
    curves = TC.clean(windows(cnet, nft, TC.L, TC.L // 2 - 50, CGS) * CSD + CMU)
    if a.get("RESID"): curves = TC.to_abs(curves, nft)                     # curve model predicted pitch relative to the written notes
    VIB, VDEP = float(os.environ.get("VIB", 0.15)), float(os.environ.get("VIBDEPTH", 10))   # explicit vibrato: share of notes >= 0.6 s that get it, peak depth (cents)
    if VIB > 0:                                                            # the curve models produce vibrato ~4x too rarely (they hedge); real: 6.1-6.4 Hz, ~10 c, starts ~0.12 s in
        gv = np.random.default_rng(int(os.environ.get("VSEED", 0))); on_ = np.where(nf[:, 2] > 0)[0].tolist() + [T]; pc = curves[:, 0].clone(); nv = 0
        for s_, e_ in zip(on_[:-1], on_[1:]):
            if e_ - s_ < 15 or nf[s_, 0] == 0 or gv.random() > VIB: continue
            tt = np.arange(e_ - s_) / 25; fade = np.clip((tt - 0.12) / 0.15, 0, 1) * np.clip((tt[-1] - tt) / 0.1, 0, 1)
            osc = VDEP * gv.uniform(0.8, 1.2) * fade * np.sin(2 * np.pi * gv.uniform(5.8, 6.6) * tt + gv.uniform(0, 2 * np.pi))
            pc[s_:e_] += torch.from_numpy(osc / 1200).float().to(dev) * curves[s_:e_, 1]; nv += 1
        curves[:, 0] = pc; print("explicit vibrato on %d notes (depth ~%.0f c)" % (nv, VDEP), flush=True)
    os.environ["TEND"] = "1.0"                                             # latent model keeps its usual sampler
    z = windows(net, curves, TL.L, 100, GS) * SD + MU
    # decode in 4 s windows (the training length) with 1 s crossfades: the excitation path's GroupNorm takes its statistics over the whole clip,
    # so decoding long stretches at once shifts the timbre (measured +10 dB above 6.4 kHz on a 30 s render)
    W, Hh, S = 100, 75, HOP * DSR; n_ = z.shape[0]; y = np.zeros(n_ * S); wsum = np.zeros(n_ * S); fp = torch.zeros(1, n_ * DSR); starts = list(range(0, max(1, n_ - W) + 1, Hh))
    if starts[-1] + W < n_: starts.append(n_ - W)
    for s0 in starts:
        e0 = min(n_, s0 + W); Tn = (e0 - s0) * DSR; hh = dec_feats(z[s0:e0].t()[None], Tn); fpc = model.pitch_f0(hh); yc = model.synth(hh, fpc, Tn)[0].cpu().numpy()
        w = np.ones(len(yc)); r = (W - Hh) * S
        if s0 > 0: w[:r] = np.linspace(0, 1, r)
        if e0 < n_: w[-r:] = np.linspace(1, 0, r)
        y[s0 * S: s0 * S + len(yc)] += yc * w; wsum[s0 * S: s0 * S + len(yc)] += w; fp[0, s0 * DSR: e0 * DSR] = fpc[0].cpu()
    y = y / np.maximum(wsum, 1e-6)
y = y / (np.abs(y).max() + 1e-6) * 0.5; sf.write(_wav, y, SR); np.save(_wav.replace(".wav", "_curves.npy"), curves.cpu().numpy())
fpn = fp[0].cpu().numpy()[::DSR][: T]; v = nf[: len(fpn), 0] > 0; e = np.abs(1200 * np.log2(np.clip(fpn[v], 40, 1200) / (440 * 2 ** ((nf[: len(fpn), 1][v] - 69) / 12))))
print("wrote", _wav, "| %.1f s, %d notes | audio pitch vs MIDI note: median %.1f cents (includes the recording's tuning offset and vibrato), %.0f%% within 50" % (end, len(notes), np.median(e), 100 * (e < 50).mean()))
