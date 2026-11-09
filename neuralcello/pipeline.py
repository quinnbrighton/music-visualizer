"""Inference pipeline: weights loading, encode / decode (codec), compose (symbolic model, port of gen_sym.py) and perform (notes -> curves -> latent ->
audio, port of midi_to_audio.py). Needs only this package and the exported weights in models/ (no training code, no dataset)."""
import os
import numpy as np, torch
from . import codec as C, flow as FL, symlm as SY

MODELS = os.environ.get("NEURALCELLO_MODELS", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
NC, NN = 6, 4                                                               # performance curves (pitch, voicing, periodicity, loudness, brightness, noisiness); note features


def note_feats(n):                                                          # (T,4) [in note, MIDI, onset, velocity dB] -> model features (= train_curve.note_feats)
    on = n[:, 0]; return np.stack([on, (n[:, 1] - 48) / 12 * on, n[:, 2], (n[:, 3] + 40) / 20 * on], 1).astype(np.float32)


def note_lp(nf): return (nf[..., 1] - 0.75) * nf[..., 0]                   # written pitch in curve units (log2 of f/110)
def to_abs(c, nf):                                                          # note-relative -> absolute pitch (curve models trained with RESID=1)
    c = c.clone(); c[..., 0] = (c[..., 0] + note_lp(nf)) * c[..., 1] * (nf[..., 0] > 0); return c


def clean(c):                                                               # generated curves (n,6) -> the form the latent model was trained on
    c = c.clone(); v = (c[:, 1] > 0.5).float(); c[:, 1] = v; c[:, 0] *= v; c[:, 2] = c[:, 2].clamp(0, 1)
    on = (c[:, 3] > -1).float(); c[:, 4] *= on; c[:, 5] *= on; return c


class NeuralCello:
    """Holds the four models; each loads on first use. All audio is mono float at 24 kHz."""
    SR = C.SR

    def __init__(self, device=None, models=MODELS, tf32=True):
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu")); self.dir = models; self._m = {}
        if self.dev.type == "cuda" and tf32: torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True   # as the training scripts (process-wide); fp32 weights + tf32 reproduce the original pipeline exactly

    def _load(self, f): return torch.load(os.path.join(self.dir, f), map_location="cpu", weights_only=True)

    @property
    def codec(self):
        if "codec" not in self._m:
            m = C.Codec(); m.load_state_dict(self._load("codec.pt")["model"], strict=True); self._m["codec"] = m.float().to(self.dev).eval()
        return self._m["codec"]

    def _flow(self, name, zd, nc):
        if name not in self._m:
            ck = self._load(f"{name}.pt"); a = ck["args"]; net = FL.LatFlow(ZD=zd, NC=nc, D=a["D"], LAYERS=a["LAYERS"], HEADS=a["HEADS"]); net.load_state_dict(ck["ema"], strict=True)
            self._m[name] = (net.float().to(self.dev).eval(), torch.tensor(ck["norm"]["mu"], device=self.dev), torch.tensor(ck["norm"]["sd"], device=self.dev), a)
        return self._m[name]

    @property
    def latent_model(self): return self._flow("latent", C.ZD, NC)              # (net, mu, sd, args)
    @property
    def curve_model(self): return self._flow("curve", NC, NN)

    @property
    def composer(self):
        if "composer" not in self._m:
            ck = self._load("composer.pt"); a = ck["args"]; sd = ck["model"]; sd["head.weight"] = sd["emb.weight"]          # tied output embedding (stored once)
            net = SY.SymLM(sd["emb.weight"].shape[0], a["CTX"], a["D"], a["LAYERS"], a["HEADS"]); net.load_state_dict(sd, strict=True); net.head.weight = net.emb.weight
            g = np.load(os.path.join(self.dir, "copyguard.npz")); cont = {}
            for r in g["rows"].astype(int).tolist(): cont.setdefault(tuple(r[:-1]), set()).add(r[-1])
            self._m["composer"] = (net.float().to(self.dev).eval(), a, cont, int(g["maxcopy"]))
        return self._m["composer"]

    # ---- codec ----
    @torch.no_grad()
    def encode(self, wav, sr=C.SR):
        """audio (N,) or (N,ch) at sr -> latent (N*25/sr, 24) float32 numpy at 25 Hz (resampled to 24 kHz, 20 s chunks with 1 s context)."""
        x = np.asarray(wav, np.float32); x = x.mean(1) if x.ndim == 2 else x
        if sr != C.SR:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), C.SR); x = resample_poly(x, C.SR // g, int(sr) // g).astype(np.float32)
        return self.codec.encode_long(torch.from_numpy(x).to(self.dev)).float().cpu().numpy()

    @torch.no_grad()
    def decode(self, latent, seed=0):
        """latent (n,24) at 25 Hz -> audio (n*960,) float32 at 24 kHz. 4 s windows, 1 s crossfades; seed fixes the decoder's noise (None = unseeded)."""
        z = torch.as_tensor(np.asarray(latent, np.float32), device=self.dev); return self.codec.decode_ola(z, seed=seed).astype(np.float32)

    # ---- composer ----
    @torch.no_grad()
    def compose(self, style="folk", key="D minor", meter="3/4", bars=24, bpm=58, tau=0.6, seed=0, temp=0.9, top_p=0.95, scale=0.5, eta=0.1,
                maxleaps=3, to_cello=True, out=None):
        """Sample a monophonic melody -> pretty_midi.PrettyMIDI (written to `out` if given). tau > 0: Mirostat target surprisal (bits/token), else top-p.
        Header forced; grammar mask; soft key bias `scale`; guards: <= 4 repeated pitches, no training interval run past 16 notes, <= maxleaps leaps >= a 6th in a row."""
        import pretty_midi
        net, a, CONT, MAXCOPY = self.composer; dev = self.dev; CTX = a["CTX"]
        STYLE = SY.STYLES.index(style); KEY = {SY.key_name(i): i for i in range(24)}[key]; num, den = (int(v) for v in meter.split("/")); METER = SY.meter_index(num, den); BARLEN = num * 4.0 / den
        torch.manual_seed(seed)
        tonic = KEY % 12; steps_ = [0, 2, 4, 5, 7, 9, 11] if KEY < 12 else [0, 2, 3, 5, 7, 8, 10, 11]; INKEY = {(tonic + s) % 12 for s in steps_}
        PB = torch.tensor([0.0 if (SY.PMIN + i) % 12 in INKEY else -scale for i in range(SY.PMAX - SY.PMIN + 1)], device=dev); MU = [2 * tau]

        def pick(toks, allowed, bias=None):
            x = torch.tensor([toks[-CTX:]], device=dev); lg = net(x)[0, -1].float() / temp; m = torch.full_like(lg, -1e9); m[allowed] = 0; lg = lg + m
            if bias is not None: lg[SY.OFF_P: SY.OFF_D] += bias
            p = torch.softmax(lg, -1); sp, si = torch.sort(p, descending=True)
            if tau > 0: surpr = -torch.log2(sp.clamp_min(1e-12)); keep = surpr <= MU[0]; keep[0] = True; sp = sp * keep   # Mirostat: drop tokens above the running threshold
            else: keep = torch.cumsum(sp, 0) - sp < top_p; sp = sp * keep
            j = int(torch.multinomial(sp / sp.sum(), 1)); s_obs = float(-torch.log2(p[si[j]].clamp_min(1e-12)))
            if tau > 0: MU[0] -= eta * (s_obs - tau)
            return int(si[j])

        toks = [SY.BOS, SY.OFF_ST + STYLE, SY.OFF_M + METER, SY.OFF_K + KEY, SY.OFF_BAR]; bar, pos_end = 0, 0.0; maxpos = int(round(BARLEN * 12))   # pos_end: earliest onset allowed (monophonic)
        while bar < bars:
            lo = int(round(pos_end * 12)); t = pick(toks, [SY.OFF_BAR] + [SY.OFF_POS + q for q in range(lo, min(maxpos, SY.NPOS))]); toks.append(t)
            if t == SY.OFF_BAR: bar += 1; pos_end = max(0.0, pos_end - BARLEN); continue
            pos = (t - SY.OFF_POS) / 12; pb = PB.clone(); hist = [x - SY.OFF_P for x in toks if SY.OFF_P <= x < SY.OFF_D]; last = hist[-4:]
            if len(last) == 4 and len(set(last)) == 1: pb[last[0]] = -1e9      # repetition guard
            if len(hist) >= MAXCOPY - 1:                                        # copy guard
                for nxt in CONT.get(tuple(np.diff(hist[-(MAXCOPY - 1):]).astype(int).tolist()), ()):
                    if 0 <= hist[-1] + nxt < len(pb): pb[hist[-1] + nxt] = -1e9
            if len(hist) > maxleaps and all(abs(hist[-i] - hist[-i - 1]) >= 8 for i in range(1, maxleaps + 1)):   # leap guard
                for q in range(len(pb)):
                    if abs(q - hist[-1]) >= 8: pb[q] = -1e9
            toks.append(pick(toks, list(range(SY.OFF_P, SY.OFF_D)), pb)); dt = pick(toks, list(range(SY.OFF_D, SY.OFF_BAR))); toks.append(dt)
            pos_end = pos + SY.DUR_Q[dt - SY.OFF_D]
        notes = [n for n in SY.decode(toks, BARLEN) if n[0] < bars * BARLEN]; ps = np.array([p for _, p, _ in notes])
        if to_cello:                                                            # whole-piece octave shift into C2-A4
            while np.median(ps) > 62: ps -= 12
            while np.median(ps) < 45: ps += 12
        spb = 60.0 / bpm; pm = pretty_midi.PrettyMIDI(initial_tempo=bpm); ins = pretty_midi.Instrument(42, name="cello"); pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
        for (on, _, d), p in zip(notes, ps):                                    # downbeats louder + a swell every 4 bars
            inbar = on % BARLEN; acc = 6 if inbar < 1e-6 else (3 if abs(inbar - round(inbar)) < 1e-6 else 0); phrase = 6 * np.sin(np.pi * ((on / BARLEN) % 4) / 4)
            ins.notes.append(pretty_midi.Note(int(np.clip(52 + acc + phrase, 30, 100)), int(p), 0.5 + on * spb, 0.5 + (on + d) * spb))
        pm.instruments.append(ins)
        if out: pm.write(out)
        return pm

    # ---- performer ----
    @torch.no_grad()
    def perform(self, midi, vib=0.6, seed=0, gs=2.0, cgs=1.0, vib_depth=10.0, transpose=0, seconds=None, tend=0.9, solver="euler", nstep=32, out=None, return_curves=False):
        """MIDI (path or PrettyMIDI) -> audio (n,) float32 at 24 kHz, peak 0.5 (written to `out` if given).
        notes -> curve model (500-frame windows, 200 held, guidance cgs, sampler stops at tend) -> explicit vibrato on a share `vib` of notes >= 0.6 s
        -> latent model (300-frame windows, 100 held, guidance gs) -> codec, 4 s overlap-add."""
        import pretty_midi
        pm = midi if isinstance(midi, pretty_midi.PrettyMIDI) else pretty_midi.PrettyMIDI(midi); dev = self.dev
        notes = sorted((n for i in pm.instruments if not i.is_drum for n in i.notes), key=lambda n: n.start)
        end = max(n.end for n in notes); end = min(end, seconds) if seconds else end; T = int(end * 25) + 25
        nf = np.zeros((T, 4), np.float32)                                      # [in note, MIDI, onset, velocity dB]
        for n in notes:
            s, e = int(round(n.start * 25)), max(int(round(n.end * 25)), int(round(n.start * 25)) + 1)
            if s >= T: continue
            e = min(e, T); nf[s:e, 0] = 1; nf[s:e, 1] = n.pitch + transpose; nf[s, 2] = 1; nf[s:e, 3] = (n.velocity - 127) / 2.5
        nft = torch.from_numpy(note_feats(nf)).to(dev)
        cnet, cmu, csd, ca = self.curve_model; lnet, lmu, lsd, la = self.latent_model
        curves = clean(FL.windows(cnet, nft, ca["L"], ca["L"] // 2 - 50, cgs, seed=seed, nstep=nstep, solver=solver, tend=tend) * csd + cmu)
        if ca.get("RESID"): curves = to_abs(curves, nft)
        if vib > 0:                                                            # the curve model under-produces vibrato; real: 5.8-6.6 Hz, ~10 c, starting ~0.12 s into the note
            gv = np.random.default_rng(seed); on_ = np.where(nf[:, 2] > 0)[0].tolist() + [T]; pc = curves[:, 0].clone()
            for s_, e_ in zip(on_[:-1], on_[1:]):
                if e_ - s_ < 15 or nf[s_, 0] == 0 or gv.random() > vib: continue
                tt = np.arange(e_ - s_) / 25; fade = np.clip((tt - 0.12) / 0.15, 0, 1) * np.clip((tt[-1] - tt) / 0.1, 0, 1)
                osc = vib_depth * gv.uniform(0.8, 1.2) * fade * np.sin(2 * np.pi * gv.uniform(5.8, 6.6) * tt + gv.uniform(0, 2 * np.pi))
                pc[s_:e_] += torch.from_numpy(osc / 1200).float().to(dev) * curves[s_:e_, 1]
            curves[:, 0] = pc
        z = FL.windows(lnet, curves, la["L"], 100, gs, seed=seed, nstep=nstep, solver=solver, tend=1.0) * lsd + lmu
        y = self.codec.decode_ola(z, seed=seed); y = (y / (np.abs(y).max() + 1e-6) * 0.5).astype(np.float32)
        if out:
            import soundfile as sf; sf.write(out, y, C.SR)
        return (y, curves.cpu().numpy(), z.cpu().numpy()) if return_curves else y
