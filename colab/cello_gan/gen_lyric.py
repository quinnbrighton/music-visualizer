"""Lyrical, structured melody from the symbolic note model: phrase form (default A A' B A of PHRASE-bar phrases), stepwise-motion and
long-note bias, cadences (long chord-tone note ending each phrase: dominant mid-form, tonic at the end), a higher-register climax in B,
and an expressive performance map (dynamics follow the form and the melodic contour, ritardando into cadences).
usage: python gen_lyric.py SYMLM.pt OUT.mid     env: STYLE (cello) KEY ("D minor") METER ("3/4") PHRASE (4 bars) FORM ("AABA") BPM (60)
TEMP (0.85) STEP (1.0 = leap penalty strength) LONG (1.0 = short-note penalty) SEED"""
import os, sys, numpy as np, torch, pretty_midi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _ck, _out = sys.argv[1:3]
ck = torch.load(_ck, map_location="cuda"); a = ck["args"]
for k in ("CTX", "D", "LAYERS", "HEADS"): os.environ[k] = str(a[k])
os.environ["DROP"] = "0"; import sym_tokens as ST, train_symlm as TS
net = TS.SymLM().cuda(); net.load_state_dict(ck["model"]); net.eval()
STYLE = ST.STYLES.index(os.environ.get("STYLE", "cello")); KEYS = {ST.key_name(i): i for i in range(24)}; KEY = KEYS[os.environ.get("KEY", "D minor")]
num, den = (int(x) for x in os.environ.get("METER", "3/4").split("/")); BARLEN = num * 4.0 / den; PH = int(os.environ.get("PHRASE", 4)); FORM = os.environ.get("FORM", "AABA")
TEMP, STEP, LONG, BPM = float(os.environ.get("TEMP", 0.85)), float(os.environ.get("STEP", 1.0)), float(os.environ.get("LONG", 1.0)), float(os.environ.get("BPM", 60))
torch.manual_seed(int(os.environ.get("SEED", 0))); np.random.seed(int(os.environ.get("SEED", 0)))
tonic = KEY % 12; minor = KEY >= 12; scale = [0, 2, 3, 5, 7, 8, 10, 11] if minor else [0, 2, 4, 5, 7, 9, 11]; INKEY = {(tonic + s) % 12 for s in scale}
TONIC_T = {(tonic + s) % 12 for s in ([0, 3, 7] if minor else [0, 4, 7])}; DOM_T = {(tonic + s) % 12 for s in [7, 11, 2]}
NP = ST.PMAX - ST.PMIN + 1; PIT = np.arange(ST.PMIN, ST.PMAX + 1); DURS = ST.DUR_Q


@torch.no_grad()
def logits(toks):
    return net(torch.tensor([toks[-a["CTX"]:]], device="cuda"))[0, -1].float() / TEMP


def sample(lg, lo, hi, bias=None):
    l = lg[lo:hi].clone()
    if bias is not None: l = l + torch.as_tensor(bias, device="cuda", dtype=l.dtype)
    p = torch.softmax(l, -1); sp, si = torch.sort(p, descending=True); keep = torch.cumsum(sp, 0) - sp < 0.92; sp = sp * keep
    return lo + int(si[torch.multinomial(sp / sp.sum(), 1)])


def pitch_bias(prev, section, center):
    b = np.where(np.isin(PIT % 12, list(INKEY)), 0.0, -1.5)                 # stay in key
    if prev is not None:
        iv = np.abs(PIT - prev); b += -STEP * (0.35 * np.clip(iv - 2, 0, None)) - STEP * 2.0 * (iv >= 8) - 3.0 * (iv == 0) * 0.3   # prefer steps / small leaps
    if section == "B": b += 0.12 * np.clip(PIT - (center + 2), -6, 8)        # B climbs toward a climax
    b += -0.08 * np.clip(np.abs(PIT - center) - 9, 0, None)                  # keep the line in a singable register
    return b


def dur_bias(pos):
    b = -LONG * 2.0 * (DURS < 0.5) - LONG * 0.8 * (DURS < 1.0) + LONG * 0.4 * (DURS >= 1.0)   # favour quarter notes and longer
    return b - 4.0 * (pos + DURS > BARLEN + 1e-6)                            # no ties over the bar line (keeps bars tidy)


toks = [ST.BOS, ST.OFF_ST + STYLE, ST.OFF_M + ST.meter_index(num, den), ST.OFF_K + KEY]; notes = []; phrases = {}; center = 50 if STYLE == 2 else 62
prev = None; bar = -1
for si, sec in enumerate(FORM):
    final = si == len(FORM) - 1; mid = si == len(FORM) // 2 - 1; reuse = phrases.get(sec) if sec in phrases else None
    force_bars = (2 if si == 1 else PH - 1) if reuse is not None else 0   # A' keeps the first 2 bars of A; the final A keeps 3
    start_note = len(notes); start_bar = bar + 1
    for b in range(PH):
        toks.append(ST.OFF_BAR); bar += 1; pos = 0.0; last_bar = b == PH - 1
        if b < force_bars:                                                 # restate the phrase: feed the stored notes as given
            for (bb, ppos, ptok, dtok) in reuse:
                if bb != b: continue
                toks += [ST.OFF_POS + int(round(ppos * 12)), ptok, dtok]; p = ptok - ST.OFF_P + ST.PMIN; notes.append((bar * BARLEN + ppos, p, DURS[dtok - ST.OFF_D], sec, b)); prev = p
            continue
        while pos < BARLEN - 1e-6:
            q = int(round(pos * 12)); toks.append(ST.OFF_POS + q)
            if last_bar and pos >= 0:                                       # cadence: one long chord-tone note fills the rest of the phrase
                target = TONIC_T if (final or not mid) and sec != "B" else DOM_T
                if sec == "B": target = DOM_T
                if final: target = TONIC_T
                pb = pitch_bias(prev, sec, center); pb = np.where(np.isin(PIT % 12, list(target)), pb, -1e9)
                if final: pb = np.where(PIT % 12 == tonic, pb + 2.0, pb)
                pt = sample(logits(toks), ST.OFF_P, ST.OFF_D, pb); toks.append(pt)
                dt = ST.OFF_D + int(np.abs(DURS - (BARLEN - pos)).argmin()); toks.append(dt); p = pt - ST.OFF_P + ST.PMIN
                notes.append((bar * BARLEN + pos, p, BARLEN - pos, sec, b)); prev = p; pos = BARLEN; break
            pt = sample(logits(toks), ST.OFF_P, ST.OFF_D, pitch_bias(prev, sec, center)); toks.append(pt)
            dt = sample(logits(toks), ST.OFF_D, ST.OFF_BAR, dur_bias(pos)); toks.append(dt); d = min(DURS[dt - ST.OFF_D], BARLEN - pos)
            p = pt - ST.OFF_P + ST.PMIN; notes.append((bar * BARLEN + pos, p, d, sec, b)); prev = p; pos += d
    if sec not in phrases:                                                  # remember this phrase (bar-relative) for later restatements
        phrases[sec] = [(n[4], n[0] - (start_bar + n[4]) * BARLEN, ST.OFF_P + n[1] - ST.PMIN, ST.OFF_D + int(np.abs(DURS - n[2]).argmin())) for n in notes[start_note:]]

# performance map: tempo slows into each cadence; dynamics follow the form (B is the climax) and the melodic contour
ps = np.array([n[1] for n in notes])
shift = 0
while np.median(ps) + shift > 62: shift -= 12
while np.median(ps) + shift < 45: shift += 12
spb = 60.0 / BPM; tmap = []; t = 0.5; total_bars = len(FORM) * PH
for bb in range(total_bars):                                                # bar durations: ritardando in the last bar of each phrase, broader at the very end
    rit = 1.0 + (0.18 if bb % PH == PH - 1 else 0.0) + (0.25 if bb == total_bars - 1 else 0.0)
    tmap.append((t, BARLEN * spb * rit)); t += BARLEN * spb * rit
pm = pretty_midi.PrettyMIDI(initial_tempo=BPM); ins = pretty_midi.Instrument(42, name="cello"); pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
sec_level = {"A": 0, "B": 10}; pmin, pmax = ps.min(), ps.max()
for on, p, d, sec, b in notes:
    bi = int(on // BARLEN); t0, bl = tmap[bi]; s = t0 + (on - bi * BARLEN) / BARLEN * bl; e = s + d / BARLEN * bl * 0.98
    arch = 8 * np.sin(np.pi * ((on / BARLEN) % PH + 0.5) / PH); contour = 10 * (p - pmin) / max(1, pmax - pmin); acc = 3 if abs(on % 1) < 1e-6 and (on % BARLEN) < 1e-6 else 0
    v = 46 + sec_level.get(sec, 0) + arch + contour + acc
    ins.notes.append(pretty_midi.Note(int(np.clip(v, 30, 110)), int(p + shift), s, e))
pm.instruments.append(ins); pm.write(_out)
iv = np.abs(np.diff(ps)); dq = np.array([n[2] for n in notes])
print("wrote %s | form %s, %d bars of %d/%d at %d BPM (%.0f s) | %s | %d notes | steps (<=2 semitones) %.0f%%, leaps>=7 %.0f%% | median note %.2f quarters | range %d-%d" % (
    _out, FORM, total_bars, num, den, BPM, t, ST.key_name(KEY), len(notes), 100 * np.mean(iv <= 2), 100 * np.mean(iv >= 7), np.median(dq), ps.min() + shift, ps.max() + shift))
