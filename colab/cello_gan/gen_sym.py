"""Generate a melody with the meter-aware symbolic model (train_symlm.py) and write a MIDI file for midi_to_audio.py.
Header (STYLE, METER, KEY) is forced; a grammar mask keeps [BAR] POS PITCH DUR order, positions increasing inside a bar and within the bar
length, notes monophonic; optional soft bias toward the key's scale. Dynamics: downbeats slightly louder, plus a phrase-level swell every 4 bars.
Reports in-key share and how much is copied from the training lines (8-note interval patterns).
usage: python gen_sym.py SYMLM.pt OUT.mid     env: STYLE (cello) KEY ("D minor") METER ("3/4") BARS (32) TEMP (0.9) TOPP (0.95) SCALE (0.5) BPM (80)
SEED (0) TRANSPOSE_TO_CELLO (1: shift by octaves into C2-A4)"""
import os, sys, pickle, numpy as np, torch, pretty_midi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _ck, _out = sys.argv[1:3]
ck = torch.load(_ck, map_location="cuda"); a = ck["args"]
for k in ("CTX", "D", "LAYERS", "HEADS"): os.environ[k] = str(a[k])
os.environ["DROP"] = "0"; import sym_tokens as ST, train_symlm as TS
net = TS.SymLM().cuda(); net.load_state_dict(ck["model"]); net.eval(); print("symbolic model step", ck["step"], "| val", {k: round(v, 3) for k, v in ck["val"].items()}, flush=True)
STYLE = ST.STYLES.index(os.environ.get("STYLE", "cello")); KEYS = {ST.key_name(i): i for i in range(24)}; KEY = KEYS[os.environ.get("KEY", "D minor")]
num, den = (int(x) for x in os.environ.get("METER", "3/4").split("/")); METER = ST.meter_index(num, den); BARLEN = num * 4.0 / den
BARS, TEMP, TOPP, SCALE, BPM = int(os.environ.get("BARS", 32)), float(os.environ.get("TEMP", 0.9)), float(os.environ.get("TOPP", 0.95)), float(os.environ.get("SCALE", 0.5)), float(os.environ.get("BPM", 80))
torch.manual_seed(int(os.environ.get("SEED", 0)))
tonic = KEY % 12; steps_ = [0, 2, 4, 5, 7, 9, 11] if KEY < 12 else [0, 2, 3, 5, 7, 8, 10, 11]; INKEY = {(tonic + s) % 12 for s in steps_}
PB = torch.tensor([0.0 if (ST.PMIN + i) % 12 in INKEY else -SCALE for i in range(ST.PMAX - ST.PMIN + 1)], device="cuda")


TAU, ETA = float(os.environ.get("TAU", 0)), float(os.environ.get("ETA", 0.1)); MU = [2 * TAU]; SURP = []   # Mirostat: target surprisal TAU bits/token (0 = off)


@torch.no_grad()
def pick(toks, allowed, bias=None):
    x = torch.tensor([toks[-a["CTX"]:]], device="cuda"); lg = net(x)[0, -1].float() / TEMP; m = torch.full_like(lg, -1e9); m[allowed] = 0; lg = lg + m
    if bias is not None: lg[ST.OFF_P: ST.OFF_D] += bias
    p = torch.softmax(lg, -1); sp, si = torch.sort(p, descending=True)
    if TAU > 0:                                                             # keep only tokens no more surprising than the running threshold mu, then adapt mu toward TAU
        surpr = -torch.log2(sp.clamp_min(1e-12)); keep = surpr <= MU[0]; keep[0] = True; sp = sp * keep
    else: keep = torch.cumsum(sp, 0) - sp < TOPP; sp = sp * keep
    j = int(torch.multinomial(sp / sp.sum(), 1)); s_obs = float(-torch.log2(p[si[j]].clamp_min(1e-12))); SURP.append(s_obs)
    if TAU > 0: MU[0] -= ETA * (s_obs - TAU)
    return int(si[j])


MAXCOPY, MAXLEAPS = int(os.environ.get("MAXCOPY", 16)), int(os.environ.get("MAXLEAPS", 3))   # copy guard (notes) and max consecutive leaps >= a 6th
_data = pickle.load(open(os.environ.get("SYMDATA", os.path.expanduser("~/music-visualizer/symbolic/sym_data.pkl")), "rb")); CONT = {}
for d_ in _data:
    iv_ = np.diff(d_[2][:, 2]).astype(int).tolist()
    for i_ in range(len(iv_) - MAXCOPY + 1): CONT.setdefault(tuple(iv_[i_:i_ + MAXCOPY - 2]), set()).add(iv_[i_ + MAXCOPY - 2])
CTRL = [ST.OFF_DEN + int(os.environ["DENS"]), ST.OFF_STP + int(os.environ["STEPC"])] if os.environ.get("DENS") and hasattr(ST, "OFF_DEN") else []   # learned style controls
toks = [ST.BOS, ST.OFF_ST + STYLE, ST.OFF_M + METER, ST.OFF_K + KEY] + CTRL + [ST.OFF_BAR]; bar, pos_end = 0, 0.0   # pos_end: earliest allowed onset in this bar (monophonic)
maxpos = int(round(BARLEN * 12))
while bar < BARS:
    lo = int(round(pos_end * 12))
    allowed = ([ST.OFF_BAR] if True else []) + [ST.OFF_POS + q for q in range(lo, min(maxpos, ST.NPOS))]
    t = pick(toks, allowed)
    toks.append(t)
    if t == ST.OFF_BAR: bar += 1; pos_end = max(0.0, pos_end - BARLEN); continue
    pos = (t - ST.OFF_POS) / 12; pb = PB.clone(); last = [x - ST.OFF_P for x in toks if ST.OFF_P <= x < ST.OFF_D][-4:]
    if len(last) == 4 and len(set(last)) == 1: pb[last[0]] = -1e9          # repetition guard: never more than 4 identical pitches in a row
    hist = [x - ST.OFF_P for x in toks if ST.OFF_P <= x < ST.OFF_D]
    if len(hist) >= MAXCOPY - 1:                                            # copy guard: don't extend an exact training interval sequence past MAXCOPY notes
        for nxt in CONT.get(tuple(np.diff(hist[-(MAXCOPY - 1):]).astype(int).tolist()), ()):
            if 0 <= hist[-1] + nxt < len(pb): pb[hist[-1] + nxt] = -1e9
    if len(hist) > MAXLEAPS and all(abs(hist[-i] - hist[-i - 1]) >= 8 for i in range(1, MAXLEAPS + 1)):   # leap guard: after MAXLEAPS big leaps in a row, the next step must be < a 6th
        for q in range(len(pb)):
            if abs(q - hist[-1]) >= 8: pb[q] = -1e9
    toks.append(pick(toks, list(range(ST.OFF_P, ST.OFF_D)), pb)); dt = pick(toks, list(range(ST.OFF_D, ST.OFF_BAR))); toks.append(dt)
    pos_end = pos + ST.DUR_Q[dt - ST.OFF_D]
notes = [n for n in ST.decode(toks, BARLEN) if n[0] < BARS * BARLEN]
ps = np.array([p for _, p, _ in notes])
if int(os.environ.get("TRANSPOSE_TO_CELLO", 1)):                           # whole-piece octave shift into the cello's comfortable range
    while np.median(ps) > 62: ps -= 12
    while np.median(ps) < 45: ps += 12
spb = 60.0 / BPM; pm = pretty_midi.PrettyMIDI(initial_tempo=BPM); ins = pretty_midi.Instrument(42, name="cello")
pm.time_signature_changes.append(pretty_midi.TimeSignature(num, den, 0))
for (on, _, d), p in zip(notes, ps):
    inbar = on % BARLEN; acc = 6 if inbar < 1e-6 else (3 if abs(inbar - round(inbar)) < 1e-6 else 0); phrase = 6 * np.sin(np.pi * ((on / BARLEN) % 4) / 4)
    ins.notes.append(pretty_midi.Note(int(np.clip(52 + acc + phrase, 30, 100)), int(p), 0.5 + on * spb, 0.5 + (on + d) * spb))
pm.instruments.append(ins); pm.write(_out)
inkey = np.mean([p % 12 in INKEY for p in ps]); iv = np.abs(np.diff(ps))


def grams(pp, n=8): iv_ = np.diff(pp).astype(int); return [tuple(iv_[i:i + n - 1]) for i in range(len(iv_) - n + 2)]


data = pickle.load(open(os.environ.get("SYMDATA", os.path.expanduser("~/music-visualizer/symbolic/sym_data.pkl")), "rb")); tg = set(g for d in data if d[1] not in TS.HELD for g in grams(d[2][:, 2]))
hit = np.array([g in tg for g in grams(ps)]); run = best = 0
for h in hit: run = run + 1 if h else 0; best = max(best, run)
print("wrote", _out, "| %d notes, %d bars of %d/%d at %d BPM (%.0f s) | %s | pitch %d-%d | in key %.0f%% | leaps>=7 %.0f%%, >=12 %.0f%%" % (
    len(notes), BARS, num, den, BPM, BARS * BARLEN * spb, ST.key_name(KEY), ps.min(), ps.max(), 100 * inkey, 100 * np.mean(iv >= 7), 100 * np.mean(iv >= 12)))
print("mean surprisal %.2f bits/token (sampling)" % np.mean(SURP), "| notes per bar %.1f | steps (<=2) %.0f%%" % (len(notes) / BARS, 100 * np.mean(iv <= 2)))
print("copying check: %.0f%% of 8-note interval patterns occur in the training lines | longest copied run %d notes" % (100 * hit.mean(), best + 7 if best else 0))
