"""Meter-aware note tokens for the symbolic MIDI generator. Header: STYLE (folk / chorale / cello), METER, KEY. Then per note: [BAR if the note
starts a new bar], POS (onset within the bar, 1/12-quarter grid), PITCH, DUR (quarter-note buckets). Rests are implicit (POS / BAR jumps).
Also builds the dataset: python sym_tokens.py CORPUS.pkl BACH_MIDI_DIR OUT.pkl"""
import os, sys, glob, pickle, numpy as np

PMIN, PMAX = 24, 96
METERS = [(2, 4), (3, 4), (4, 4), (2, 2), (3, 2), (3, 8), (6, 8), (9, 8), (12, 8), (6, 4), (5, 4), (0, 0)]   # last = other
STYLES = ["folk", "chorale", "cello"]
DUR_Q = np.array([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 192]) / 12.0   # durations in quarters
NPOS = 96                                                                   # positions 0..95 on a 1/12-quarter grid (bars up to 8 quarters)
OFF_POS = 0; OFF_P = NPOS; OFF_D = OFF_P + (PMAX - PMIN + 1); OFF_BAR = OFF_D + len(DUR_Q); OFF_K = OFF_BAR + 1; OFF_M = OFF_K + 24; OFF_ST = OFF_M + len(METERS)
PAD = OFF_ST + len(STYLES); BOS = PAD + 1
OFF_DEN = BOS + 1; OFF_STP = OFF_DEN + 4; VOCAB = OFF_STP + 4                  # learned style controls (appended so older token ids are unchanged)
DEN_EDGES, STP_EDGES = [2.5, 4.5, 8.5], [0.5, 0.7, 0.85]                     # notes per bar; share of steps (<= 2 semitones)


def attrs(line):                                                            # (density bucket, stepwise bucket) of a stretch of notes
    nb = max(1, len(np.unique(line[:, 3]))); dens = len(line) / nb; iv = np.abs(np.diff(line[:, 2])); stp = float(np.mean(iv <= 2)) if len(iv) else 1.0
    return int(np.searchsorted(DEN_EDGES, dens)), int(np.searchsorted(STP_EDGES, stp))
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def key_name(k): return NAMES[k % 12] + (" major" if k < 12 else " minor")
def transpose_key(k, s): return (k % 12 + s) % 12 + (12 if k >= 12 else 0)
def meter_index(num, den): return METERS.index((num, den)) if (num, den) in METERS else len(METERS) - 1


def encode(line, key, meter, style, shift=0, ctrl=None):                   # line (N,6) [onset q, dur q, pitch, bar, pos q, barlen q] -> tokens; ctrl = (density, stepwise) buckets
    t = [OFF_ST + style, OFF_M + meter, OFF_K + transpose_key(key, shift)] + ([OFF_DEN + ctrl[0], OFF_STP + ctrl[1]] if ctrl is not None else []); prevbar = None
    for on, d, p, bar, pos, bl in line:
        if prevbar is None or bar != prevbar:
            t += [OFF_BAR] * (1 if prevbar is None else int(min(max(bar - prevbar, 1), 4)))
        prevbar = bar
        t += [OFF_POS + int(np.clip(round(pos * 12), 0, NPOS - 1)), OFF_P + int(np.clip(p + shift, PMIN, PMAX)) - PMIN, OFF_D + int(np.abs(DUR_Q - d).argmin())]
    return t


def decode(toks, barlen=4.0):                                               # -> list of (onset q, pitch, dur q); bars advance by barlen quarters
    out, bar, cur = [], -1, {}
    for x in toks:
        if x == OFF_BAR: bar += 1
        elif OFF_POS <= x < OFF_P: cur = {"on": max(bar, 0) * barlen + (x - OFF_POS) / 12}
        elif OFF_P <= x < OFF_D and "on" in cur: cur["p"] = x - OFF_P + PMIN
        elif OFF_D <= x < OFF_BAR and "p" in cur: out.append((cur["on"], cur["p"], float(DUR_Q[x - OFF_D]))); cur = {}
    return out


def bach_lines(d):                                                          # Mutopia MIDI -> lines with meter (skips files without a time signature)
    import pretty_midi, warnings; warnings.filterwarnings("ignore"); out = []
    for f in sorted(glob.glob(f"{d}/**/*.mid", recursive=True)):
        if "viola" in f: continue
        pm = pretty_midi.PrettyMIDI(f)
        if not pm.time_signature_changes: print("no meter, skipped:", os.path.basename(f)); continue
        res = pm.resolution; tsc = [(pm.time_to_tick(ts.time) / res, ts.numerator * 4.0 / ts.denominator, (ts.numerator, ts.denominator)) for ts in pm.time_signature_changes]
        ns = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: (n.start, -n.pitch)); rows = []; last_on = None
        for n in ns:
            on = pm.time_to_tick(n.start) / res
            if last_on is not None and abs(on - last_on) < 1e-6: continue          # double stops: keep the top note
            last_on = on; seg = [s for s in tsc if s[0] <= on + 1e-6][-1]; bars_before = 0.0
            for (a, bl_, _), nxt in zip(tsc, tsc[1:] + [(1e9, 0, 0)]):            # bar index across meter changes
                if a > on: break
                span = min(on, nxt[0]) - a; bars_before += span / bl_ if nxt[0] <= on else 0
            bar = int(bars_before + (on - seg[0]) // seg[1]); pos = (on - seg[0]) % seg[1]
            rows.append([on, pm.time_to_tick(n.end) / res - on, n.pitch, bar, pos, seg[1]])
        m = tsc[0][2]; arr = np.array(rows, np.float32)
        from note_tokens import key_of
        out.append(("cello", os.path.basename(f), arr, key_of(arr[:, 2].astype(int), arr[:, 1]), meter_index(*m)))
    return out


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    corpus = pickle.load(open(sys.argv[1], "rb")); data = []
    import music21 as m21
    for src, fname, arr, kk in corpus:                                      # meter of each corpus line from its bar length (good enough for the common meters)
        bl = float(np.median(arr[:, 5])); m = {2.0: (2, 4), 3.0: (3, 4), 4.0: (4, 4), 1.5: (3, 8), 6.0: (6, 4), 4.5: (9, 8), 5.0: (5, 4)}.get(bl, (0, 0))
        if bl == 3.0 and np.median(np.diff(np.unique(arr[:, 4]))) % 0.5 == 0 and np.mean((arr[:, 4] * 2) % 3 == 0) > 0.8: m = (6, 8)
        if kk < 0: from note_tokens import key_of; kk = key_of(arr[:, 2].astype(int), arr[:, 1])
        data.append(("chorale" if src == "chorales" else "folk", fname, arr, kk, meter_index(*m)))
    data += bach_lines(sys.argv[2]); pickle.dump(data, open(sys.argv[3], "wb"))
    from collections import Counter
    print("lines", len(data), "| by style", Counter(d[0] for d in data), "| notes by style", {s: int(sum(len(d[2]) for d in data if d[0] == s)) for s in STYLES})
    print("meters", Counter(METERS[d[4]] for d in data).most_common(8)); print("vocab", VOCAB)
    t = encode(data[-1][2][:12], data[-1][3], data[-1][4], 2); print("round trip:", decode(t)[:5], "vs", data[-1][2][:5, [0, 2, 1]].tolist())
