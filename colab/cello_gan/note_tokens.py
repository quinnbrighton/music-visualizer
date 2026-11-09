"""Note-event tokens for the MIDI generator, from the guessed notes (guess_notes.py, TOK_DIR/notes/{k}.npy at 25 Hz).
Each note = 4 tokens: SHIFT (frames since the previous onset, bucketed), PITCH (MIDI 28-88), DUR (frames, bucketed), VEL (loudness dB, 8 buckets).
A KEY token (24 keys, Krumhansl-Schmuckler on the next ~30 s of notes) is inserted at the start of every training window and every KEYEVERY notes.
usage: python note_tokens.py   (writes TOK_DIR/note_events.npz: per recording an (N,4) array of [onset frame, pitch, duration frames, velocity dB])"""
import os, glob, numpy as np

PMIN, PMAX = 28, 88
SHIFT_B = np.array(list(range(0, 25)) + [26, 28, 30, 33, 36, 40, 45, 50, 56, 63, 70, 80, 90, 100, 115, 130, 150, 175, 200, 250])   # frames (40 ms each)
DUR_B = np.array(list(range(1, 25)) + [26, 28, 30, 33, 36, 40, 45, 50, 56, 63, 70, 80, 90, 100, 125, 150, 200])
VEL_B = np.array([-46, -40, -35, -31, -27, -23, -19])                    # dB edges -> 8 buckets
NS, NP, ND, NV = len(SHIFT_B), PMAX - PMIN + 1, len(DUR_B), len(VEL_B) + 1
OFF_S, OFF_P = 0, NS; OFF_D = OFF_P + NP; OFF_V = OFF_D + ND; OFF_K = OFF_V + NV; PAD = OFF_K + 24; BOS = PAD + 1; VOCAB = BOS + 1
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
KS_MAJ = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]); KS_MIN = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def key_of(pitches, durs):                                                  # 0-11 major, 12-23 minor (tonic pitch class)
    h = np.bincount(np.asarray(pitches) % 12, weights=np.asarray(durs, float), minlength=12)
    if h.sum() == 0: return 0
    sc = [np.corrcoef(np.roll(KS_MAJ, t), h)[0, 1] for t in range(12)] + [np.corrcoef(np.roll(KS_MIN, t), h)[0, 1] for t in range(12)]
    return int(np.nanargmax(sc))


def key_name(k): return NAMES[k % 12] + (" major" if k < 12 else " minor")
def transpose_key(k, s): return (k % 12 + s) % 12 + (12 if k >= 12 else 0)
def bucket(v, edges): return int(np.clip(np.searchsorted(edges, v, side="right") - 1, 0, len(edges) - 1))


def encode(ev, key_every=64, stretch=1.0, shift=0):                        # ev (N,4) -> token list (KEY tokens every key_every notes, computed on the following ~30 s)
    toks = []; prev = ev[0, 0]
    for i, (on, p, d, v) in enumerate(ev):
        if i % key_every == 0:
            j = i;
            while j < len(ev) and ev[j, 0] - on < 750: j += 1
            toks.append(OFF_K + transpose_key(key_of(ev[i:max(j, i + 1), 1].astype(int), ev[i:max(j, i + 1), 2]), shift))
        toks += [OFF_S + bucket((on - prev) * stretch, SHIFT_B), OFF_P + int(np.clip(p + shift, PMIN, PMAX)) - PMIN, OFF_D + bucket(d * stretch, DUR_B), OFF_V + int(np.searchsorted(VEL_B, v))]
        prev = on
    return toks


def decode(toks):                                                           # tokens -> list of (onset frame, pitch, duration frames, velocity dB); ignores KEY/PAD/BOS
    out, t, cur = [], 0, {}
    vel_mid = np.concatenate([[VEL_B[0] - 3], (VEL_B[1:] + VEL_B[:-1]) / 2, [VEL_B[-1] + 3]])
    for x in toks:
        if OFF_S <= x < OFF_P: t += SHIFT_B[x - OFF_S]; cur = {"on": t}
        elif OFF_P <= x < OFF_D and "on" in cur: cur["p"] = x - OFF_P + PMIN
        elif OFF_D <= x < OFF_V and "p" in cur: cur["d"] = DUR_B[x - OFF_D]
        elif OFF_V <= x < OFF_K and "d" in cur: out.append((cur["on"], cur["p"], cur["d"], float(vel_mid[x - OFF_V]))); cur = {}
    return out


if __name__ == "__main__":
    TD = os.environ["TOK_DIR"]; EV = {}
    for f in sorted(glob.glob(f"{TD}/notes/*.npy")):
        n = np.load(f); on = np.where(n[:, 2] > 0)[0]; ends = np.append(on[1:], len(n)); ev = []
        for s, e in zip(on, ends):
            d = int(n[s:e, 0].sum()); ev.append((s, int(n[s, 1]), max(d, 1), float(n[s, 3])))
        EV[os.path.basename(f)[:-4]] = np.array(ev, np.float32)
    np.savez(f"{TD}/note_events.npz", **EV); tot = sum(len(v) for v in EV.values())
    allp = np.concatenate([v[:, 1] for v in EV.values()]); print(len(EV), "recordings |", tot, "notes | pitch range", int(allp.min()), "-", int(allp.max()), "| vocab", VOCAB)
    ks = [key_name(key_of(v[:, 1].astype(int), v[:, 2])) for v in list(EV.values())[:12]]; print("whole-recording keys (first 12):", ks)
    t = encode(EV[list(EV)[0]][:20]); print("round trip ok:", [(int(a), int(b), int(c)) for a, b, c, _ in decode(t)[:5]], "vs", EV[list(EV)[0]][:5, :3].astype(int).tolist())
