"""Guess a note-level transcription ("baseline MIDI") from the 25 Hz control curves of every recording: estimate the recording's tuning,
snap pitch to semitones ignoring vibrato and brief slides, split notes on pitch changes and on bow re-attacks (loudness dip then rise).
Writes TOK_DIR/notes/{k}.npy (T,4) per frame: [in a note, MIDI pitch (0 in rests), onset flag, note velocity = mean loudness dB of the note]
and TOK_DIR/notes_midi/{k}.mid. usage: python guess_notes.py     env: TOK_DIR"""
import os, glob, numpy as np, pretty_midi
from scipy.ndimage import median_filter
TD = os.environ["TOK_DIR"]; FPS = 25; NOUT = os.environ.get("NOTES_OUT", "notes"); os.makedirs(f"{TD}/{NOUT}", exist_ok=True); os.makedirs(f"{TD}/{NOUT}_midi", exist_ok=True)
MINLEN, HOLD, DIP, RISE = 3, 3, 6.0, 4.0                                   # min note 120 ms; a new semitone must hold 3 frames; re-attack = >=6 dB dip then >=4 dB rise


def guess(c):
    f0, per, db = c[:, 0], c[:, 1], c[:, 2]; v = (per > 0.5) & (db > -60) & (f0 > 40)
    m = 69 + 12 * np.log2(np.clip(f0, 40, 2000) / 440); ang = 2 * np.pi * (m[v] % 1); tune = np.angle(np.exp(1j * ang).mean()) / (2 * np.pi)   # tuning offset in semitones (circular mean)
    q = np.round(median_filter(m - tune, 3)).astype(int); out = np.zeros((len(c), 4), np.float32); notes = []; t = 0
    while t < len(c):
        if not v[t]: t += 1; continue
        e = t
        while e < len(c) and v[e]: e += 1                                  # voiced run [t, e)
        s = t; cur = q[t]; i = t + 1
        while i < e:
            new_pitch = q[i] != cur and np.all(q[i: min(e, i + HOLD)] == q[i]) and i + HOLD <= e
            lo = db[max(s, i - 4): i].min() if i - s >= 2 else db[i]; reatt = i - s >= MINLEN and db[i - 1] <= db[max(s, i - 6): i].max() - DIP and db[min(e - 1, i + 2)] - db[i - 1] >= RISE
            if (new_pitch or reatt) and i - s >= MINLEN:
                notes.append((s, i)); s = i; cur = q[i]
            elif new_pitch: cur = q[i]
            i += 1
        notes.append((s, e)); t = e
    clean = []
    for s, e in notes:                                                     # pitch = most common semitone in the note; merge too-short notes into the previous one
        if e - s < MINLEN and clean and clean[-1][1] == s: clean[-1] = (clean[-1][0], e); continue
        clean.append((s, e))
    fixed = []                                                             # artifact cleanup: drop blips shorter than MINLEN frames; fix single-note octave errors
    for i, (s, e) in enumerate(clean):
        if e - s < MINLEN: continue
        fixed.append([s, e, int(np.bincount(q[s:e] - q[s:e].min()).argmax() + q[s:e].min())])
    for i in range(1, len(fixed) - 1):
        a_, b_, c_ = fixed[i - 1][2], fixed[i][2], fixed[i + 1][2]
        if abs(b_ - a_) >= 10 and abs(b_ - c_) >= 10 and abs(a_ - c_) <= 7 and fixed[i][0] - fixed[i - 1][1] <= 2 and fixed[i + 1][0] - fixed[i][1] <= 2:
            cand = b_ + 12 * round((((a_ + c_) / 2) - b_) / 12)                 # move the outlier by whole octaves toward its neighbours
            if abs(cand - (a_ + c_) / 2) <= 7: fixed[i][2] = cand
    res = []
    for s, e, p in fixed:
        vel = float(np.median(db[s:e])); out[s:e, 0] = 1; out[s:e, 1] = p; out[s, 2] = 1; out[s:e, 3] = vel; res.append((s, e, int(p), vel))
    return out, res, tune


def to_midi(res, path):
    pm = pretty_midi.PrettyMIDI(); inst = pretty_midi.Instrument(42, name="cello")
    for s, e, p, vel in res: inst.notes.append(pretty_midi.Note(int(np.clip(127 + 2.5 * vel, 20, 127)), p, s / FPS, e / FPS))
    pm.instruments.append(inst); pm.write(path)


if __name__ == "__main__":
    rates = []
    for f in sorted(glob.glob(f"{TD}/controls/*.npy")):
        k = os.path.basename(f)[:-4]; c = np.load(f); out, res, tune = guess(c); np.save(f"{TD}/{NOUT}/{k}.npy", out); to_midi(res, f"{TD}/{NOUT}_midi/{k}.mid")
        dur = out[:, 0].sum() / FPS; rates.append(len(res) / max(dur, 1)); print(k[:8], "notes", len(res), "| %.2f notes per voiced second" % rates[-1], "| tuning %+.0f cents" % (100 * tune), flush=True)
    print("median notes per voiced second %.2f" % np.median(rates))
