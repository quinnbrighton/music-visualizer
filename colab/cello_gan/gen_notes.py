"""Generate a melody with the note model and write a MIDI file (render it with midi_to_audio.py). The KEY token is forced to the requested key
every 64 notes, and a grammar mask keeps tokens in SHIFT, PITCH, DUR, VEL order. Also reports how much of the result is copied from the training
recordings: share of 8-note patterns (melodic intervals + onset gaps) that occur verbatim in training, and the longest copied run in notes.
usage: python gen_notes.py NOTELM.pt OUT.mid     env: TOK_DIR CELLO_DIR; KEY ("D minor") NOTES (300) TEMP (1.0) TOPP (0.95) SEED (0) PROMPT (.mid to continue)"""
import os, sys, numpy as np, torch, pretty_midi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); _ck, _out = sys.argv[1:3]
ck = torch.load(_ck, map_location="cuda"); a = ck["args"]
for k in ("CTX", "D", "LAYERS", "HEADS"): os.environ[k] = str(a[k])
os.environ["DROP"] = "0"; import note_tokens as NT, train_notelm as TN
net = TN.NoteLM().cuda(); net.load_state_dict(ck["model"]); net.eval(); print("note model step", ck["step"], "| val loss %.3f" % ck["val"], flush=True)
KEYS = {NT.key_name(i): i for i in range(24)}; KEY = KEYS[os.environ.get("KEY", "D minor")]; N = int(os.environ.get("NOTES", 300)); TEMP = float(os.environ.get("TEMP", 0.85)); TOPP = float(os.environ.get("TOPP", 0.95))
SCALE = float(os.environ.get("SCALE", 0.8))                                # logit penalty for pitch classes outside the requested key's scale (minor: natural + raised 7th); 0 = off
tonic = KEY % 12; steps_ = [0, 2, 4, 5, 7, 9, 11] if KEY < 12 else [0, 2, 3, 5, 7, 8, 10, 11]; INKEY = {(tonic + s_) % 12 for s_ in steps_}
PBIAS = torch.tensor([0.0 if (NT.PMIN + i) % 12 in INKEY else -SCALE for i in range(NT.NP)], device="cuda")
torch.manual_seed(int(os.environ.get("SEED", 0))); RANGES = [(NT.OFF_S, NT.OFF_P), (NT.OFF_P, NT.OFF_D), (NT.OFF_D, NT.OFF_V), (NT.OFF_V, NT.OFF_K)]

toks = [NT.BOS, NT.OFF_K + KEY]; nprompt = 0
if os.environ.get("PROMPT"):
    pm = pretty_midi.PrettyMIDI(os.environ["PROMPT"]); ns = sorted((n for i in pm.instruments for n in i.notes), key=lambda n: n.start)
    ev = np.array([(round(n.start * 25), n.pitch, max(1, round((n.end - n.start) * 25)), (n.velocity - 127) / 2.5) for n in ns], np.float32)
    toks = [NT.BOS] + NT.encode(ev); nprompt = len(ev)


@torch.no_grad()
def step(toks, field):                                                      # sample one token of the given field (0 shift, 1 pitch, 2 dur, 3 vel)
    x = torch.tensor([toks[-a["CTX"]:]], device="cuda"); lg = net(x)[0, -1].float() / TEMP; lo, hi = RANGES[field]; m = torch.full_like(lg, -1e9); m[lo:hi] = 0; lg = lg + m
    if field == 1: lg[NT.OFF_P:NT.OFF_D] += PBIAS
    p = torch.softmax(lg, -1); sp, si = torch.sort(p, descending=True); keep = torch.cumsum(sp, 0) - sp < TOPP; sp = sp * keep
    return int(si[torch.multinomial(sp / sp.sum(), 1)])


count = nprompt
while count < nprompt + N:
    if count % 64 == 0 and count > 0: toks.append(NT.OFF_K + KEY)          # keep the requested key in view
    for f in range(4): toks.append(step(toks, f))
    count += 1
notes = NT.decode(toks)[nprompt:]
pm = pretty_midi.PrettyMIDI(); ins = pretty_midi.Instrument(42, name="cello"); t0 = notes[0][0]
for on, p, d, v in notes:
    nxt = [o for o, *_ in notes if o > on]; dd = min(d, (nxt[0] - on) if nxt else d)    # monophonic: cut a note when the next one starts
    ins.notes.append(pretty_midi.Note(int(np.clip(127 + 2.5 * v, 20, 127)), int(p), (on - t0 + 25) / 25, (on - t0 + 25 + max(dd, 1)) / 25))
pm.instruments.append(ins); pm.write(_out)
ev = np.array(notes, np.float32); gen_key = NT.key_name(NT.key_of(ev[:, 1].astype(int), ev[:, 2]))


def grams(e, n=8):                                                          # 8-note melodic patterns as interval sequences (transposition- and tempo-invariant; performers' timing differs too much to match)
    iv = np.diff(e[:, 1]).astype(int); return [tuple(iv[i:i + n - 1]) for i in range(len(iv) - n + 2)]


EVD = dict(np.load(f"{os.environ['TOK_DIR']}/note_events.npz")); sp = torch.load(os.path.join(os.environ["CELLO_DIR"], "multirate_nested.pt"), map_location="cpu")
train_grams = set(g for k in sp["train"] if k in EVD for g in grams(EVD[k])); gg = grams(ev); hit = np.array([g in train_grams for g in gg])
run = best = 0
for h in hit: run = run + 1 if h else 0; best = max(best, run)
val_hit = np.mean([g in train_grams for k in sp["val"] if k in EVD for g in grams(EVD[k])])
print("wrote", _out, "| %d notes, %.1f s | requested %s, detected %s | pitch range %d-%d" % (len(notes), (notes[-1][0] - t0) / 25, NT.key_name(KEY), gen_key, ev[:, 1].min(), ev[:, 1].max()))
inkey = float(np.mean([int(p) % 12 in INKEY for p in ev[:, 1]])); print("in-key notes: %.0f%% (share of notes in the requested key's scale)" % (100 * inkey))
print("copying check: %.0f%% of 8-note patterns occur in training (held-out real recordings: %.0f%%) | longest copied run %d notes" % (100 * hit.mean(), 100 * val_hit, best + 7 if best else 0))
