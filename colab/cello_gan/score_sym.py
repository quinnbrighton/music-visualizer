"""Score generated MIDI pieces for musicality and rank them: coherence = mean per-note surprisal under the pretrained symbolic model
(lower = more like real melodies), plus penalties for statistics far from the real cello suites (big-leap share, octave leaps,
out-of-key notes, rhythmic monotony, too-wide range). usage: python score_sym.py PRETRAINED.pt piece1.mid ... (meter/key read from file names
written by batch_sym.sh: <key>_<num>-<den>_... )"""
import os, sys, numpy as np, torch, pretty_midi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); ck = torch.load(sys.argv[1], map_location="cuda"); a = ck["args"]
for k in ("CTX", "D", "LAYERS", "HEADS"): os.environ[k] = str(a[k])
os.environ["DROP"] = "0"; import sym_tokens as ST, train_symlm as TS
net = TS.SymLM().cuda(); net.load_state_dict(ck["model"]); net.eval(); KEYS = {ST.key_name(i).replace(" ", ""): i for i in range(24)}
rows = []
for f in sys.argv[2:]:
    base = os.path.basename(f)[:-4]; kname, meter = base.split("_")[:2]; key = KEYS[kname]; num, den = (int(x) for x in meter.split("-")); bl = num * 4.0 / den
    pm = pretty_midi.PrettyMIDI(f); bpm = pm.get_tempo_changes()[1][0]; spb = 60 / bpm; ns = pm.instruments[0].notes
    arr = []
    for n in ns:
        on = (n.start - 0.5) / spb; arr.append([on, (n.end - n.start) / spb, n.pitch, int(on // bl + 1e-6), on - (on // bl) * bl, bl])
    arr = np.array(arr, np.float32); t = [ST.BOS] + ST.encode(arr, key, ST.meter_index(num, den), 2); t = t[: a["CTX"]]
    with torch.no_grad():
        x = torch.tensor([t], device="cuda"); lp = torch.log_softmax(net(x[:, :-1])[0].float(), -1).gather(1, x[0, 1:, None])[:, 0].cpu().numpy()
    notes_in = max(1, sum(1 for y in t if ST.OFF_P <= y < ST.OFF_D)); surpr = -lp.sum() / notes_in / np.log(2)      # bits per note
    p = arr[:, 2].astype(int); iv = np.abs(np.diff(p)); tonic = key % 12; sc = [0, 2, 4, 5, 7, 9, 11] if key < 12 else [0, 2, 3, 5, 7, 8, 10, 11]
    inkey = np.mean([(q - tonic) % 12 in sc for q in p]); leaps = np.mean(iv >= 7); octs = np.mean(iv >= 12)
    durs = np.round(arr[:, 1] * 12).astype(int); runs = max(len(list(g)) for _, g in __import__("itertools").groupby(durs)); variety = len(set(durs))
    pen = 3 * max(0, abs(leaps - 0.16) - 0.06) * 10 + 3 * max(0, octs - 0.05) * 10 + 2 * max(0, 0.9 - inkey) * 10 + (2 if runs > 48 else 0) + (2 if variety < 3 else 0) + max(0, np.ptp(p) - 30) * 0.2
    rows.append((surpr + pen, surpr, pen, base, leaps, octs, inkey, variety, runs))
rows.sort()
print("rank | score (lower better) = surprisal bits/note + penalty | piece | leaps 5th+ / oct+ | in key | rhythm values | longest same-rhythm run")
for i, r in enumerate(rows): print("%2d | %5.2f = %5.2f + %4.2f | %-36s | %3.0f%% / %2.0f%% | %3.0f%% | %d | %d" % (i + 1, r[0], r[1], r[2], r[3], 100 * r[4], 100 * r[5], 100 * r[6], r[7], r[8]))
