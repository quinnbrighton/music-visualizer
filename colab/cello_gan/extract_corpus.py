"""Extract monophonic melodic lines WITH METER from the music21 corpus already installed locally (public-domain folk collections + Bach chorale
voices), for pretraining the note model. Each line -> (N,6) float array [onset (quarters), duration (quarters), MIDI pitch, bar index,
position in bar (quarters), bar length (quarters)] plus its key (0-11 major, 12-23 minor) and source.
usage: python extract_corpus.py OUT.pkl     env: PROCS (6)"""
import os, sys, glob, pickle, numpy as np
from multiprocessing import Pool
import music21 as m21
CORP = os.path.join(os.path.dirname(m21.__file__), "corpus")
SOURCES = {"essen": "essenFolksong/*.abc", "nottingham": "nottingham-dataset/**/*.abc", "oneills": "oneills1850/*.abc", "ryans": "ryansMammoth/*.abc",
           "airds": "airdsAirs/*.abc", "miscfolk": "miscFolk/**/*.abc", "chorales": "bach/bwv*.mxl"}
NAMES = ["C", "C#", "D", "E-", "E", "F", "F#", "G", "A-", "A", "B-", "B"]


def key_index(k):
    try: return (k.tonic.pitchClass + (12 if k.mode == "minor" else 0))
    except Exception: return -1


def lines_of(score):
    out = []
    try: kk = key_index(score.analyze("key"))
    except Exception: kk = -1
    for part in (score.parts if hasattr(score, "parts") and len(score.parts) else [score]):
        rows = []
        for n in part.recurse().notesAndRests:
            if n.isRest or n.duration.quarterLength == 0: continue
            p = n.pitches[-1].midi if n.isChord else n.pitch.midi              # chords (rare in these collections): keep the top note
            ms = n.getContextByClass("Measure"); ts = n.getContextByClass("TimeSignature")
            bar = ms.number if ms is not None and ms.number is not None else 0; bl = ts.barDuration.quarterLength if ts is not None else 4.0
            off = float(n.getOffsetInHierarchy(part)); pos = float(n.offset) if ms is not None else off % bl
            if n.tie is not None and n.tie.type in ("stop", "continue") and rows and rows[-1][2] == p: rows[-1][1] += float(n.duration.quarterLength); continue
            rows.append([off, float(n.duration.quarterLength), p, bar, pos, bl])
        if len(rows) >= 16: out.append((np.array(rows, np.float32), kk))
    return out


def work(args):
    src, f = args; res = []
    try:
        s = m21.converter.parse(f)
        scores = list(s.scores) if isinstance(s, m21.stream.Opus) else [s]
        for sc in scores:
            for arr, kk in lines_of(sc): res.append((src, os.path.basename(f), arr, kk))
    except Exception as e: print("skip", f, type(e).__name__, flush=True)
    return res


if __name__ == "__main__":
    jobs = [(src, f) for src, pat in SOURCES.items() for f in sorted(glob.glob(os.path.join(CORP, pat), recursive=True))]
    print(len(jobs), "files", {s: sum(1 for j in jobs if j[0] == s) for s in SOURCES}, flush=True)
    with Pool(int(os.environ.get("PROCS", 6))) as pool:
        allres = []
        for i, r in enumerate(pool.imap_unordered(work, jobs, chunksize=1)):
            allres += r
            if i % 50 == 0: print(i, "files done |", len(allres), "lines", flush=True)
    pickle.dump(allres, open(sys.argv[1], "wb"))
    from collections import Counter
    c = Counter(r[0] for r in allres); nn = Counter(); [nn.update({r[0]: len(r[2])}) for r in allres]
    print("lines per source", dict(c), "| notes per source", dict(nn), "| total notes", sum(nn.values()))
