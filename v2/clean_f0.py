"""Clean yin f0: remove short spikes / octave blips, then median-filter in the log domain. Pure numpy."""
import glob, os, numpy as np
from scipy.signal import medfilt
def clean(f):
    lf = np.log2(np.clip(f, 40, 1200)); lf = lf.copy()
    for _ in range(2):                                              # blip removal: a run of <=6 frames that is >9 semitones from BOTH sides gets interpolated
        d = np.abs(lf - medfilt(lf, 15)) * 12; bad = d > 9
        idx = np.where(bad)[0]
        if len(idx):
            good = np.where(~bad)[0]; lf[bad] = np.interp(idx, good, lf[good])
    lf = medfilt(lf, 7)
    return (2 ** lf).astype(np.float32)
def jumps(f): j = np.abs(12 * np.log2(f[1:] / f[:-1])); return 100 * np.mean(j > 5), 100 * np.mean(j > 11)
if __name__ == "__main__":
    tot0, tot1, n = np.zeros(2), np.zeros(2), 0
    for p in sorted(glob.glob("f0/*.npy")):
        f = np.load(p); g = clean(f); np.save(p.replace("f0/", "f0clean/"), g); tot0 += jumps(f); tot1 += jumps(g); n += 1
    print(f"{n} files. mean % of frames with >5 / >11-semitone jumps: raw {np.round(tot0/n,2)}  cleaned {np.round(tot1/n,2)}")
