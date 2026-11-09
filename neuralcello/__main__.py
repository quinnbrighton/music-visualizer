"""Command line: compose a piece and perform it, or perform / reconstruct existing files.
  python -m neuralcello compose OUT.wav [--style folk --key "D minor" --meter 3/4 --bars 24 --bpm 58 --tau 0.6 --seed 0 --vib 0.6]   (also writes OUT.mid)
  python -m neuralcello perform IN.mid OUT.wav [--vib 0.6 --seed 0]
  python -m neuralcello reconstruct IN.wav OUT.wav          (encode -> 24-ch latent -> decode)"""
import argparse, time
import numpy as np, soundfile as sf
from . import load

P = argparse.ArgumentParser(prog="python -m neuralcello"); P.add_argument("cmd", choices=["compose", "perform", "reconstruct"]); P.add_argument("paths", nargs="+")
for k, d in (("style", "folk"), ("key", "D minor"), ("meter", "3/4")): P.add_argument("--" + k, default=d)
for k, d, t in (("bars", 24, int), ("bpm", 58, float), ("tau", 0.6, float), ("seed", 0, int), ("vib", 0.6, float)): P.add_argument("--" + k, default=d, type=t)
P.add_argument("--device", default=None); a = P.parse_args(); m = load(a.device); t0 = time.time()
if a.cmd == "compose":
    out = a.paths[0]; mid = out.rsplit(".", 1)[0] + ".mid"
    pm = m.compose(style=a.style, key=a.key, meter=a.meter, bars=a.bars, bpm=a.bpm, tau=a.tau, seed=a.seed, out=mid); print("wrote", mid, "| %d notes" % len(pm.instruments[0].notes), flush=True)
    y = m.perform(pm, vib=a.vib, seed=a.seed, out=out)
elif a.cmd == "perform": y = m.perform(a.paths[0], vib=a.vib, seed=a.seed, out=a.paths[1]); out = a.paths[1]
else:
    x, sr = sf.read(a.paths[0], dtype="float32"); y = m.decode(m.encode(x, sr), seed=a.seed); out = a.paths[1]; sf.write(out, y, m.SR)
print("wrote %s | %.1f s of audio in %.1f s on %s" % (out, len(y) / m.SR, time.time() - t0, m.dev))
