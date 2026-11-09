"""Export the trained checkpoints to small inference-only files in models/ (fp16 state_dicts + norm/args metadata; no optimizer, discriminator,
RVQ or non-EMA weights). Also writes the composer's copy-guard index (interval patterns of the training lines, so compose() needs no dataset).
usage: python -m neuralcello.export_weights [--fp32] [--src COLAB_RESULTS] [--sym SYM_DATA.pkl] [--out DIR]
Run once by the author; outsiders just use the files in models/."""
import os, argparse, pickle
import numpy as np, torch

H = os.path.expanduser("~/music-visualizer")
P = argparse.ArgumentParser(); P.add_argument("--src", default=f"{H}/colab_results"); P.add_argument("--sym", default=f"{H}/symbolic/sym_data.pkl")
P.add_argument("--out", default=f"{H}/models"); P.add_argument("--fp32", action="store_true", help="keep float32 (reference copy for verification)")
P.add_argument("--maxcopy", type=int, default=16)


def half(sd, fp32):                                                        # floats -> fp16 (ints, e.g. BatchNorm counters, unchanged); clone so tied tensors don't share storage
    return {k: (v.detach().clone() if fp32 or not v.is_floating_point() else v.detach().half().clone()) for k, v in sd.items()}


def main(a):
    os.makedirs(a.out, exist_ok=True); L = lambda p: torch.load(f"{a.src}/{p}", map_location="cpu", weights_only=False); sfx = "_fp32" if a.fp32 else ""
    c = L("reg3/codec_snd_s8000.pt"); torch.save({"model": half(c["model"], a.fp32), "step": c["step"], "src": "reg3/codec_snd_s8000.pt"}, f"{a.out}/codec{sfx}.pt")
    for name, p in (("latent", "reg3/lat4_s20000.pt"), ("curve", "curve3/curve3_last.pt")):
        c = L(p); torch.save({"ema": half(c["ema"], a.fp32), "norm": c["norm"], "args": c["args"], "step": c["step"], "src": p}, f"{a.out}/{name}{sfx}.pt")
    c = L("symlm/symlm_default.pt"); sd = {k: v for k, v in c["model"].items() if k != "head.weight"}                  # head is tied to emb
    torch.save({"model": half(sd, a.fp32), "args": c["args"], "step": c["step"], "src": "symlm/symlm_default.pt"}, f"{a.out}/composer{sfx}.pt")
    if not a.fp32:                                                          # copy guard: (MAXCOPY-2 intervals -> next interval) seen in any training line, as gen_sym.py builds CONT
        rows = set()
        for d_ in pickle.load(open(a.sym, "rb")):
            iv = np.diff(d_[2][:, 2]).astype(int).tolist(); n = a.maxcopy - 2
            for i in range(len(iv) - a.maxcopy + 1): rows.add(tuple(iv[i:i + n + 1]))
        R = np.array(sorted(rows), np.int8); np.savez_compressed(f"{a.out}/copyguard.npz", rows=R, maxcopy=a.maxcopy)
    for f in sorted(os.listdir(a.out)): print("%-22s %6.1f MB" % (f, os.path.getsize(f"{a.out}/{f}") / 2 ** 20))


if __name__ == "__main__": main(P.parse_args())
