"""Is the train/held-out gap real? Evaluate a checkpoint with the IDENTICAL protocol (20 s window, mu not sampled, global from whole window) on TRAIN vs HELD-OUT recordings."""
import math, sys, numpy as np, torch
from voc_model import *
B = "/home/qbrighto/music-visualizer/v2"; name = sys.argv[1]; dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); train, val, test = ck["train"], ck["val"], ck["test"]
m = VocModel().to(dev).eval(); m.load_state_dict(torch.load(f"{B}/{name}.pt", map_location=dev)["model"])
rng = np.random.RandomState(0); trs = [k for k in train if np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r").shape[0] > 45 * SR]; trs = list(rng.choice(trs, 6, replace=False))
@torch.no_grad()
def run(k, s0):
    a = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); Fe = np.load(f"{B}/mertfeat/{k}.npy", mmap_mode="r"); n = 20 * SR; T = n // HOP
    x = torch.from_numpy(np.asarray(a[s0:s0 + n], dtype=np.float32))[None].to(dev); f = torch.from_numpy(np.asarray(Fe[s0 // 320: s0 // 320 + n // 320], dtype=np.float32))[None].to(dev)
    mu, lv = m.encode(f, T); zg = m.encode_global(f, T)[0]; y, _ = m.decode(mu, zg)
    # variants: (b) sampled z like training, (c) global code from a DIFFERENT 10 s of the same recording like training
    z = mu + torch.randn_like(mu) * (0.5 * lv).exp(); ys, _ = m.decode(z, zg)
    g2 = 960 * ((s0 + 30 * SR) // 960 % ((len(a) - 10 * SR) // 960)); fg = torch.from_numpy(np.asarray(Fe[g2 // 320: g2 // 320 + 750], dtype=np.float32))[None].to(dev); zg2 = m.encode_global(fg, 1000)[0]; yg, _ = m.decode(mu, zg2)
    return [mss_loss(v, x).item() for v in (y, ys, yg)]
rows = {}
for tag, ks in (("TRAIN recordings", trs), ("held-out val", val), ("Maisky test", test)):
    r = np.array([run(k, 960 * (min(len(np.load(f"{B}/audio24k/{k}.npy", mmap_mode='r')) // 3, 30 * SR) // 960)) for k in ks]); rows[tag] = r.mean(0)
    print(f"{tag:18s} mss: eval-protocol {r[:,0].mean():.3f} | sampled-z {r[:,1].mean():.3f} | global-from-other-10s {r[:,2].mean():.3f}   (n={len(ks)})")
