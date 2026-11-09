"""Cache 4 MERT hidden-state layers (T,4,768) @ 75 Hz, fp16, for every recording (the model learns its own mix later)."""
import json, os, sys, numpy as np, torch
from mert_ddsp import MertDDSP
B = os.path.expanduser("~/music-visualizer/v2"); LAYERS = [1, 4, 8, 12]
m = MertDDSP().cuda().eval(); idx = json.load(open(f"{B}/mel512_index.json")); L, STEP = 240080, 240000
open(f"{B}/mertfeat/layers.txt", "w").write(json.dumps(LAYERS))
for k in sorted(idx):
    o = f"{B}/mertfeat/{k}.npy"
    if os.path.exists(o): continue
    a = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); out = []
    for s in range(0, len(a), STEP):
        x = torch.from_numpy(np.asarray(a[s:s + L], dtype=np.float32))[None].cuda()
        if x.shape[1] < 2000: continue
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hs = m.mert(x, output_hidden_states=True).hidden_states
        f = torch.stack([hs[i][0] for i in LAYERS], 1).float()                       # (T,4,768)
        out.append(f[:750].half().cpu().numpy() if s + STEP < len(a) else f.half().cpu().numpy())
    np.save(o, np.concatenate(out)); print(k[:8], np.concatenate(out).shape, flush=True)
print("mert cache done")
