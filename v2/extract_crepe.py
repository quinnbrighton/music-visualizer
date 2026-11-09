"""CREPE (torchcrepe, full) pitch + periodicity at 100 Hz for every recording, Viterbi continuity decoding, 60 s chunks. GPU inference."""
import json, os, time, numpy as np, torch, torchaudio, torchcrepe
B = os.path.expanduser("~/music-visualizer/v2"); idx = json.load(open(f"{B}/mel512_index.json")); t0 = time.time(); tot = 0
ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); order = ck["test"] + ck["val"] + [k for k in sorted(idx) if k not in ck["test"] + ck["val"]]
for k in order:
    o = f"{B}/f0crepe/{k}.npy"
    if os.path.exists(o): continue
    a = np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r"); CH = 24000 * 60; P, Q = [], []
    for s in range(0, len(a), CH):
        x = torchaudio.functional.resample(torch.from_numpy(np.asarray(a[s:s + CH], dtype=np.float32))[None], 24000, 16000)
        if x.shape[1] < 1024: break
        p, q = torchcrepe.predict(x, 16000, hop_length=160, fmin=60, fmax=1100, model="full", decoder=torchcrepe.decode.viterbi, return_periodicity=True, batch_size=512, device="cuda")
        last = s + CH >= len(a); P.append(p[0].cpu().numpy() if last else p[0, :-1].cpu().numpy()); Q.append(q[0].cpu().numpy() if last else q[0, :-1].cpu().numpy())
    f0 = np.concatenate(P).astype(np.float32); per = np.concatenate(Q).astype(np.float16); n = 1 + len(a) // 240
    f0 = np.pad(f0, (0, max(0, n - len(f0))), mode="edge")[:n]; per = np.pad(per, (0, max(0, n - len(per))), mode="edge")[:n]
    np.save(o, f0); np.save(f"{B}/per/{k}.npy", per); tot += len(a) / 24000
    print(k[:8], f"{len(a)/24000/60:.1f} min  | cumulative {tot/3600:.2f} h audio in {(time.time()-t0)/60:.1f} min", flush=True)
print("crepe done")
