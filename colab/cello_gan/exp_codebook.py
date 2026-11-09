"""Does a smaller codebook give better quality per bit on this cello data? Fit residual k-means codebooks of size 256/512/1024 on real 25 Hz
code frames from training audio, measure the code error left on the validation songs vs real bits per second. env: as tokenize_rvq.py, OUTPNG"""
import os, sys, json
sys.argv = ["x", "cb", "1"]
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
model.eval(); rng = np.random.default_rng(0); CH = 20 * SR
@torch.no_grad()
def enc(k, s):
    x = torch.from_numpy(np.asarray(A[k][s:s + CH], np.float32))[None].to(dev); return to25(feats(x, None), CH // HOP)[0].t().float()
Xtr = torch.cat([enc(train[i], int(rng.integers(0, len(A[train[i]]) - CH)) // 960 * 960) for i in rng.choice(len(train), 400)])
Xva = torch.cat([enc(k, s) for k in val for s in range(0, len(A[k]) - CH, CH)])
print("code frames: fit %d, validation %d" % (len(Xtr), len(Xva)), flush=True)
var_va = (Xva - Xva.mean(0)).pow(2).sum(1).mean().item(); res = {}
for KC_, NQ_ in [tuple(map(int, p.split("x"))) for p in os.environ.get("SIZES", "256x12,512x10,1024x8").split(",")]:
    q = RVQ(NQ_, KC_, ZD).to(dev); q.kmeans_init(Xtr, iters=20); rt, rv = Xtr.clone(), Xva.clone(); rows = []
    for i in range(NQ_):
        rt = rt - q.cb[i][RVQ.nearest(rt, q.cb[i])]; rv = rv - q.cb[i][RVQ.nearest(rv, q.cb[i])]
        rows.append({"stages": i + 1, "bits_per_s": 25 * (i + 1) * int(np.log2(KC_)), "val_err": rv.pow(2).sum(1).mean().item() / var_va, "fit_err": rt.pow(2).sum(1).mean().item() / (Xtr - Xtr.mean(0)).pow(2).sum(1).mean().item()})
    res[KC_] = rows; print(KC_, " ".join("%d b/s: %.4f" % (r["bits_per_s"], r["val_err"]) for r in rows), flush=True)
json.dump(res, open(os.environ["OUTPNG"].replace(".png", ".json"), "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
BG, TX, TS, GR = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"; C = {32: "#e87ba4", 64: "#eda100", 128: "#6250d6", 256: "#eb6834", 512: "#1baf7a", 1024: "#2a78d6"}
plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": TX, "axes.labelcolor": TS, "xtick.color": TS, "ytick.color": TS, "axes.edgecolor": GR})
fig, axs = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG)
for KC_, rows in res.items():
    b = [r["bits_per_s"] for r in rows]; axs[0].plot(b, [r["val_err"] for r in rows], color=C[KC_], lw=2, marker="o", ms=4, label=f"{KC_} entries ({int(np.log2(KC_))} bits/stage)")
    axs[1].plot(b, [r["val_err"] / r["fit_err"] for r in rows], color=C[KC_], lw=2, marker="o", ms=4, label=f"{KC_}")
axs[0].set_yscale("log"); axs[0].set_title("Code error left on validation songs (lower is better)", loc="left", fontsize=12); axs[0].set_xlabel("real bits per second"); axs[0].set_ylabel("share of code variance not captured"); axs[0].legend(frameon=False)
axs[1].set_title("Overfitting: validation error / fitting error (1 = none)", loc="left", fontsize=12); axs[1].set_xlabel("real bits per second"); axs[1].legend(frameon=False, title="entries")
for a in axs: a.set_facecolor(BG); a.grid(color=GR, lw=0.6); [a.spines[k].set_visible(False) for k in ("top", "right")]
fig.tight_layout(); fig.savefig(os.environ["OUTPNG"], dpi=105, facecolor=BG)
