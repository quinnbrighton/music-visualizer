"""Where the surprise is: train the same causal predictor (1.8 s context) on the old codec's latents and on the regularized
codec's latents (training recordings only), then plot its per-frame error over the held-out passage with note onsets marked.
Writes docs/assets/surprise.png.   usage: python make_surprise_fig.py   (GPU, a few minutes)"""
import os, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, pretty_midi, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
H = os.path.expanduser("~/music-visualizer"); R = f"{H}/colab_results"; OUTD = f"{H}/docs/assets"; dev = "cuda"; ZD = 24; torch.manual_seed(0)
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
sp = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); k = sp["test"][0]
SETS = {"old": f"{H}/tokens64/latents", "new": f"{R}/reg3/tok/latents"}


class Pred(nn.Module):                                                      # same causal predictor as train_reg.py: frame t from frames < t, 46-frame receptive field
    def __init__(self, d=256):
        super().__init__(); self.inp = nn.Conv1d(ZD, d, 1); self.cs = nn.ModuleList([nn.Conv1d(d, d, 4, dilation=r) for r in (1, 2, 4, 8)]); self.out = nn.Conv1d(d, ZD, 1)
    def forward(self, z):
        h = self.inp(F.pad(z, (1, 0))[:, :, :-1])
        for c_ in self.cs: h = h + F.gelu(c_(F.pad(h, (3 * c_.dilation[0], 0))))
        return self.out(h)


def per_frame_error(name, steps=3000):
    tr = [np.load(f"{SETS[name]}/{t}.npy").astype(np.float32) for t in sp["train"] if os.path.exists(f"{SETS[name]}/{t}.npy")]
    allz = np.concatenate(tr); mu, sd = allz.mean(0), float(allz.std())       # one global scale (sound-space units), as in the frozen-predictor test
    tr = [torch.from_numpy((z - mu) / sd).to(dev) for z in tr]; net = Pred().to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3)
    for s in range(steps):
        b = []
        for _ in range(32):
            z = tr[np.random.randint(len(tr))]; i = np.random.randint(len(z) - 160); b.append(z[i: i + 160].t())
        zb = torch.stack(b); loss = (net(zb) - zb)[:, :, 46:].pow(2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); f0 = 5572800 // 960; zt = torch.from_numpy((np.load(f"{SETS[name]}/{k}.npy").astype(np.float32)[f0 - 46: f0 + 300] - mu) / sd).to(dev).t()[None]
    with torch.no_grad(): e = (net(zt) - zt).pow(2).mean(1)[0, 46:].cpu().numpy(); var = zt[0, :, 46:].var(1).mean().item()
    print(name, "unexplained share on the held-out window %.3f" % (e.mean() / var)); return e / var


eo, en = per_frame_error("old"), per_frame_error("new"); t = np.arange(300) / 25
on = np.array([n.start for n in pretty_midi.PrettyMIDI(f"{R}/curve3/maisky_30s_guessed.mid").instruments[0].notes if n.start < 12])
near = np.zeros(300, bool)
for o in on: near[max(0, int(o * 25) - 2): int(o * 25) + 3] = True        # within 80 ms of a note onset
share = lambda e: e[near].sum() / e.sum()
print("frames near onsets %.0f%% | error share there: old %.0f%%, new %.0f%%" % (100 * near.mean(), 100 * share(eo), 100 * share(en)))
fig, (a1, a2) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.4], "hspace": 0.25})
c = np.load(f"{H}/tokens64/controls/{k}.npy")[5572800 // 960: 5572800 // 960 + 300]; m = np.where(c[:, 1] > 0.5, 69 + 12 * np.log2(np.clip(c[:, 0], 40, 2000) / 440), np.nan)
a1.plot(t, m, color=INK2, lw=1.4); a1.set_ylabel("pitch (MIDI)"); a1.set_title("The held-out passage, with each note onset marked", loc="left")
for a in (a1, a2):
    for o in on: a.axvline(o, color=GRID, lw=1, zorder=0)
a2.plot(t, eo, color=ORANGE, lw=1.2, label="old codec latent: %.0f%% unexplained on this passage" % (100 * eo.mean()))
a2.plot(t, en, color=BLUE, lw=1.4, label="regularized latent: %.0f%%" % (100 * en.mean()))
a2.set_ylabel("surprise (unexplained share)"); a2.set_xlabel("seconds"); a2.set_xlim(0, 12); a2.legend(frameon=False, loc="upper right", fontsize=9)
a2.set_title("Surprise: how badly the next frame is predicted from the previous 1.8 s", loc="left")
fig.text(0.01, 0.0, "Same predictor, training and scale for both latents (training recordings only). Onsets from the automatically guessed notes.", color=INK2, fontsize=8.5)
fig.savefig(f"{OUTD}/surprise.png", dpi=110, bbox_inches="tight"); print("wrote surprise.png")
