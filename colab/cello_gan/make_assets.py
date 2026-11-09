"""Build the figures and short audio clips used by README.md and tour.ipynb (docs/assets/). Reproducible from the trained models.
usage: python make_assets.py     env: as gen_lat.py (codec skeleton)"""
import os, sys, shutil, subprocess
sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import soundfile as sf, pretty_midi, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
R = os.path.expanduser("~/music-visualizer/colab_results"); OUTD = os.path.expanduser("~/music-visualizer/docs/assets"); AUD = f"{OUTD}/audio"; os.makedirs(AUD, exist_ok=True)
TD = os.environ["TOK_DIR"]; DS_ = HOP * DSR; k = test[0]; S0 = 5572800; SEC = 12; f0 = S0 // DS_; n = SEC * 25
# palette (dataviz reference instance, light mode): series blue / orange, text and surface roles
BG, INK, INK2, GRID, BLUE, ORANGE = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0", "#2a78d6", "#eb6834"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "text.color": INK, "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.spines.top": False, "axes.spines.right": False})
c = torch.load(f"{R}/reg3/codec_snd_s8000.pt", map_location=dev, weights_only=False); model.load_state_dict(c["model"]); model.eval()


@torch.no_grad()
def dec_ola(z, W=100, Hh=75):                                               # 4 s windows with crossfades (the decoder's training length)
    nn_ = z.shape[0]; S = HOP * DSR; out = np.zeros(nn_ * S); ws = np.zeros(nn_ * S); st = list(range(0, max(1, nn_ - W) + 1, Hh))
    if st[-1] + W < nn_: st.append(nn_ - W)
    for s in st:
        e = min(nn_, s + W); T = (e - s) * DSR; hh = dec_feats(z[s:e].t()[None], T); y = model.synth(hh, model.pitch_f0(hh), T)[0].cpu().numpy()
        w = np.ones(len(y)); r = (W - Hh) * S
        if s > 0: w[:r] = np.linspace(0, 1, r)
        if e < nn_: w[-r:] = np.linspace(1, 0, r)
        out[s * S: s * S + len(y)] += y * w; ws[s * S: s * S + len(y)] += w
    return out / np.maximum(ws, 1e-6)


def norm(y, ref=None): y = np.asarray(y, np.float64); r = np.sqrt(np.mean((ref if ref is not None else y) ** 2)); return (y / (np.sqrt(np.mean(y ** 2)) + 1e-9) * r).astype(np.float32)


def mp4(wav, out, title=None):                                              # small playable video (waveform) for GitHub's file viewer
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-filter_complex", "[0:a]showwaves=s=640x160:mode=cline:colors=0x2a78d6,format=yuv420p[v]",
                    "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "30", "-c:a", "aac", "-b:a", "128k", out], check=True)


def spec_db(y):
    import scipy.signal as ss
    f, t, Z = ss.stft(y, SR, nperseg=1024, noverlap=768); Sdb = 20 * np.log10(np.abs(Z) + 1e-6); return f, t, Sdb


# 1. original vs codec reconstruction (12 s of the held-out Maisky recording)
x = np.asarray(A[k][S0: S0 + n * DS_], np.float32); x = x / (np.abs(x).max() + 1e-6) * 0.7
z = torch.from_numpy(np.load(f"{R}/reg3/tok/latents/{k}.npy").astype(np.float32))[f0: f0 + n].to(dev); y = norm(dec_ola(z), x)
sf.write(f"{AUD}/1_original_maisky.wav", x, SR); sf.write(f"{AUD}/2_codec_reconstruction.wav", y, SR)
fig, axs = plt.subplots(2, 2, figsize=(12, 5.2), gridspec_kw={"height_ratios": [1, 2]}, sharex=True)
for j, (sig, name, col) in enumerate([(x, "Original recording (held out, never trained on)", INK2), (y, "Rebuilt from the 24-number-per-frame code", BLUE)]):
    tt = np.arange(len(sig)) / SR; axs[0, j].plot(tt, sig, color=col, lw=0.4); axs[0, j].set_title(name, loc="left"); axs[0, j].set_yticks([]); axs[0, j].set_ylim(-0.8, 0.8)
    f_, t_, Sd = spec_db(sig); m = f_ <= 6000; axs[1, j].pcolormesh(t_, f_[m] / 1000, Sd[m], shading="auto", cmap="magma", vmin=Sd[m].max() - 80, vmax=Sd[m].max())
    axs[1, j].set_ylabel("frequency (kHz)" if j == 0 else ""); axs[1, j].set_xlabel("seconds")
fig.text(0.01, 0.97, "Top: waveform (air pressure over time).  Bottom: spectrogram (which frequencies are sounding; brighter = louder).", color=INK2, fontsize=9)
fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUTD}/codec_compare.png", dpi=110); plt.close(fig)

# 2. the embedding: latent path in 2D, coloured by pitch, + all 24 channels over time
zz = np.load(f"{R}/reg3/tok/latents/{k}.npy").astype(np.float32)[f0: f0 + 40 * 25]; craw = np.load(f"{TD}/controls/{k}.npy")[f0: f0 + 40 * 25]
v = craw[:, 1] > 0.5; midi = np.where(v, 69 + 12 * np.log2(np.clip(craw[:, 0], 40, 1200) / 440), np.nan)
zc = zz - zz.mean(0); U, Sv, Vt = np.linalg.svd(zc, full_matrices=False); xy = zc @ Vt[:2].T
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
axs[0].plot(xy[:, 0], xy[:, 1], color=GRID, lw=0.6, zorder=1); sc = axs[0].scatter(xy[v, 0], xy[v, 1], c=midi[v], cmap="Blues", s=6, zorder=2, vmin=np.nanpercentile(midi, 5) - 6)
axs[0].set_title("The music as a path through the code (40 s)", loc="left"); axs[0].set_xlabel("direction 1"); axs[0].set_ylabel("direction 2"); axs[0].set_xticks([]); axs[0].set_yticks([])
cb = fig.colorbar(sc, ax=axs[0], fraction=0.046, pad=0.02); cb.set_label("pitch (darker = higher)", color=INK2); cb.set_ticks([])
zs = (zz - zz.mean(0)) / zz.std(0); axs[1].imshow(zs[:250].T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[0, 10, 23.5, -0.5], interpolation="nearest")
axs[1].set_title("All 24 numbers of the code, frame by frame (first 10 s)", loc="left"); axs[1].set_xlabel("seconds"); axs[1].set_ylabel("channel")
fig.tight_layout(); fig.savefig(f"{OUTD}/embedding.png", dpi=110); plt.close(fig)

# 3. notes -> performance: written notes vs the performed pitch and loudness the curve model wrote
cv = np.load(f"{R}/midi/maisky_from_midi_NEW_vibrato_curves.npy"); pmid = pretty_midi.PrettyMIDI(f"{R}/curve3/maisky_30s_guessed.mid")
fig, ax = plt.subplots(figsize=(12, 3.6)); tt = np.arange(len(cv)) / 25; lo, hi = 0, 10
for nt in pmid.instruments[0].notes:
    if nt.start < hi: ax.plot([nt.start, min(nt.end, hi)], [nt.pitch, nt.pitch], color=ORANGE, lw=5, solid_capstyle="butt", alpha=0.85, zorder=1)
perf = np.where(cv[:, 1] > 0.5, 45 + 12 * cv[:, 0], np.nan)   # curve pitch = log2(f / 110 Hz); 110 Hz = MIDI 45; m = (tt >= lo) & (tt < hi); ax.plot(tt[m], perf[m], color=BLUE, lw=1.4, zorder=2)
ax.set_xlim(lo, hi); ax.set_xlabel("seconds"); ax.set_ylabel("pitch (MIDI note)"); ax.set_title("Written notes (orange) and how the model performs them (blue): slides, vibrato, note shaping", loc="left")
fig.tight_layout(); fig.savefig(f"{OUTD}/notes_to_performance.png", dpi=110); plt.close(fig)

# 4. a generated piece: piano roll + spectrogram
gm = f"{R}/symlm/lyric2/air_dminor_34_tau06"; pm = pretty_midi.PrettyMIDI(gm + ".mid"); ga, _ = sf.read(gm + ".wav"); ga = ga[: 30 * SR]
fig, axs = plt.subplots(2, 1, figsize=(12, 5), sharex=True, gridspec_kw={"height_ratios": [1, 1.2]})
for nt in pm.instruments[0].notes:
    if nt.start < 30: axs[0].add_patch(plt.Rectangle((nt.start, nt.pitch - 0.4), min(nt.end, 30) - nt.start, 0.8, color=BLUE, lw=0))
ps_ = [nt.pitch for nt in pm.instruments[0].notes]; axs[0].set_ylim(min(ps_) - 2, max(ps_) + 2); axs[0].set_ylabel("pitch"); axs[0].set_title("A piece the model composed (D minor, 3/4): the notes it wrote...", loc="left")
f_, t_, Sd = spec_db(ga.astype(np.float32)); mm = f_ <= 4000; axs[1].pcolormesh(t_, f_[mm] / 1000, Sd[mm], shading="auto", cmap="magma", vmin=Sd[mm].max() - 80, vmax=Sd[mm].max())
axs[1].set_title("...and the cello sound it then generated for them", loc="left"); axs[1].set_ylabel("frequency (kHz)"); axs[1].set_xlabel("seconds"); axs[1].set_xlim(0, 30)
fig.tight_layout(); fig.savefig(f"{OUTD}/generated_piece.png", dpi=110); plt.close(fig)

# 5. short audio clips (wav for the notebook, mp4 for GitHub)
clips = {"3_performed_from_notes_maisky": f"{R}/midi/maisky_from_midi_NEW_fixed.wav", "4_composed_air_dminor": gm + ".wav",
         "5_composed_air_gmajor": f"{R}/symlm/lyric2/air_gmajor_44_tau09.wav", "6_composed_chorale_dminor": f"{R}/symlm/lyric2/chorale_dminor_44_tau08.wav",
         "7_composed_suite_style_gmajor": f"{R}/symlm/pieces/gmajor_allemande_44.wav"}
for name, src in clips.items():
    a_, sr_ = sf.read(src); a_ = a_[: 40 * sr_]; fade = np.minimum(1, (len(a_) - np.arange(len(a_))) / (sr_ * 1.5)); sf.write(f"{AUD}/{name}.wav", (a_ * fade).astype(np.float32), sr_)
for wv in sorted(os.listdir(AUD)):
    if wv.endswith(".wav"): mp4(f"{AUD}/{wv}", f"{AUD}/{wv[:-4]}.mp4")
shutil.copy(f"{R}/reg/sensitivity_heatmap.png", f"{OUTD}/channel_sensitivity.png")
print("assets:", sorted(os.listdir(OUTD)), "| audio:", sorted(os.listdir(AUD)))
