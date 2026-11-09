"""README demo, 5 s vs 5 s: the held-out Maisky passage with plain volume turned up (A = recon * the push's true level gain) vs the
+20 latent loudness push (B), level-matched over the 12 s so only tone differs. Picks the 5 s window (0.25 s steps) where A and B
differ most (mean |log-mel diff| after per-window level match) and writes docs/assets/audio/12_volume_vs_latent_5s.{wav,mp4}.
usage: cd colab/cello_gan; CELLO_DIR=.. TOK_DIR=.. OUT_DIR=.. INIT=harm_final_z24lo_s2000 INZD=24 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5 \
       BETA=0 AMPG=0 BS=2 NQ=12 K=64 PYTHONPATH=$CELLO_DIR python make_push_window.py      (env: as make_push_demo.py)"""
import os, sys, subprocess, tempfile
sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import soundfile as sf, librosa
from PIL import Image, ImageDraw, ImageFont
H = os.path.expanduser("~/music-visualizer"); R = f"{H}/colab_results"; LAT = f"{R}/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; AUD = f"{H}/docs/assets/audio"
c_ = torch.load(f"{R}/reg3/codec_snd_s8000.pt", map_location=dev, weights_only=False); model.load_state_dict(c_["model"]); model.eval()
spl = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); k = spl["test"][0]
rng = np.random.default_rng(0); Z, S, L, RC = [], [], [], []               # loudness direction, exactly as in make_push_demo.py (within recordings, pitch removed)
for t in spl["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); c = np.load(f"{CTL}/{t}.npy"); n = min(len(z), len(c)); pick = rng.choice(n, min(800, n), replace=False)
    Z.append(z[pick]); S.append(np.where(c[pick, 1] > 0.5, np.round(69 + 12 * np.log2(np.clip(c[pick, 0], 40, 1200) / 440)), -1)); L.append(c[pick, 2]); RC.append(np.full(len(pick), len(RC)))
Z, S, L, RC = np.concatenate(Z), np.concatenate(S).astype(int), np.concatenate(L), np.concatenate(RC); r = Z - Z.mean(0); Lc = L - L.mean()
for g in np.unique(RC): r[RC == g] -= r[RC == g].mean(0); Lc[RC == g] -= Lc[RC == g].mean()
for g in np.unique(S): r[S == g] -= r[S == g].mean(0)
dirL = (r * Lc[:, None]).sum(0) / (Lc ** 2).sum()


@torch.no_grad()
def dec_ola(zz, W=100, Hh=75):                                              # 4 s overlap-add decode windows, as make_push_demo.py
    zz = torch.from_numpy(zz.astype(np.float32)).to(dev); nn_ = zz.shape[0]; Sm = HOP * DSR; out = np.zeros(nn_ * Sm); ws = np.zeros(nn_ * Sm); st = list(range(0, max(1, nn_ - W) + 1, Hh))
    if st[-1] + W < nn_: st.append(nn_ - W)
    for a in st:
        e = min(nn_, a + W); TT = (e - a) * DSR; torch.manual_seed(0); hh = dec_feats(zz[a:e].t()[None], TT); y = model.synth(hh, model.pitch_f0(hh), TT)[0].cpu().numpy()
        w = np.ones(len(y)); rr = (W - Hh) * Sm
        if a > 0: w[:rr] = np.linspace(0, 1, rr)
        if e < nn_: w[-rr:] = np.linspace(1, 0, rr)
        out[a * Sm: a * Sm + len(y)] += y * w; ws[a * Sm: a * Sm + len(y)] += w
    return out / np.maximum(ws, 1e-6)


f0 = 5572800 // 960; zz = np.load(f"{LAT}/{k}.npy").astype(np.float32)[f0: f0 + 300]   # same 12 s as clips 1/11 (232.2 s into the recording)
y0, y1 = dec_ola(zz), dec_ola(zz + 20 * dirL); rms = lambda v: np.sqrt(np.mean(v ** 2))
gdb = 20 * np.log10(rms(y1) / rms(y0)); print("true level shift %.2f dB" % gdb)
A, B = y0 * rms(y1) / rms(y0), y1                                           # A = plain volume by the true gain; same 12 s RMS as B by construction
fb = librosa.filters.mel(sr=SR, n_fft=2048, n_mels=128, fmin=30, fmax=SR / 2)
spec = lambda v: np.abs(librosa.stft(v.astype(np.float32), n_fft=2048, hop_length=240)) ** 2
fr = librosa.fft_frequencies(sr=SR, n_fft=2048); lo, hi = fr < 1000, (fr >= 1000) & (fr < 4000)


def measures(a, b):                                                         # b level-matched to a in this window; returns |logmel diff| dB, hi/lo ratio diff dB, centroid diff Hz
    b = b * rms(a) / rms(b); Pa, Pb = spec(a), spec(b); Ma, Mb = 10 * np.log10(fb @ Pa + 1e-10), 10 * np.log10(fb @ Pb + 1e-10)
    fl = max(Ma.max(), Mb.max()) - 80; dm = np.abs(np.maximum(Ma, fl) - np.maximum(Mb, fl)).mean()
    hl = lambda P: 10 * np.log10(P[hi].sum() / P[lo].sum()); cen = lambda P: (fr[:, None] * P).sum() / P.sum()   # energy-weighted over the window
    return dm, hl(Pb) - hl(Pa), cen(Pb) - cen(Pa)


Wn, st = 5 * SR, int(0.25 * SR); T0 = np.arange(0, len(A) - Wn + 1, st); M = np.array([measures(A[t: t + Wn], B[t: t + Wn]) for t in T0])
for t, m in zip(T0, M): print("%5.2f s  logmel %.2f dB  hi/lo %+.2f dB  centroid %+.0f Hz" % (t / SR, *m))
i = int(M[:, 0].argmax()); t0 = T0[i]; med = np.median(M, 0)
print("BEST %.2f-%.2f s (abs %.2f s): logmel %.2f dB (median %.2f), hi/lo %+.2f dB (median %+.2f), centroid %+.0f Hz (median %+.0f)"
      % (t0 / SR, t0 / SR + 5, 232.2 + t0 / SR, M[i, 0], med[0], M[i, 1], med[1], M[i, 2], med[2]))
a, b = A[t0: t0 + Wn], B[t0: t0 + Wn]; b = b * rms(a) / rms(b)                # per-window level match so the 5 s pair is loudness-equal
g = 0.9 / max(np.abs(a).max(), np.abs(b).max()); out = np.concatenate([a * g, np.zeros(int(0.8 * SR)), b * g]); sf.write(f"{AUD}/12_volume_vs_latent_5s.wav", out.astype(np.float32), SR)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22); tmp = tempfile.gettempdir(); l1, l2 = f"{tmp}/_w1.png", f"{tmp}/_w2.png"
for nm, txt in ((l1, "1. Volume turned up"), (l2, "2. Latent loudness push (same level)")):
    im = Image.new("RGBA", (720, 40), (0, 0, 0, 0)); ImageDraw.Draw(im).text((16, 8), txt, font=font, fill=(255, 255, 255, 255)); im.save(nm)
D = Wn / SR + 0.4                                                           # draw=full: the default draw=scale renders 0x2a78d6 as green
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{AUD}/12_volume_vs_latent_5s.wav", "-loop", "1", "-i", l1, "-loop", "1", "-i", l2, "-filter_complex",
                f"[0:a]showwaves=s=720x200:mode=cline:draw=full:colors=0x2a78d6,format=yuv420p[w];[w][1:v]overlay=0:0:enable='lt(t,{D})':shortest=1[w1];[w1][2:v]overlay=0:0:enable='gte(t,{D})':shortest=1,format=yuv420p[v]",
                "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "28", "-c:a", "aac", "-b:a", "160k", "-shortest", f"{AUD}/12_volume_vs_latent_5s.mp4"], check=True)
print("wrote 12_volume_vs_latent_5s.{wav,mp4}")
