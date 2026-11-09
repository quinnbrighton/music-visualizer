"""README demo of editing through the latent: the held-out Maisky passage decoded as is, then pushed +20 along the loudness direction,
level-matched and played back to back. Writes docs/assets/audio/11_loudness_push.{wav,mp4}.   env: as gen_lat.py (codec skeleton)"""
import os, sys, subprocess
sys.argv = ["x", "free", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont
H = os.path.expanduser("~/music-visualizer"); R = f"{H}/colab_results"; LAT = f"{R}/reg3/tok/latents"; CTL = f"{H}/tokens64/controls"; AUD = f"{H}/docs/assets/audio"
c_ = torch.load(f"{R}/reg3/codec_snd_s8000.pt", map_location=dev, weights_only=False); model.load_state_dict(c_["model"]); model.eval()
spl = torch.load(f"{H}/colab/cello_gan/multirate_nested.pt", map_location="cpu", weights_only=False); k = spl["test"][0]
rng = np.random.default_rng(0); Z, S, L, RC = [], [], [], []               # loudness direction, exactly as in loudness_push_stats.py (within recordings)
for t in spl["train"]:
    if not os.path.exists(f"{LAT}/{t}.npy"): continue
    z = np.load(f"{LAT}/{t}.npy").astype(np.float32); c = np.load(f"{CTL}/{t}.npy"); n = min(len(z), len(c)); pick = rng.choice(n, min(800, n), replace=False)
    Z.append(z[pick]); S.append(np.where(c[pick, 1] > 0.5, np.round(69 + 12 * np.log2(np.clip(c[pick, 0], 40, 1200) / 440)), -1)); L.append(c[pick, 2]); RC.append(np.full(len(pick), len(RC)))
Z, S, L, RC = np.concatenate(Z), np.concatenate(S).astype(int), np.concatenate(L), np.concatenate(RC); r = Z - Z.mean(0); Lc = L - L.mean()
for g in np.unique(RC): r[RC == g] -= r[RC == g].mean(0); Lc[RC == g] -= Lc[RC == g].mean()
for g in np.unique(S): r[S == g] -= r[S == g].mean(0)
dirL = (r * Lc[:, None]).sum(0) / (Lc ** 2).sum()


@torch.no_grad()
def dec_ola(zz, W=100, Hh=75):
    zz = torch.from_numpy(zz.astype(np.float32)).to(dev); nn_ = zz.shape[0]; Sm = HOP * DSR; out = np.zeros(nn_ * Sm); ws = np.zeros(nn_ * Sm); st = list(range(0, max(1, nn_ - W) + 1, Hh))
    if st[-1] + W < nn_: st.append(nn_ - W)
    for a in st:
        e = min(nn_, a + W); TT = (e - a) * DSR; torch.manual_seed(0); hh = dec_feats(zz[a:e].t()[None], TT); y = model.synth(hh, model.pitch_f0(hh), TT)[0].cpu().numpy()
        w = np.ones(len(y)); rr = (W - Hh) * Sm
        if a > 0: w[:rr] = np.linspace(0, 1, rr)
        if e < nn_: w[-rr:] = np.linspace(1, 0, rr)
        out[a * Sm: a * Sm + len(y)] += y * w; ws[a * Sm: a * Sm + len(y)] += w
    return out / np.maximum(ws, 1e-6)


f0 = 5572800 // 960; zz = np.load(f"{LAT}/{k}.npy").astype(np.float32)[f0: f0 + 300]   # the same 12 s as clip 1
y0, y1 = dec_ola(zz), dec_ola(zz + 20 * dirL); rms = lambda v: np.sqrt(np.mean(v ** 2))
print("true level shift %.1f dB" % (20 * np.log10(rms(y1) / rms(y0)))); y1 = y1 / rms(y1) * rms(y0)
g = 0.9 / max(np.abs(y0).max(), np.abs(y1).max()); out = np.concatenate([y0 * g, np.zeros(int(0.8 * SR)), y1 * g]); sf.write(f"{AUD}/11_loudness_push.wav", out.astype(np.float32), SR)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
for nm, txt in (("/tmp/_l1.png", "1. Reconstruction"), ("/tmp/_l2.png", "2. Latent loudness push, same level")):
    im = Image.new("RGBA", (720, 40), (0, 0, 0, 0)); ImageDraw.Draw(im).text((16, 8), txt, font=font, fill=(255, 255, 255, 255)); im.save(nm)
D = len(y0) / SR + 0.4
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", f"{AUD}/11_loudness_push.wav", "-loop", "1", "-i", "/tmp/_l1.png", "-loop", "1", "-i", "/tmp/_l2.png", "-filter_complex",
                f"[0:a]showwaves=s=720x200:mode=cline:colors=0x2a78d6,format=yuv420p[w];[w][1:v]overlay=0:0:enable='lt(t,{D})':shortest=1[w1];[w1][2:v]overlay=0:0:enable='gte(t,{D})':shortest=1,format=yuv420p[v]",
                "-map", "[v]", "-map", "0:a", "-c:v", "libx264", "-crf", "28", "-c:a", "aac", "-b:a", "160k", "-shortest", f"{AUD}/11_loudness_push.mp4"], check=True)
print("wrote 11_loudness_push.mp4")
