"""Level-matched comparison video of systems on the same 15 s held-out Maisky window (232.2 s). usage: build_video.py OUT.mp4 key=path|label ..."""
import json, os, subprocess, sys, numpy as np, soundfile as sf, torch, torchaudio
from PIL import Image, ImageDraw, ImageFont
from metrics import logmel_dist
B = "/home/qbrighto/music-visualizer/v2"; SR = 24000; n = 15 * SR; s0 = 5572800
k = torch.load(f"{B}/multirate_nested.pt", map_location="cpu")["test"][0]
x = torch.from_numpy(np.asarray(np.load(f"{B}/audio24k/{k}.npy", mmap_mode="r")[s0:s0 + n], dtype=np.float32)); r0 = x.pow(2).mean().sqrt()
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26); segs = []
for i, spec in enumerate(sys.argv[2:]):
    path, label = spec.split("|", 1)
    if path == "ORIG": y = x.clone(); err = None
    else:
        w, sr = sf.read(f"{B}/{path}", dtype="float32"); y = torch.from_numpy(w if w.ndim == 1 else w.mean(1)); y = torchaudio.functional.resample(y, sr, SR) if sr != SR else y; y = y[:n]
        y = y * (r0 / (y.pow(2).mean().sqrt() + 1e-9)); y = y / max(1.0, float(y.abs().max()) / 0.95); err = logmel_dist(y, x)["lm"]
    wav = f"{B}/report/seg{i}.wav"; sf.write(wav, y.numpy(), SR)
    text = f"{i+1}. {label}" + ("" if err is None else f"   [log-mel err {err:.2f}]")
    im = Image.new("RGBA", (1280, 540), (0, 0, 0, 0)); d = ImageDraw.Draw(im); d.rectangle([0, 0, 1280, 46], fill=(0, 0, 0, 175)); d.text((12, 8), text, font=font, fill=(255, 255, 255, 255)); im.save(f"{B}/report/lab{i}.png")
    mp4 = f"{B}/report/seg{i}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav, "-i", f"{B}/report/lab{i}.png", "-filter_complex", "[0:a]showspectrum=s=1280x540:color=intensity:scale=log:fscale=log:legend=0,format=yuva420p[v];[v][1:v]overlay=0:0,format=yuv420p[vv]", "-map", "[vv]", "-map", "0:a", "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-b:a", "160k", "-shortest", mp4], check=True); segs.append(mp4); print(text, flush=True)
open(f"{B}/report/list.txt", "w").write("".join(f"file '{s}'\n" for s in segs))
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", f"{B}/report/list.txt", "-c", "copy", f"{B}/report/{sys.argv[1]}"], check=True); print("wrote report/" + sys.argv[1])
