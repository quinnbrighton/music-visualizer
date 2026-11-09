"""Stage 1: fine-tune the Vocos backbone+ISTFT head on cello (true mel -> waveform).
Loss: multi-resolution STFT (spectral convergence + log-mag L1) plus 100-bin log-mel L1. No GAN (kept simple; stated limitation)."""
import glob, hashlib, os, sys, time, json, random
import numpy as np, torch, torch.nn.functional as F, torchaudio, soundfile as sf
from vocos import Vocos
from metrics import logmel_dist
B = os.path.expanduser("~/music-visualizer"); STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
random.seed(0); torch.manual_seed(0)
files, seen = [], set()
for f in sorted(glob.glob(f"{B}/cleaned/cello_dataset/*.wav")) + sorted(glob.glob(f"{B}/cello_dataset/*.wav")) + [f"{B}/downloads/Yo-Yo Ma Bach Cello Suite No.1 in G Major.wav"]:
    h = hashlib.md5(open(f, "rb").read(1 << 22)).hexdigest() + str(os.path.getsize(f))
    if h not in seen and "maisky" not in f.lower(): seen.add(h); files.append(f)
val_files = files[::15]; train_files = [f for f in files if f not in val_files]
def load(f):
    x, sr = sf.read(f, dtype="float32", always_2d=True)
    return torchaudio.functional.resample(torch.from_numpy(x.mean(1)), sr, 24000).half()
train = [load(f) for f in train_files]; val = []
for f in val_files:
    w = load(f)
    if len(w) > 24000 * 40: val.append(w[24000 * 10: 24000 * 40])
    if len(val) == 4: break
print(len(train_files), "train files,", len(val_files), "val files", flush=True)
voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda()
fe = voc.feature_extractor
opt = torch.optim.AdamW(list(voc.backbone.parameters()) + list(voc.head.parameters()), lr=5e-5, betas=(0.8, 0.9))
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
def stft_loss(y, x):
    l = 0
    for n in (512, 1024, 2048):
        w = torch.hann_window(n, device=y.device)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs(); X = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
        l = l + torch.norm(X - Y) / torch.norm(X) + (torch.log(X + 1e-5) - torch.log(Y + 1e-5)).abs().mean()
    return l / 3
def batch(bs=16, L=24000):
    out = []
    for _ in range(bs):
        w = random.choice(train); i = random.randrange(0, len(w) - L); out.append(w[i:i + L].float())
    return torch.stack(out).cuda()
def evaluate():
    voc.eval(); r = []
    with torch.no_grad():
        for w in val:
            w = w.float().cuda(); y = voc.decode(fe(w[None]))[0]; r.append(logmel_dist(y.cpu(), w.cpu()))
    voc.train(); return {k: float(np.mean([x[k] for x in r])) for k in r[0]}
log = {"val_before": evaluate()}; print(log, flush=True); t0 = time.time()
voc.train()
for s in range(1, STEPS + 1):
    x = batch()
    with torch.autocast("cuda", enabled=False):
        with torch.no_grad(): m = fe(x)
        y = voc.head(voc.backbone(m))
    y = y.float()[:, : x.shape[1]]; x = x[:, : y.shape[1]]
    loss = stft_loss(y, x) + 5 * F.l1_loss(torch.log(fe(y).clamp_min(1e-5)), torch.log(m.float().clamp_min(1e-5)))
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(voc.parameters(), 5); opt.step(); sched.step()
    if s % 250 == 0: print(s, round(loss.item(), 4), round(time.time() - t0), "s", flush=True)
    if s % 4000 == 0 or s == STEPS:
        log[f"val_{s}"] = evaluate(); print(log[f"val_{s}"], flush=True)
        torch.save({"backbone": voc.backbone.state_dict(), "head": voc.head.state_dict()}, f"{B}/v2/vocos_cello.pt")
json.dump({**log, "train_files": len(train_files), "val_files": [os.path.basename(f) for f in val_files], "steps": STEPS}, open(f"{B}/v2/finetune_vocos.json", "w"), indent=1)
