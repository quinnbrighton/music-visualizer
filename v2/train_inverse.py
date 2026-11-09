"""Learned mel->linear inverse: 512-bin log-mel (4096/512 @44.1k) -> 2049-bin log-magnitude. Half the inputs are the nested AE's own reconstructions."""
import json, random, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa, soundfile as sf
from model import AE, B
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 6000
random.seed(0); torch.manual_seed(0); dev = "cuda"
ck = torch.load(f"{B}/multirate_nested.pt"); ae = AE(*ck["args"], rates=ck["rates"]).to(dev).eval(); ae.load_state_dict(ck["model"])
mean, std = ck["mean"].to(dev), ck["std"].to(dev); idx = json.load(open(f"{B}/mel512_index.json"))
files = [idx[k]["src"] for k in ck["train"] if sf.info(idx[k]["src"]).samplerate == 44100]
print(len(files), "train files at 44.1 kHz", flush=True)
mb = torch.from_numpy(librosa.filters.mel(sr=44100, n_fft=4096, n_mels=512)).float().to(dev); win = torch.hann_window(4096, device=dev)
LF = 1024; NS = LF * 512
def seg():
    f = random.choice(files); n = sf.info(f).frames; i = random.randrange(0, n - NS - 1)
    x, _ = sf.read(f, start=i, frames=NS, dtype="float32", always_2d=True); return torch.from_numpy(x.mean(1))
def lin(x): return torch.stft(x, 4096, 512, window=win, return_complex=True).abs()      # (B, 2049, T)
class Inv(nn.Module):
    def __init__(s, c=768):
        super().__init__()
        s.net = nn.Sequential(nn.Conv1d(512, c, 5, padding=2), nn.GELU(), nn.Conv1d(c, c, 5, padding=2), nn.GELU(), nn.Conv1d(c, c, 5, padding=4, dilation=2), nn.GELU(), nn.Conv1d(c, 2049, 3, padding=1))
        s.register_buffer("pinv", torch.linalg.pinv(mb))
    def forward(s, L):                                          # L: (B,512,T) natural-log mel
        base = torch.log((torch.einsum("lm,bmt->blt", s.pinv, L.clamp(-12, 9).exp())).clamp_min(1e-5))     # pseudo-inverse as the starting point
        return (base + s.net((L - mean[None, :, None]) / std[None, :, None])).clamp(-12, 9)
net = Inv().to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-4); sch = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-4, total_steps=STEPS, pct_start=0.05)
print(sum(p.numel() for p in net.parameters()) / 1e6, "M params", flush=True); t0 = time.time()
for s in range(1, STEPS + 1):
    x = torch.stack([seg() for _ in range(6)]).to(dev); S = lin(x); Lt = torch.log(mb @ S + 1e-5)       # true mel (B,512,T)
    with torch.no_grad():
        xin = ((Lt[:3] - mean[None, :, None]) / std[None, :, None]); rec = ae.decode(ae.enc_frame(xin)[0]) * std[None, :, None] + mean[None, :, None]
    Lin = torch.cat([rec, Lt[3:] + 0.05 * torch.randn_like(Lt[3:])], 0)
    pred = net(Lin); tgt = torch.log(S + 1e-5)
    mel_cons = F.l1_loss(torch.log(torch.einsum("ml,blt->bmt", mb, pred.clamp(-12, 9).exp()) + 1e-5), Lin)     # pred must re-project to the input mel
    loss = F.l1_loss(pred, tgt) + 0.5 * mel_cons
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5); opt.step(); sch.step()
    if s % 250 == 0: print(s, round(loss.item(), 4), "lin L1", round(F.l1_loss(pred, tgt).item(), 4), round(time.time() - t0), "s", flush=True)
torch.save(net.state_dict(), f"{B}/inv_net.pt"); print("saved")
