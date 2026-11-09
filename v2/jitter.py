import json, numpy as np, torch, torch.nn.functional as F
from model import AE, read
ck = torch.load("multirate_nested.pt"); m = AE(*ck["args"], rates=ck["rates"]).cuda().eval(); m.load_state_dict(ck["model"]); std = ck["std"].cuda()
fs = 44100 / 512
def jit(L):            # mean |frame-to-frame change| of log-mel, and share of temporal-modulation power above 10 Hz
    d = (L[:, 1:] - L[:, :-1]).abs().mean().item()
    P = torch.fft.rfft(L - L.mean(1, keepdim=True), dim=1).abs() ** 2; f = torch.fft.rfftfreq(L.shape[1], 1 / fs).cuda()
    return round(d, 4), round((P[:, f > 10].sum() / P.sum()).item(), 4)
def smooth(y, w): return y if w == 1 else F.avg_pool1d(F.pad(y[None], (w // 2, w // 2), mode="replicate"), w, stride=1)[0]
out = {}
with torch.no_grad():
    for k in ck["val"][:3] + ck["test"]:
        X = ((read(k) - ck["mean"][:, None]) / ck["std"][:, None]).cuda()[:, :4096]
        mu = m.enc_frame(X[None])[0]; y = m.decode(mu)[0]
        r = {"true": jit(X * std[:, None]), "recon": jit(y * std[:, None]), "recon_err": ((y - X) * std[:, None]).abs().mean().item()}
        for w in (3, 5, 9):
            ys = smooth(y, w); r[f"recon_smooth{w}"] = jit(ys * std[:, None]); r[f"err_smooth{w}"] = ((ys - X) * std[:, None]).abs().mean().item()
        # smoothing only the fast latent groups (dims 0-10) instead of the output
        mu2 = mu.clone(); mu2[:, :11] = F.avg_pool1d(F.pad(mu[:, :11], (2, 2), mode="replicate"), 5, stride=1); y2 = m.decode(mu2)[0]
        r["fastlatent_smooth5"] = jit(y2 * std[:, None]); r["err_fastlatent5"] = ((y2 - X) * std[:, None]).abs().mean().item()
        out[k[:8]] = r
keys = list(out[next(iter(out))])
print("each cell: (mean |frame-to-frame change|, share of power above 10 Hz); errors in log-mel units; mean over 4 held-out recordings")
for key in keys:
    v = [out[k][key] for k in out]
    print(f"{key:20s}", (tuple(np.round(np.mean(v, 0), 4)) if isinstance(v[0], (tuple, list)) else round(float(np.mean(v)), 4)))
json.dump(out, open("jitter.json", "w"), indent=1)
