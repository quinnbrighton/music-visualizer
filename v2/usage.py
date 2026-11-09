"""Per-dimension latent usage on held-out audio: KL per dim (nats, bits the model spends) and how much each dim varies across time vs across recordings."""
import sys, numpy as np, torch
from model import AE, read
name = sys.argv[1]; ck = torch.load(f"{name}.pt"); FD, GD = ck["args"]
m = AE(FD, GD, rates=ck.get("rates")).cuda().eval(); m.load_state_dict(ck["model"]); mean, std = ck["mean"], ck["std"]
keys = ck["val"] + ck["test"]; KL, MU = [], {}
with torch.no_grad():
    for k in keys:
        x = ((read(k) - mean[:, None]) / std[:, None]).cuda()[None]
        mu, lv = m.enc_frame(x[:, :, :6000]); kl = 0.5 * (mu ** 2 + lv.exp() - 1 - lv)
        KL.append(kl[0].mean(1).cpu().numpy()); MU[k] = mu[0].cpu().numpy()
KL = np.mean(KL, 0); order = np.argsort(-KL)
allmu = np.concatenate([v for v in MU.values()], 1); tot = allmu.var(1)
within = np.mean([v.var(1) for v in MU.values()], 0); between = np.var([v.mean(1) for v in MU.values()], 0)
print(f"{name}: frame dims={FD} global dims={GD}; total KL {KL.sum():.2f} nats ({KL.sum()/np.log(2):.1f} bits per frame)")
print("per-frame dims, most to least used (KL nats | var over time | var across recordings):")
for d in order: print(f"  dim {d:2d}: KL {KL[d]:.3f} | within {within[d]:.3f} | between {between[d]:.4f}")
print("active dims (KL > 0.05 nats):", int((KL > 0.05).sum()), "of", FD)
if GD:
    with torch.no_grad():
        gs = []
        for k in keys:
            x = ((read(k) - mean[:, None]) / std[:, None]).cuda()[None]
            f = torch.stack([m.enc_global_feat(x[:, :, i:i + 860])[0] for i in range(0, x.shape[2] - 860, 860)]).mean(0, keepdim=True)
            g = m.global_from_feat(f); gk = 0.5 * (g[0] ** 2 + g[1].exp() - 1 - g[1]); gs.append((g[0][0].cpu().numpy(), gk[0].cpu().numpy()))
    print("global code KL per dim:", np.round(np.mean([g[1] for g in gs], 0), 3), " spread of global means across recordings:", np.round(np.std([g[0] for g in gs], 0), 3))
