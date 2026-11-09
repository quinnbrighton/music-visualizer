"""v2 autoencoder on the original 512-bin log-mel front end.

Per-frame VAE latent (zf) + optional global code (zg) pooled over time from a clip of the SAME recording drawn from a
different time than the window being reconstructed, so zg can only carry what is constant within a recording.
"""
import json, os
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

B = os.path.expanduser("~/music-visualizer/v2")
N_MEL = 512


def load_data():
    idx = json.load(open(f"{B}/mel512_index.json"))
    keys = sorted(idx)
    test = [k for k in keys if "Maisky" in idx[k]["src"]]
    yym = [k for k in keys if "Yo-Yo Ma Bach Cello Suite No.1" in idx[k]["src"]]
    rest = [k for k in keys if k not in test and k not in yym and idx[k]["frames"] > 3000]
    val = rest[::13][:7]
    train = [k for k in rest if k not in val] + yym
    return idx, train, val, test


def read(k):
    return torch.from_numpy(np.load(f"{B}/mel512/{k}.npy")).float()   # (512, T) log-mel (natural log)


def norm_stats(train):
    s, s2, n = 0, 0, 0
    for k in train:
        x = read(k); s = s + x.sum(1); s2 = s2 + (x ** 2).sum(1); n += x.shape[1]
    m = s / n
    return m, (s2 / n - m ** 2).clamp_min(1e-4).sqrt()


def block(cin, cout, k=5, d=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d), nn.GroupNorm(8, cout), nn.GELU())


class Trunk(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.net = nn.Sequential(block(N_MEL, ch), block(ch, ch, 5, 2), block(ch, ch, 5, 4))   # RF ~ 57 frames (0.66 s)

    def forward(self, x):
        return self.net(x)


class AE(nn.Module):
    def __init__(self, frame_dim=16, global_dim=8, ch=256, rates=None):
        super().__init__()
        self.fd, self.gd = frame_dim, global_dim
        self.rates = rates   # e.g. [(6,1),(5,5),(5,21),(4,85),(4,341)]: (n_dims, odd box-filter width in frames)
        self.f_trunk = Trunk(ch); self.f_head = nn.Conv1d(ch, 2 * frame_dim, 1)
        if global_dim:
            self.g_trunk = Trunk(ch); self.g_head = nn.Linear(ch, 2 * global_dim)
        self.dec = nn.Sequential(block(frame_dim + global_dim, ch), block(ch, ch, 5, 2), block(ch, ch, 5, 4), nn.Conv1d(ch, N_MEL, 1))

    def enc_frame(self, x):
        mu, lv = self.f_head(self.f_trunk(x)).chunk(2, 1)
        if self.rates:
            mu, lv = self._smooth(mu), self._smooth(lv)
        return mu, lv.clamp(-8, 4)

    def _smooth(self, z):
        out, i = [], 0
        for n, w in self.rates:
            g = z[:, i:i + n]; i += n
            if w > 1:
                g = F.avg_pool1d(F.pad(g, (w // 2, w // 2), mode="replicate"), w, stride=1)
            out.append(g)
        return torch.cat(out, 1)

    def enc_global_feat(self, x):
        return self.g_trunk(x).mean(2)                       # (B, ch): linear in time -> chunkable at eval

    def global_from_feat(self, f):
        mu, lv = self.g_head(f).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def decode(self, zf, zg=None):
        if self.gd:
            zf = torch.cat([zf, zg[:, :, None].expand(-1, -1, zf.shape[2])], 1)
        return self.dec(zf)


def kl(mu, lv, free=0.02):
    k = 0.5 * (mu ** 2 + lv.exp() - 1 - lv)                  # per element
    k = k.mean(tuple(i for i in range(k.dim()) if i != 1)) if k.dim() == 3 else k.mean(0)
    return k.clamp_min(free).sum(), k.detach()
