"""Discriminators for adversarial fine-tuning of the harmonic vocoder.

MPD: HiFi-GAN multi-period discriminator. The waveform is folded at each period so 2D convs see cycle-level structure.
MBD: Descript-audio-codec style multi-band, multi-scale complex STFT discriminator. Real and imaginary parts, split into frequency
bands with separate conv stacks, at several window sizes, so it sees blur and missing detail at several time and frequency scales.
Each discriminator returns a list of (score, feature_maps) pairs, one per sub-discriminator.
"""
import torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm


class DiscP(nn.Module):
    def __init__(self, period, ch=(32, 128, 512, 1024, 1024)):
        super().__init__(); self.p = period; layers, i = [], 1
        for k, o in enumerate(ch): layers.append(weight_norm(nn.Conv2d(i, o, (5, 1), (3, 1) if k < len(ch) - 1 else (1, 1), padding=(2, 0)))); i = o
        self.convs = nn.ModuleList(layers); self.out = weight_norm(nn.Conv2d(i, 1, (3, 1), padding=(1, 0)))

    def forward(self, x):                                                   # x (B,T)
        b, t = x.shape
        if t % self.p: x = F.pad(x, (0, self.p - t % self.p), mode="reflect")
        h = x.view(b, 1, -1, self.p); fm = []
        for c in self.convs: h = F.leaky_relu(c(h), 0.1); fm.append(h)
        h = self.out(h); fm.append(h); return h.flatten(1), fm


class MPD(nn.Module):
    def __init__(self, periods=(2, 3, 5, 7, 11)):
        super().__init__(); self.ds = nn.ModuleList([DiscP(p) for p in periods])

    def forward(self, x): return [d(x) for d in self.ds]


BANDS = ((0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))


class DiscB(nn.Module):
    def __init__(self, win, ch=32):
        super().__init__(); self.win, self.hop = win, win // 4; self.register_buffer("w", torch.hann_window(win)); nb = win // 2 + 1
        self.edges = [(int(a * nb), int(b * nb)) for a, b in BANDS]
        def stack(): return nn.ModuleList([weight_norm(nn.Conv2d(2, ch, (3, 9), padding=(1, 4)))] + [weight_norm(nn.Conv2d(ch, ch, (3, 9), (1, 2), padding=(1, 4))) for _ in range(3)] + [weight_norm(nn.Conv2d(ch, ch, (3, 3), padding=(1, 1)))])
        self.bands = nn.ModuleList([stack() for _ in BANDS]); self.out = weight_norm(nn.Conv2d(ch, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        S = torch.stft(x, self.win, self.hop, window=self.w, return_complex=True); S = torch.view_as_real(S).permute(0, 3, 2, 1)     # (B,2,T,F)
        fm, hs = [], []
        for (a, b), st in zip(self.edges, self.bands):
            h = S[..., a:b]
            for c in st: h = F.leaky_relu(c(h), 0.1); fm.append(h)
            hs.append(h)
        h = self.out(torch.cat(hs, -1)); fm.append(h); return h.flatten(1), fm


class MBD(nn.Module):
    def __init__(self, wins=(2048, 1024, 512)):
        super().__init__(); self.ds = nn.ModuleList([DiscB(w) for w in wins])

    def forward(self, x): return [d(x) for d in self.ds]


if __name__ == "__main__":
    x = torch.randn(2, 24000)
    for D in (MPD(), MBD()):
        out = D(x); print(type(D).__name__, [tuple(s.shape) for s, _ in out], "params %.1fM" % (sum(p.numel() for p in D.parameters()) / 1e6))
