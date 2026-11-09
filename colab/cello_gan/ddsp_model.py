"""Interpretable multi-timescale DDSP autoencoder for solo cello (24 kHz).

controls:   f0 (yin, explicit), loudness (explicit), multi-rate latents z (24 dims in groups smoothed over 0.01/0.05/0.25/1/3 s)
global:     one code per clip (8 dims) -> tone conditioning + a room IR built from interpretable (band x decay) gains
synthesis:  harmonic oscillator bank + filtered noise + room convolution. No phase retrieval, no vocoder.
"""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchaudio

SR, HOP, NH, NB = 24000, 240, 64, 65
IR_LEN = 12000                                     # 0.5 s
BANDS, TAUS = 8, (0.015, 0.05, 0.12, 0.25)        # reverb basis: 8 frequency bands x 4 decay times
RATES = [(6, 1), (5, 5), (5, 25), (4, 101), (4, 301)]    # (dims, box-filter width in 100 Hz frames)


def block(cin, cout, k=5, d=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d), nn.GroupNorm(8, cout), nn.GELU())


def frames_to_samples(x, n):                       # (B,T,C) -> (B,n,C), linear interpolation
    return F.interpolate(x.transpose(1, 2), size=n, mode="linear", align_corners=False).transpose(1, 2)


def harmonic_synth(f0, amp, dist_logits, n):
    """f0 (B,T,1) Hz; amp (B,T,1) overall level; dist_logits (B,T,NH). Returns (B,n)."""
    k = torch.arange(1, NH + 1, device=f0.device)
    ok = (f0 * k < SR / 2 * 0.95).float()                                          # no aliasing
    dist = torch.softmax(dist_logits, -1) * ok
    dist = dist / (dist.sum(-1, keepdim=True) + 1e-6)
    A = frames_to_samples(amp * dist, n); f = frames_to_samples(f0, n)[..., 0]
    phi = torch.cumsum(2 * math.pi * f / SR, 1)
    return (A * torch.sin(phi[..., None] * k)).sum(-1)


def filtered_noise(mag, n):
    """mag (B,T,NB) non-negative band gains -> noise shaped per frame (STFT-domain filter)."""
    win = torch.hann_window(2 * HOP, device=mag.device)
    S = torch.stft(torch.randn(mag.shape[0], n, device=mag.device), 2 * HOP, HOP, window=win, return_complex=True)
    Bn, T, _ = mag.shape
    m = F.interpolate(mag.reshape(Bn * T, 1, NB), size=HOP + 1, mode="linear", align_corners=False).reshape(Bn, T, HOP + 1).transpose(1, 2)   # bands -> STFT bins
    m = F.interpolate(m, size=S.shape[-1], mode="linear", align_corners=False)                                                               # frames -> STFT frames
    return torch.istft(S * m, 2 * HOP, HOP, window=win, length=n)


class Reverb(nn.Module):
    """Room = dry path + sum of (band x decay) noise tails weighted by gains that come from the global code (interpretable: 32 numbers)."""
    def __init__(self, gdim):
        super().__init__()
        g = torch.Generator().manual_seed(0); t = torch.arange(IR_LEN) / SR
        base = torch.randn(BANDS * len(TAUS), IR_LEN, generator=g)
        edges = torch.logspace(math.log10(100), math.log10(10000), BANDS + 1)
        Fq = torch.fft.rfftfreq(IR_LEN, 1 / SR)
        for b in range(BANDS):
            for d, tau in enumerate(TAUS):
                H = ((Fq >= edges[b]) & (Fq < edges[b + 1])).float()
                h = torch.fft.irfft(torch.fft.rfft(base[b * len(TAUS) + d]) * H, n=IR_LEN) * torch.exp(-t / tau)
                base[b * len(TAUS) + d] = h / (h.norm() + 1e-8)
        self.register_buffer("base", base)
        self.net = nn.Sequential(nn.Linear(gdim, 64), nn.GELU(), nn.Linear(64, BANDS * len(TAUS)))
        nn.init.constant_(self.net[-1].bias, -4.0)

    def gains(self, zg):
        return F.softplus(self.net(zg))

    def forward(self, x, zg):
        g = self.gains(zg)                                              # (B,32)
        ir = torch.einsum("bk,kt->bt", g, self.base); ir = F.pad(ir, (1, 0))[:, :IR_LEN]; ir[:, 0] = ir[:, 0] + 1.0       # + dry path
        n = x.shape[1] + IR_LEN
        return torch.fft.irfft(torch.fft.rfft(x, n) * torch.fft.rfft(ir, n), n)[:, : x.shape[1]], g


class DDSPAE(nn.Module):
    def __init__(self, zdim=24, gdim=8, ch=256, rates=RATES, use_global=True, f0res=False):
        super().__init__()
        self.f0res, self.h = f0res, None
        self.zdim, self.gdim, self.rates, self.use_global = zdim, gdim, rates, use_global
        self.mel = torchaudio.transforms.MelSpectrogram(SR, 1024, 1024, HOP, n_mels=128, power=1.0)
        self.trunk = nn.Sequential(block(128, ch), block(ch, ch, 5, 2), block(ch, ch, 5, 4)); self.head = nn.Conv1d(ch, 2 * zdim, 1)
        if use_global:
            self.gtrunk = nn.Sequential(block(128, ch), block(ch, ch, 5, 2), block(ch, ch, 5, 4)); self.ghead = nn.Linear(ch, 2 * gdim)
            self.reverb = Reverb(gdim)
        gd = gdim if use_global else 0
        if f0res:
            self.f0head = nn.Conv1d(ch, 1, 1); nn.init.zeros_(self.f0head.weight); nn.init.zeros_(self.f0head.bias)     # starts as identity: f0 correction within +-1 semitone
        self.gru = nn.GRU(2 + zdim + gd, 256, batch_first=True)
        self.out = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NH + 1 + NB))
        nn.init.constant_(self.out[-1].bias, 0.0)

    def logmel(self, x):                                                 # (B,N) -> (B,128,T)
        return ((torch.log(self.mel(x) + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP]

    def smooth(self, z):
        out, i = [], 0
        for n, w in self.rates:
            g = z[:, i:i + n]; i += n
            out.append(g if w == 1 else F.avg_pool1d(F.pad(g, (w // 2, w // 2), mode="replicate"), w, stride=1))
        return torch.cat(out, 1)

    def encode(self, x):
        h = self.trunk(self.logmel(x)); self.h = h
        mu, lv = self.head(h).chunk(2, 1)
        return self.smooth(mu), self.smooth(lv).clamp(-8, 4)             # (B,zdim,T)

    def encode_global(self, x):
        mu, lv = self.ghead(self.gtrunk(self.logmel(x)).mean(2)).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def decode(self, f0, loud, z, zg, n):
        """f0 (B,T) Hz, loud (B,T) normalized dB, z (B,zdim,T), zg (B,gdim) or None. Returns audio (B,n) and a dict of controls."""
        T = z.shape[2]
        if self.f0res and self.h is not None and self.h.shape[2] == T:
            f0 = f0 * 2 ** (torch.tanh(self.f0head(self.h))[:, 0] / 12)
        c = [torch.log2(f0.clamp(40, 1200) / 110)[..., None], loud[..., None], z.transpose(1, 2)]
        if self.use_global: c.append(zg[:, None].expand(-1, T, -1))
        h, _ = self.gru(torch.cat(c, -1)); o = self.out(h)
        logits, a, nz = o[..., :NH], o[..., NH:NH + 1], o[..., NH + 1:]
        amp = F.softplus(a) * 0.3
        hz = harmonic_synth(f0.clamp(40, 1200)[..., None], amp, logits, n)
        nse = filtered_noise(F.softplus(nz - 3) * 0.05, n)
        y = hz + nse; g = None
        if self.use_global: y, g = self.reverb(y, zg)
        return y, {"harm": hz, "noise": nse, "gains": g}


def loudness(x):                                                          # (B,N) -> (B,T) normalized dB at 100 Hz
    r = F.pad(x, (512, 512), mode="reflect").unfold(1, 1024, HOP).pow(2).mean(-1).add(1e-8).log10() * 10      # dB re 1
    return ((r + 60) / 30)[:, : x.shape[1] // HOP]


def mss_loss(y, x, sizes=(2048, 1024, 512, 256, 128)):
    l = 0
    for n in sizes:
        w = torch.hann_window(n, device=x.device)
        Y = torch.stft(y, n, n // 4, window=w, return_complex=True).abs(); X = torch.stft(x, n, n // 4, window=w, return_complex=True).abs()
        l = l + (X - Y).abs().mean() + (torch.log(X + 1e-4) - torch.log(Y + 1e-4)).abs().mean()
    return l / len(sizes)


def kl_vec(mu, lv, free=0.02):
    return (0.5 * (mu ** 2 + lv.exp() - 1 - lv)).mean(tuple(i for i in range(mu.dim()) if i != 1) if mu.dim() == 3 else 0).clamp_min(free)
