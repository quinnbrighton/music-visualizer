"""The cello codec (inference only): log-mel -> layer-fusion encoder -> 24-channel multi-timescale code averaged to 25 Hz; decoder = up-projection ->
CREPE-style pitch head -> phase-coherent harmonic excitation -> Vocos ConvNeXt backbone -> STFT magnitude + phase offset + shaped noise -> iSTFT.
Same module names and shapes as colab/cello_gan (train_harm.Harm + train_gan.HarmBN, built with MEL=1 ATTN=0 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5),
so the trained state_dict loads with strict=True. The decoder's GroupNorm takes statistics over the whole input, so always decode in 4 s windows (decode_ola)."""
import math
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from .vocos_backbone import VocosBackbone

SR, HOP, DSR, ZD, D = 24000, 240, 4, 24, 256                                # 24 kHz, 100 Hz frames, code at 100/DSR = 25 Hz
NF, NMEL, K = 1024, 128, 64                                                 # STFT size, mel bands, excitation harmonics
FMIN, CENT, NB = 60.0, 25.0, 209                                            # pitch bins: 25 cents from 60 Hz
RATES = [(6, 1), (5, 5), (5, 25), (4, 1), (4, 5)]                           # (channels, box-filter width in 100 Hz frames) = RATESSPEC


def block(cin, cout, k=5, d=1):
    return nn.Sequential(nn.Conv1d(cin, cout, k, padding=d * (k // 2), dilation=d), nn.GroupNorm(8, cout), nn.GELU())


def mel_fb(n_freqs=NF // 2 + 1, n_mels=NMEL, sr=SR):                       # = torchaudio.functional.melscale_fbanks(htk, norm=None, f 0..sr/2)
    fr = torch.linspace(0, sr // 2, n_freqs); hz2m = lambda f: 2595.0 * math.log10(1.0 + f / 700.0)
    fp = 700.0 * (10 ** (torch.linspace(hz2m(0.0), hz2m(sr / 2), n_mels + 2) / 2595.0) - 1.0); fd = fp[1:] - fp[:-1]; sl = fp[None] - fr[:, None]
    return torch.clamp(torch.min(-sl[:, :-2] / fd[:-1], sl[:, 2:] / fd[1:]), min=0)          # (n_freqs, n_mels)


class LayerFusionEncoder(nn.Module):                                        # enc_layers.LayerFusionEncoder with attn=0 (conv only)
    def __init__(self, n_layers=1, din=NMEL, d=D):
        super().__init__(); self.L, self.d = n_layers, d; self.attn = nn.ModuleList()
        self.norm = nn.ModuleList([nn.BatchNorm1d(din) for _ in range(n_layers)]); self.adapt = nn.ModuleList([nn.Linear(din, d) for _ in range(n_layers)])
        self.gate = nn.Linear(n_layers * d, n_layers); self.cat = nn.Linear(n_layers * d, d); self.out_norm = nn.GroupNorm(8, d)
        self.trunk = nn.Sequential(block(d, d, 5, 1), block(d, d, 5, 2), block(d, d, 5, 4))

    def forward(self, f75, T):                                              # (B,T',L,din) -> (B,d,T) at 100 Hz
        h = torch.stack([self.adapt[i](self.norm[i](f75[:, :, i].float().transpose(1, 2)).transpose(1, 2)) for i in range(self.L)], 2)
        g = torch.softmax(self.gate(h.flatten(2)), -1); x = F.gelu((g[..., None] * h).sum(2) + self.cat(h.flatten(2)))
        x = F.interpolate(x.transpose(1, 2), size=T, mode="linear", align_corners=False); return self.trunk(self.out_norm(x)), g


@torch.no_grad()
def excitation(f0, n):                                                      # f0 (B,T) Hz at 100 Hz -> (B,n) phase-coherent harmonic stack (1/k amplitudes) + a little noise
    f = F.interpolate(f0[:, None].clamp(40, 1200), size=n, mode="linear", align_corners=False)[:, 0]
    phi = torch.remainder(torch.cumsum(2 * math.pi * f.double() / SR, 1), 2 * math.pi).float()
    k = torch.arange(1, K + 1, device=f0.device).float(); ok = (f[..., None] * k < SR / 2 * 0.95).float()
    return (torch.sin(phi[..., None] * k) * ok / k).sum(-1) + 0.02 * torch.randn_like(f)


def pdecode(lg):                                                            # local weighted mean (cents) around the argmax bin -> Hz
    ce = torch.arange(NB, device=lg.device).float() * CENT; p = torch.softmax(lg, 1); i = p.argmax(1, keepdim=True); m = (torch.arange(NB, device=lg.device)[None, :, None] - i).abs() <= 2
    c = (p * m * ce[None, :, None]).sum(1) / (p * m).sum(1).clamp_min(1e-6); return FMIN * 2 ** (c / 1200)


class Codec(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = LayerFusionEncoder(1, NMEL, D)
        self.exp = nn.Sequential(nn.Conv1d(NF // 2 + 1, 128, 1), nn.GroupNorm(8, 128), nn.GELU())
        self.backbone = VocosBackbone(D + 128, 512, 1536, 8); self.out = nn.Linear(512, NF + 2)                      # [log-magnitude (513) | phase offset (513)]
        self.down = nn.Conv1d(D, 2 * ZD, 1); self.up = nn.Sequential(nn.Conv1d(ZD, D, 3, padding=1), nn.GELU(), nn.Conv1d(D, D, 1))
        self.ph = nn.Sequential(nn.Conv1d(D, 256, 5, padding=2), nn.GELU(), nn.Conv1d(256, NB + 1, 1)); self.nz = nn.Linear(512, NF // 2 + 1)
        self.pr1 = nn.Conv1d(ZD, 128, 5); self.pr2 = nn.Conv1d(128, 128, 5, dilation=2); self.pr3 = nn.Conv1d(128, 2 * ZD, 1)          # training-time temporal prior (unused here; kept for strict loading)
        self.register_buffer("win", torch.hann_window(NF), persistent=False); self.register_buffer("fb", mel_fb(), persistent=False)

    def lmel(self, x):                                                      # (B,N) -> (B,T,1,128), = train_harm.lmel (torchaudio MelSpectrogram, power 1)
        S = torch.stft(x, NF, HOP, window=self.win, center=True, pad_mode="reflect", return_complex=True).abs()
        M = (S.transpose(1, 2) @ self.fb).transpose(1, 2); return ((torch.log(M + 1e-5) + 5) / 3)[..., : x.shape[1] // HOP].transpose(1, 2)[:, :, None]

    def smooth(self, z):
        out, i = [], 0
        for n, w in RATES:
            g = z[:, i:i + n]; i += n
            out.append(g if w == 1 else F.avg_pool1d(F.pad(g, (w // 2, w // 2), mode="replicate"), w, stride=1))
        return torch.cat(out, 1)

    def to25(self, x):                                                      # audio (B,N) -> code (B,24,N//960), = train_rvq.to25(feats(x), N//HOP)
        T = x.shape[1] // HOP; h, _ = self.enc(self.lmel(x), T); mu, _lv = self.down(h).chunk(2, 1); return F.avg_pool1d(self.smooth(mu), DSR, DSR)

    def dec_feats(self, z, T): return self.up(F.interpolate(z, size=T, mode="linear", align_corners=False))
    def pitch_f0(self, hh): return pdecode(self.ph(hh)[:, :NB])

    def synth(self, h, f0, T):
        n = T * HOP; win = self.win; E = torch.stft(excitation(f0, n), NF, HOP, window=win, return_complex=True)[:, :, :T]; e = self.exp(torch.log(E.abs() + 1e-3))
        bb = self.backbone(torch.cat([h, e], 1)); o = self.out(bb).transpose(1, 2); zn = self.nz(bb).transpose(1, 2).clamp(max=2.0)
        mag, dph = torch.exp(o[:, : NF // 2 + 1]).clamp(max=1e2), math.pi * torch.tanh(o[:, NF // 2 + 1:])
        S = mag * torch.exp(1j * (torch.angle(E) + dph))
        env = F.avg_pool1d(mag.transpose(1, 2), 41, 1, 20, count_include_pad=False).transpose(1, 2)              # local spectral level (~500 Hz)
        with torch.no_grad(): Nn = torch.stft(torch.randn(h.shape[0], n, device=h.device), NF, HOP, window=win, return_complex=True)[:, :, :T]; Nn = Nn / Nn.abs().pow(2).mean().sqrt()
        S = S + torch.exp(zn) * env * Nn; S = torch.cat([S, S[:, :, -1:]], 2)
        return torch.istft(S, NF, HOP, window=win, length=n)

    def decode_window(self, z):                                             # z (B,24,t) at 25 Hz -> audio (B, t*960); one window, no overlap-add
        T = z.shape[2] * DSR; hh = self.dec_feats(z, T); return self.synth(hh, self.pitch_f0(hh), T)

    @torch.no_grad()
    def encode_long(self, x, chunk=20 * SR, ctx=SR):                        # (N,) tensor -> (N//960, 24): 20 s chunks with 1 s context each side, as encode_latents.py
        S = HOP * DSR; N = len(x) // S * S; parts = []
        for c0 in range(0, N, chunk):
            s, e = max(0, c0 - ctx), min(N, c0 + chunk + ctx); z = self.to25(x[None, s:e]); lo = (c0 - s) // S
            parts.append(z[0, :, lo: lo + min(chunk, N - c0) // S].t())
        return torch.cat(parts) if parts else x.new_zeros(0, ZD)

    @torch.no_grad()
    def decode_ola(self, z, W=100, Hh=75, seed=0):                          # z (n,24) at 25 Hz -> numpy audio: 4 s windows, 1 s linear crossfades (as make_assets.dec_ola)
        n_ = z.shape[0]; S = HOP * DSR; out = np.zeros(n_ * S); ws = np.zeros(n_ * S); st = list(range(0, max(1, n_ - W) + 1, Hh))
        if st[-1] + W < n_: st.append(n_ - W)
        for s in st:
            e = min(n_, s + W)
            if seed is not None: torch.manual_seed(seed)                     # decoder noise and excitation noise are random; seeded per window
            y = self.decode_window(z[s:e].t()[None])[0].cpu().double().numpy(); w = np.ones(len(y)); r = (W - Hh) * S
            if s > 0: w[:r] = np.linspace(0, 1, r)
            if e < n_: w[-r:] = np.linspace(1, 0, r)
            out[s * S: s * S + len(y)] += y * w; ws[s * S: s * S + len(y)] += w
        return out / np.maximum(ws, 1e-6)
