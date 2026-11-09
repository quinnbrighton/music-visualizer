"""Neural-vocoder decoder on top of cached MERT features.
encoder: cached MERT mix (768 @75 Hz) -> 100 Hz -> conv head -> multi-rate latent z(24) + global code(8)
decoder: pretrained Vocos ConvNeXt backbone (input layer replaced) + ISTFT head (hop 240) -> 24 kHz waveform
probes:  linear heads z -> (f0 in octaves, loudness): interpretability aids + weak supervision"""
import torch, torch.nn as nn, torch.nn.functional as F
from vocos import Vocos
from vocos.heads import ISTFTHead
from vocos.models import VocosBackbone
from ddsp_model import block, RATES, SR, HOP, kl_vec, mss_loss, loudness


class VocModel(nn.Module):
    def __init__(self, zdim=24, gdim=8, ch=256, rates=RATES, extra=0, mel_in=False):
        super().__init__()
        self.zdim, self.gdim, self.rates, self.extra, self.mel_in = zdim, gdim, rates, extra, mel_in
        self.lw = nn.Parameter(torch.zeros(4))                                  # learned mix over the 4 cached MERT layers
        self.proj = nn.Sequential(nn.Conv1d(128 if mel_in else 768, ch, 1), nn.GroupNorm(8, ch), nn.GELU())
        self.trunk = nn.Sequential(block(ch, ch, 5, 1), block(ch, ch, 5, 2), block(ch, ch, 5, 4)); self.head = nn.Conv1d(ch, 2 * zdim, 1)
        self.gtrunk = nn.Sequential(block(ch, ch, 5, 1), block(ch, ch, 5, 2)); self.ghead = nn.Linear(ch, 2 * gdim)
        pre = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        self.backbone = VocosBackbone(zdim + gdim + extra, 512, 1536, 8)
        miss = self.backbone.load_state_dict({k: v for k, v in pre.backbone.state_dict().items() if not k.startswith("embed.")}, strict=False)
        assert all(k.startswith("embed.") for k in miss.missing_keys) and not miss.unexpected_keys, miss
        self.voc_head = ISTFTHead(512, 1024, HOP, padding="same"); self.voc_head.out.load_state_dict(pre.head.out.state_dict())
        self.f0_probe, self.loud_probe = nn.Linear(zdim, 1), nn.Linear(zdim, 1)

    def smooth(self, z):
        out, i = [], 0
        for n, w in self.rates:
            g = z[:, i:i + n]; i += n
            out.append(g if w == 1 else F.avg_pool1d(F.pad(g, (w // 2, w // 2), mode="replicate"), w, stride=1))
        return torch.cat(out, 1)

    def feats(self, f75, T):                                                    # f75: (B,T75,4,768) cached layers, or (B,128,T) log-mel if mel_in
        if self.mel_in: return self.proj(f75)
        f75 = (torch.softmax(self.lw, 0)[None, None, :, None] * f75).sum(2)
        return self.proj(F.interpolate(f75.transpose(1, 2), size=T, mode="linear", align_corners=False))

    def encode(self, f75, T):
        mu, lv = self.head(self.trunk(self.feats(f75, T))).chunk(2, 1)
        return self.smooth(mu), self.smooth(lv).clamp(-8, 4)

    def encode_global(self, f75, T):
        mu, lv = self.ghead(self.gtrunk(self.feats(f75, T)).mean(2)).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def decode(self, z, zg, ex=None):                                           # ex: optional explicit controls (B,extra,T), e.g. log2 f0 and loudness
        T = z.shape[2]; x = torch.cat([z, zg[:, :, None].expand(-1, -1, T)] + ([ex] if self.extra else []), 1)
        y = self.voc_head(self.backbone(x))
        zt = z.transpose(1, 2)
        return y, {"f0": 110 * 2 ** self.f0_probe(zt)[..., 0].clamp(-2, 3.2), "loud": self.loud_probe(zt)[..., 0]}


class DiscR(nn.Module):
    def __init__(self, n_fft, hop, ch=32):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop; self.register_buffer("win", torch.hann_window(n_fft))
        c = lambda i, o, k, s: nn.Conv2d(i, o, k, stride=s, padding=(k[0] // 2, k[1] // 2))
        self.convs = nn.ModuleList([c(1, ch, (3, 9), (1, 1)), c(ch, ch, (3, 9), (1, 2)), c(ch, ch, (3, 9), (1, 2)), c(ch, ch, (3, 9), (1, 2)), c(ch, ch, (3, 3), (1, 1))])
        self.out = nn.Conv2d(ch, 1, (3, 3), padding=1)

    def forward(self, x):
        h = torch.log(torch.stft(x, self.n_fft, self.hop, window=self.win, return_complex=True).abs() + 1e-5)[:, None]; fm = []
        for c in self.convs: h = F.leaky_relu(c(h), 0.1); fm.append(h)
        return self.out(h), fm


class MRD(nn.Module):
    def __init__(self):
        super().__init__(); self.ds = nn.ModuleList([DiscR(2048, 512, 16), DiscR(1024, 256, 16), DiscR(512, 128, 16)])
    def forward(self, x): return [d(x) for d in self.ds]
