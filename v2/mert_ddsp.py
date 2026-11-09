"""MERT-encoder DDSP autoencoder. Decoder sees ONLY the latent (no explicit f0 / loudness): every bit is in the rate.
Pitch and loudness are predicted from the latent; CREPE / RMS act as TEACHERS (auxiliary losses) during training only."""
import glob, torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModel
from ddsp_model import *

MERT_PATH = glob.glob("/home/qbrighto/.cache/huggingface/hub/models--m-a-p--MERT-v1-95M/snapshots/*")[0]


class MertDDSP(nn.Module):
    def __init__(self, zdim=24, gdim=8, ch=256, rates=RATES, finetune_last=0):
        super().__init__()
        self.zdim, self.gdim, self.rates = zdim, gdim, rates
        self.mert = AutoModel.from_pretrained(MERT_PATH, trust_remote_code=True).eval()
        for p in self.mert.parameters(): p.requires_grad = False
        self.lw = nn.Parameter(torch.zeros(13))                                     # learned mixing weights over the 13 hidden states
        self.proj = nn.Sequential(nn.Conv1d(768, ch, 1), nn.GroupNorm(8, ch), nn.GELU())
        self.trunk = nn.Sequential(block(ch, ch, 5, 1), block(ch, ch, 5, 2), block(ch, ch, 5, 4)); self.head = nn.Conv1d(ch, 2 * zdim, 1)
        self.gtrunk = nn.Sequential(block(ch, ch, 5, 1), block(ch, ch, 5, 2)); self.ghead = nn.Linear(ch, 2 * gdim)
        self.reverb = Reverb(gdim)
        self.gru = nn.GRU(zdim + gdim, 256, batch_first=True)
        self.out = nn.Sequential(nn.Linear(256, 256), nn.GELU(), nn.Linear(256, NH + 1 + NB + 2))      # harmonic dist, amp, noise bands, f0, loudness
        self.smooth = lambda z: DDSPAE.smooth(self, z)

    def train(self, mode=True):
        super().train(mode); self.mert.eval(); return self                             # MERT always in eval mode (no dropout)

    def feats(self, x, T):
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)             # MERT's preprocessor normalisation
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            hs = torch.stack(self.mert(x, output_hidden_states=True).hidden_states, 0).float()     # (13,B,T75,768)
        h = (torch.softmax(self.lw, 0)[:, None, None, None] * hs).sum(0).transpose(1, 2)           # (B,768,T75)
        return self.proj(F.interpolate(h, size=T, mode="linear", align_corners=False))              # 75 Hz -> 100 Hz

    def encode(self, x):
        h = self.trunk(self.feats(x, x.shape[1] // HOP)); mu, lv = self.head(h).chunk(2, 1)
        return self.smooth(mu), self.smooth(lv).clamp(-8, 4)

    def encode_global(self, x):
        h = self.gtrunk(self.feats(x, x.shape[1] // HOP)); mu, lv = self.ghead(h.mean(2)).chunk(2, -1)
        return mu, lv.clamp(-8, 4)

    def decode(self, z, zg, n):
        T = z.shape[2]; h, _ = self.gru(torch.cat([z.transpose(1, 2), zg[:, None].expand(-1, T, -1)], -1)); o = self.out(h)
        logits, a, nz, fl = o[..., :NH], o[..., NH:NH + 1], o[..., NH + 1:NH + 1 + NB], o[..., -2:]
        f0 = 110 * 2 ** fl[..., 0].clamp(-2, 3.2)                                       # 27 Hz .. 1010 Hz
        hz = harmonic_synth(f0[..., None], F.softplus(a) * 0.3, logits, n)
        nse = filtered_noise(F.softplus(nz - 3) * 0.05, n)
        y, g = self.reverb(hz + nse, zg)
        return y, {"f0": f0, "loud": fl[..., 1], "gains": g, "harm": hz, "noise": nse}
