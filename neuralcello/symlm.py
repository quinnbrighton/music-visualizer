"""Symbolic composer (inference only): meter-aware note tokens (= colab/cello_gan/sym_tokens.py) and the causal transformer (= train_symlm.SymLM)."""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

PMIN, PMAX = 24, 96
METERS = [(2, 4), (3, 4), (4, 4), (2, 2), (3, 2), (3, 8), (6, 8), (9, 8), (12, 8), (6, 4), (5, 4), (0, 0)]   # last = other
STYLES = ["folk", "chorale", "cello"]
DUR_Q = np.array([1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24, 30, 36, 42, 48, 60, 72, 84, 96, 120, 144, 192]) / 12.0   # durations in quarters
NPOS = 96                                                                   # onset positions on a 1/12-quarter grid (bars up to 8 quarters)
OFF_POS = 0; OFF_P = NPOS; OFF_D = OFF_P + (PMAX - PMIN + 1); OFF_BAR = OFF_D + len(DUR_Q); OFF_K = OFF_BAR + 1; OFF_M = OFF_K + 24; OFF_ST = OFF_M + len(METERS)
PAD = OFF_ST + len(STYLES); BOS = PAD + 1
OFF_DEN = BOS + 1; OFF_STP = OFF_DEN + 4                                    # style-control tokens (only in checkpoints whose vocabulary has them)
NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def key_name(k): return NAMES[k % 12] + (" major" if k < 12 else " minor")
def meter_index(num, den): return METERS.index((num, den)) if (num, den) in METERS else len(METERS) - 1


def decode(toks, barlen=4.0):                                               # -> list of (onset q, pitch, dur q); bars advance by barlen quarters
    out, bar, cur = [], -1, {}
    for x in toks:
        if x == OFF_BAR: bar += 1
        elif OFF_POS <= x < OFF_P: cur = {"on": max(bar, 0) * barlen + (x - OFF_POS) / 12}
        elif OFF_P <= x < OFF_D and "on" in cur: cur["p"] = x - OFF_P + PMIN
        elif OFF_D <= x < OFF_BAR and "p" in cur: out.append((cur["on"], cur["p"], float(DUR_Q[x - OFF_D]))); cur = {}
    return out


class Block(nn.Module):
    def __init__(self, D, HEADS):
        super().__init__(); self.D, self.H = D, HEADS; self.n1, self.n2 = nn.LayerNorm(D), nn.LayerNorm(D); self.qkv = nn.Linear(D, 3 * D); self.proj = nn.Linear(D, D)
        self.ff = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D), nn.Dropout(0.0)); self.drop = nn.Dropout(0.0)

    def forward(self, x):
        B, T, _ = x.shape; q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.H, self.D // self.H).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, self.D)
        x = x + self.drop(self.proj(a)); return x + self.ff(self.n2(x))


class SymLM(nn.Module):
    def __init__(self, VOCAB=233, CTX=1024, D=384, LAYERS=8, HEADS=6):
        super().__init__(); self.ctx = CTX; self.emb = nn.Embedding(VOCAB, D); self.pos = nn.Parameter(torch.zeros(1, CTX, D)); self.drop = nn.Dropout(0.0)
        self.blocks = nn.ModuleList([Block(D, HEADS) for _ in range(LAYERS)]); self.nf = nn.LayerNorm(D); self.head = nn.Linear(D, VOCAB, bias=False); self.head.weight = self.emb.weight

    def forward(self, t):
        x = self.drop(self.emb(t) + self.pos[:, : t.shape[1]])
        for b in self.blocks: x = b(x)
        return self.head(self.nf(x))
