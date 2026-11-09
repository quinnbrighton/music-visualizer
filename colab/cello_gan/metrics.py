import numpy as np, torch, torchaudio
_mels = {}
def logmel_dist(a, b, sr=24000):
    """Multi-res log-mel L1 (dB-ish, natural log) and spectral convergence between two waveforms (tensors, 24 kHz)."""
    n = min(len(a), len(b)); a, b = a[:n].cuda(), b[:n].cuda()
    out = {"lm": 0.0, "sc": 0.0}
    for nf in (512, 1024, 2048):
        t = torchaudio.transforms.MelSpectrogram(sr, nf, nf, nf // 4, n_mels=min(128, nf // 8), power=1.0).cuda()
        A, B = t(a), t(b)
        out["lm"] += (torch.log(A.clamp_min(1e-5)) - torch.log(B.clamp_min(1e-5))).abs().mean().item() / 3
        out["sc"] += (torch.norm(A - B) / torch.norm(B)).item() / 3
    return out
