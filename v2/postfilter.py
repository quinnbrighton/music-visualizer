"""Output-only refinement (no model change): fills the too-clean valleys between harmonics with spectrally shaped noise (breath/bow noise) + a gentle band-balance EQ.
Measured: the vocoder output has MORE harmonic peak/valley contrast than the real cello (15.9 vs 13.7 dB at 1.5-3 kHz), so the fix is adding noise, not a comb.
usage: postfilter.py IN.wav OUT.wav [FILL=0.3] [CLIP_START_SAMPLE=5572800] ; f0 from f0clean for the test recording (100 Hz frames)."""
import sys, os, numpy as np, torch, soundfile as sf, torchaudio
B = os.path.expanduser("~/music-visualizer/v2")
def postfilter(y, f0, sr=24000, fill=0.0, n_fft=2048, hop=240, f_lo=800.0):
    y = torch.from_numpy(y).float(); win = torch.hann_window(n_fft)
    S = torch.stft(y, n_fft, hop, window=win, return_complex=True); T = S.shape[1]
    f0 = torch.from_numpy(f0[:T]).float().clamp(40, 1200); f0 = torch.nn.functional.pad(f0, (0, T - len(f0)), value=float(f0[-1]))
    fr = torch.arange(S.shape[0]).float() * sr / n_fft                                   # (F,)
    ph = fr[:, None] / f0[None]                                                          # harmonic number (fractional)
    dist = (ph - ph.round()).abs() * f0[None]                                            # Hz to nearest harmonic
    width = torch.clamp(0.18 * f0[None], min=25.0)                                       # keep a harmonic's skirt
    valley = torch.clamp((dist - width) / width, 0, 1)                                   # 0 on harmonic, 1 mid-valley
    onset = torch.clamp((fr[:, None] - f_lo) / 1200.0, 0, 1)                             # only act where valleys are filled (>~1 kHz)
    env = torch.nn.functional.avg_pool1d(S.abs()[None], 41, 1, 20, count_include_pad=False)[0]                # local spectral envelope (~500 Hz)
    N = torch.stft(torch.randn(len(y)), n_fft, hop, window=win, return_complex=True)[:, :T]; N = N / N.abs().pow(2).mean().sqrt(); S = S + fill * N * env * (valley * onset)     # noise built in the time domain so it survives the inverse STFT
    g = torch.ones_like(valley)
    eq = torch.ones_like(fr); eq[fr > 5000] = 10 ** (1.0 / 20); eq[fr < 500] = 10 ** (-0.5 / 20)   # measured recon-orig band offsets
    return torch.istft(S * g * eq[:, None], n_fft, hop, window=win, length=len(y)).numpy()
if __name__ == "__main__":
    i, o = sys.argv[1], sys.argv[2]; d = float(sys.argv[3]) if len(sys.argv) > 3 else 0.3; s0 = int(sys.argv[4]) if len(sys.argv) > 4 else 5572800
    ck = torch.load(f"{B}/multirate_nested.pt", map_location="cpu"); k = ck["test"][0]; f0 = np.load(f"{B}/f0clean/{k}.npy")[s0 // 240:]
    y, sr = sf.read(i); y = y.astype(np.float32); z = postfilter(y, f0, sr, d); z *= np.sqrt((y ** 2).mean() / (z ** 2).mean()); sf.write(o, z, sr)
