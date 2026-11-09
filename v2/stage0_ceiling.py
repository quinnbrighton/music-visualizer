"""Stage 0: what each vocoder path costs when given the TRUE spectrogram (the ceiling any autoencoder inherits)."""
import json, numpy as np, torch, torchaudio, librosa, soundfile as sf
from vocos import Vocos
from metrics import logmel_dist
B = "/home/qbrighto/music-visualizer"
x, sr = sf.read(f"{B}/oos_test/Bach - Cello Suite No. 2 [BWV 1008] - Mischa Maisky.wav", dtype="float32", always_2d=True)
x = x.mean(1)[sr * 60: sr * 180]                      # 2 min excerpt
ref = torchaudio.functional.resample(torch.from_numpy(x), sr, 24000)
res = {}
# (a) Vocos on true Vocos-mel
voc = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
with torch.no_grad():
    m = voc.feature_extractor(ref[None].cuda())
    y = voc.decode(m)[0].cpu()
res["vocos_true_mel"] = logmel_dist(y, ref)
torchaudio.save(f"{B}/v2/ceiling_vocos.wav", y[None], 24000)
# (b) current path: 512-bin mel of |STFT| (4096/512 @44.1k) + pinv + Griffin-Lim
y44 = librosa.resample(x, orig_sr=sr, target_sr=44100)
S = np.abs(librosa.stft(y44, n_fft=4096, hop_length=512))
mb = librosa.filters.mel(sr=44100, n_fft=4096, n_mels=512)
M = mb @ S
S_hat = np.maximum(np.linalg.pinv(mb) @ M, 0)
g = librosa.griffinlim(S_hat, n_iter=32, hop_length=512, n_fft=4096)
g24 = torchaudio.functional.resample(torch.from_numpy(g.astype(np.float32)), 44100, 24000)
res["griffinlim_true_mel512"] = logmel_dist(g24, ref)
torchaudio.save(f"{B}/v2/ceiling_griffinlim.wav", g24[None], 24000)
torchaudio.save(f"{B}/v2/ceiling_reference.wav", ref[None], 24000)
print(json.dumps(res, indent=1)); json.dump(res, open(f"{B}/v2/stage0_ceiling.json", "w"), indent=1)
