import json, numpy as np, torch, torchaudio
from vocos import Vocos
from transformers import ClapModel, ClapProcessor
B = "/home/qbrighto/music-visualizer/v2"; N = 24000 * 30
def rd(n): return torchaudio.load(f"{B}/{n}.wav")[0][0][:N]
ref, gl, st = rd("ceiling_reference"), rd("ceiling_griffinlim"), rd("ceiling_vocos")
v = Vocos.from_pretrained("charactr/vocos-mel-24khz").cuda().eval()
ck = torch.load(f"{B}/vocos_cello.pt"); v.backbone.load_state_dict(ck["backbone"]); v.head.load_state_dict(ck["head"])
with torch.no_grad(): ft = v.decode(v.feature_extractor(ref[None].cuda()))[0].cpu()[:N]
clips = {"1_original": ref, "2_griffinlim_512bin": gl, "3_vocos_stock": st, "4_vocos_cello_finetuned": ft}
for k, w in clips.items(): torchaudio.save(f"{B}/ab_{k}.wav", (w / w.abs().max() * 0.9)[None], 24000)
cm = ClapModel.from_pretrained("laion/clap-htsat-unfused").cuda().eval(); pr = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
def emb(w):
    a = torchaudio.functional.resample(w, 24000, 48000).numpy()
    with torch.no_grad():
        f = pr(audio=a, sampling_rate=48000, return_tensors="pt")["input_features"].cuda()
        e = cm.get_audio_features(input_features=f)
        e = e if torch.is_tensor(e) else e.pooler_output
    return torch.nn.functional.normalize(e, dim=-1)[0]
# CLAP on 10 s windows vs the original, averaged
res = {}
for k, w in clips.items():
    s = [float(emb(w[i:i + 240000]) @ emb(ref[i:i + 240000])) for i in range(0, N - 239999, 240000)]
    res[k] = float(np.mean(s))
print(json.dumps(res, indent=1)); json.dump(res, open(f"{B}/ab_clap.json", "w"), indent=1)
