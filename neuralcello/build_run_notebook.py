"""Write run.ipynb (repo root): a short notebook that runs every model from the exported weights in models/ - compose, perform,
reconstruct a recording, and edit it through the latent. usage: python neuralcello/build_run_notebook.py"""
import os, nbformat as nbf
nb = nbf.v4.new_notebook(); C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip())); code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
md("""
# Run Neural Cello yourself

Everything below runs from the weights in `models/` (about 125 MB, already in the repo). A GPU makes it fast; a CPU works too
(about 30 s to perform a minute of music).

1. **Compose** a new melody in a chosen key, meter and style.
2. **Perform** it: notes → expressive pitch and loudness curves → the codec's latent → audio.
3. **Reconstruct** a real recording through the 24-number-per-frame code.
4. **Edit** the recording by pushing along the latent's loudness direction.
""")
code("""
# In Colab or a fresh environment, uncomment:
# !git clone https://github.com/quinnbrighton/neural-cello && %cd neural-cello
# !pip install -q -r requirements.txt
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from IPython.display import Audio, display
import neuralcello as nc
m = nc.load()                                      # CUDA if available, else CPU; models load on first use
SR = nc.SR                                         # 24 kHz
""")
md("## 1. Compose\nStyles: `folk`, `chorale`, `cello` (Bach-suite style). `tau` is the target surprise per token: lower is more predictable.")
code("""
midi = m.compose(style="folk", key="D minor", meter="3/4", bars=16, bpm=60, tau=0.6, seed=0)
notes = midi.instruments[0].notes
plt.figure(figsize=(11, 2.4))
for n in notes: plt.plot([n.start, n.end], [n.pitch, n.pitch], lw=4, color="#2a78d6", solid_capstyle="butt")
plt.xlabel("seconds"); plt.ylabel("MIDI pitch"); plt.title(f"{len(notes)} composed notes", loc="left"); plt.show()
""")
md("## 2. Perform\nThe notes go through the performance model (slides, vibrato, dynamics), the latent model and the decoder.")
code("""
audio = m.perform(midi, vib=0.6, seed=0)
sf.write("my_piece.wav", audio, SR)
display(Audio(audio, rate=SR))
""")
md("## 3. Reconstruct a recording\nThe held-out Maisky clip is encoded to 24 numbers every 40 ms and decoded again.")
code("""
x, sr = sf.read("docs/assets/audio/1_original_maisky.wav")
z = m.encode(x, sr)                                # (frames, 24) at 25 Hz
y = m.decode(z)
print(f"{len(x) / sr:.0f} s of audio -> latent of shape {z.shape} ({z.size} numbers instead of {len(x)})")
display(Audio(x, rate=sr), Audio(y, rate=SR))
plt.figure(figsize=(11, 2.4)); plt.imshow(((z - z.mean(0)) / z.std(0)).T, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, extent=[0, len(z) / 25, 23.5, -0.5])
plt.xlabel("seconds"); plt.ylabel("channel"); plt.title("The latent: 24 channels over time", loc="left"); plt.show()
""")
md("""
## 4. Edit through the latent
`models/loudness_direction.npy` is the direction in the latent along which loudness changes (fitted within recordings, after removing
pitch). Pushing along it and then turning the result back down to the same level leaves only the change in *how* the cello is played:
more upper harmonics, a more sustained, scratchier tone.
""")
code("""
d = np.load("models/loudness_direction.npy")
pushed = m.decode(z + 20 * d)
rms = lambda v: np.sqrt(np.mean(v ** 2))
print(f"the push raised the level by {20 * np.log10(rms(pushed) / rms(y)):.1f} dB; both clips below are played at the same level")
pushed = pushed / rms(pushed) * rms(y)
display(Audio(y, rate=SR), Audio(pushed, rate=SR))
""")
md("""
## Also from the command line
```bash
python -m neuralcello compose piece.wav --style folk --key "D minor" --meter 3/4 --bars 24   # writes piece.mid and piece.wav
python -m neuralcello perform piece.mid piece.wav
python -m neuralcello reconstruct input.wav output.wav
```
""")
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "run.ipynb"); nbf.write(nb, out); print("wrote", out)
