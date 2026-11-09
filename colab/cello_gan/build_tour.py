"""Write tour.ipynb (repo root): a plain-language, listen-and-look walkthrough of the project, using docs/assets/. Then execute it with nbconvert."""
import nbformat as nbf, os
nb = nbf.v4.new_notebook(); C = nb.cells
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip())); code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Tour: listening to and looking inside the cello models

This notebook lets you **listen** to the system and **see** what it is doing. No machine-learning background needed.
Run the cells from top to bottom (in Jupyter or Google Colab, from a copy of this repository) and press play on each audio player.

The idea in one sentence: the system squeezes real cello recordings into a compact code, learns how notes turn into expressive
playing, and can then write new music and perform it itself.
""")
code("""
import numpy as np, soundfile as sf, matplotlib.pyplot as plt
from IPython.display import Audio, Image, display, Markdown
A = "docs/assets"                      # figures and short audio clips live here
def listen(name, label):
    display(Markdown(f"**{label}**")); y, sr = sf.read(f"{A}/audio/{name}.wav"); display(Audio(y, rate=sr))
def waveform(name, label, seconds=8, color="#2a78d6"):
    y, sr = sf.read(f"{A}/audio/{name}.wav"); y = y[: seconds * sr]; t = np.arange(len(y)) / sr
    plt.figure(figsize=(11, 1.6)); plt.plot(t, y, lw=0.4, color=color); plt.yticks([]); plt.xlabel("seconds"); plt.title(label, loc="left"); plt.gca().spines[["top", "right", "left"]].set_visible(False); plt.show()
""")

md("""
## 1. Why sound is hard to compress, and the trick that solves it

Sound is dense. To capture everything up to 12 kHz, a recording needs **24,000 numbers (samples) every second**, and the ear notices
small mistakes: a bow attack is over in about 20 milliseconds, and trained listeners hear pitch errors of 5 to 10 *cents* (a cent is
1/100 of a semitone). Three minutes of cello is about 4.3 million numbers, roughly a 4-megapixel photo.

Most of those numbers are predictable once you know two things: **that it is a cello** (how a bowed string and a wooden body sound,
which never changes) and **the context** (which note, how loud, how bright, where the bow changes, which changes slowly).

That is the idea behind an **encoder-decoder** (also called an autoencoder): two networks trained together.
- The **encoder** squeezes the sound into a small code, the **latent** (a short list of numbers per moment).
- The **decoder** rebuilds the sound from that code alone.

Because the code is tiny, the encoder must keep only the context, and the decoder learns to supply the cello itself. In the picture,
one real second of 24,000 samples becomes 600 numbers and back again; stored compactly, that is **213 times smaller** than raw audio.
""")
code('display(Image(f"{A}/compression.png"))')

md("""
## 2. Compressing a real recording

Below is 12 seconds of a famous recording that the system **never heard during training** (Mischa Maisky playing Bach), and the
same passage **rebuilt from the system's compact code**. The code describes the sound with just 24 numbers, 25 times per second.
Compare by ear first.
""")
code("""
listen("1_original_maisky", "Original recording")
listen("2_codec_reconstruction", "Rebuilt from the compact code")
""")
md("""
**Reading the picture:** the top row is the *waveform* (how the air pressure moves over time; taller means louder). The bottom row is a
*spectrogram*: time runs left to right, pitch runs bottom to top, and brightness shows how loud each frequency is. The stacked
horizontal lines are the cello's harmonics. If the two sides look alike, the code has kept what matters.
""")
code('display(Image(f"{A}/codec_compare.png"))')
code("""
waveform("1_original_maisky", "Original: first 8 seconds", color="#52514e")
waveform("2_codec_reconstruction", "Rebuilt: first 8 seconds")
""")

md("""
**Why the decoder is trained as a GAN.** The compact code keeps the *shape* of the sound (pitch, loudness, tone colour) but not its
fine grain: the bow's texture and the crisp start and end of each note. A decoder trained only to match the original gets its best score
by guessing the *average* of all possible grains, which sounds smooth, blurred and hollow. So we added a second network, a
*discriminator*, that tries to tell real cello from decoded cello, and rewarded the decoder for fooling it. This pairing is called a
**GAN** (generative adversarial network). The decoder no longer copies the original sample by sample; it produces grain a real cello
*could* have made.

**Reading the picture:** top, the loudness over time (both versions follow the original almost equally well). Middle, 30 milliseconds
of waveform: before GAN training the cycles are smoothed toward plain sine waves; after, the jagged detail is back. Bottom, the
spectrogram: before, a regular combed pattern and smeared note edges; after, texture and note endings that look like the original.
""")
md("![Before and after GAN training](docs/assets/gan_compare.png)")
code("""
listen("9_before_gan", "Before adversarial (GAN) training")
listen("10_after_gan", "After adversarial (GAN) training")
""")
md("""
**How small can the code get?** The code is stored as a stack of up to 12 lookup steps (a *quantizer*: each step picks the closest
entry from a small table, refining what the previous steps missed). More steps cost more bits but sound better. The first few steps
carry most of the information: one step alone leaves the pitch 177 cents off, two bring it to 27. At **1.8 kilobits per second**
(about 1/70 of a typical MP3) pitch is at the edge of what trained ears can notice.
""")
code('display(Image(f"{A}/rate_distortion.png"))')
md("""
**The codec's journey.** Each version of the codec, scored on the same 15 seconds of the held-out recording (lower is better).
Switching to a decoder built from a bank of tuned oscillators halved the error. The later steps barely move this score but changed
the sound a lot: this score measures the average spectrum, so it cannot see what GAN training adds (the figure above shows that).
""")
code('display(Image(f"{A}/journey.png"))')

md("""
## 3. What the compact code looks like (the "embedding")

Every 40 milliseconds the system writes down 24 numbers that describe the sound at that moment. That list of numbers is a point in a
24-dimensional space, and as the music plays the point moves, tracing a path.

- **Left:** the path squashed down to 2 dimensions so we can see it. Each dot is a moment, coloured by the note being played.
  Similar sounds land near each other: low notes in one region, high notes in another.
- **Right:** all 24 numbers over time (red = high, blue = low). Some channels change smoothly with the phrasing; others flicker with
  the bow.
""")
md("""
**Making the code easy to predict.** We retrained the codec with extra rules so that each moment is easier to guess from the
previous 1.8 seconds, which makes it easier for generators to write. The picture shows a predictor trying to guess the next moment of
the held-out passage: the old code spikes unpredictably throughout, the new one stays low (9% unexplained against 21%).
""")
code('display(Image(f"{A}/surprise.png"))')
md("""
**What each of the 24 numbers does.** Every channel pushed by the same amount, and what changes in the sound. Pitch lives mostly in
channels 8 to 10, channel 6 adds bow noise and air, and most others mix loudness and tone colour.
""")
md("![What each channel does](docs/assets/channel_knobs.png)")
code('display(Image(f"{A}/embedding.png"))')
md("""
**Which numbers matter?** We nudged each of the 24 numbers by the same amount and measured how much the sound and the pitch changed.
After regularization, every channel affects the *sound* by a similar amount (good for building generators on top), and *pitch* lives in
a few dedicated channels, which makes the code easier to interpret.
""")
md("""
**Watching new music move through the space (UMAP).** UMAP flattens the 24 numbers onto a 2D map while keeping similar moments close
together. We fit the map on about 50,000 moments from the training recordings (blue, darker = higher pitch), then placed the held-out
Maisky passage on it moment by moment (orange). Play the clip and watch the dot jump between neighbourhoods as the notes change.
""")
code('display(Image(f"{A}/umap_path.gif"))')
code('display(Image(f"{A}/umap_path.png"))')
code("""
from IPython.display import Video
display(Video(f"{A}/audio/8_umap_path.mp4", embed=False, width=520))   # the same animation with the original audio
""")
md("""
**Where each note lives.** The same map with every moment coloured by its note: **hue (colour) = note name, brightness = octave**.
Large dots mark the middle of each semitone's region. The code was never told note names, yet each note forms its own islands, and
following the notes upward snakes across the whole map. Two of the biggest jumps land on open strings (D3 and A3), which ring without
a finger on them and so sound different from their neighbours.
""")
md("![Every moment coloured by its note](docs/assets/umap_pitch_path.png)")
md("The same colouring on a 3D version of the map, spinning. The four open strings (C, G, D, A) have black outlines.")
md("![The sound space in 3D, spinning](docs/assets/umap3d_spin.gif)")
md("""
Neighbouring semitones in alternating black and white: every step up in pitch moves to the next stripe. Grey moments have no clear
pitch (bow noise, attacks, silence between notes) and gather in regions of their own.
""")
md("![Neighbouring semitones in black and white](docs/assets/umap_semitones_bw.png)")
code('display(Image(f"{A}/channel_sensitivity.png"))')

md("""
## 4. Peeling the layers, then editing music with them

If you remove the effect of one thing at a time from the 24 numbers (subtracting each group's average), the variation peels apart
in a clear order:

| Removed | Variation left | What organizes the rest |
|---|---|---|
| nothing | 100% | the note: 66% of each moment's ten nearest neighbours share its semitone (3% by chance) |
| pitch | 57% | **loudness**, a smooth gradient across the whole space |
| + loudness | 49% | **the recording** (instrument, microphone, room): 37% of neighbours share one (1% by chance) |
| + recording | 39% | nothing measurable; fine detail such as bow grain |

With pitch subtracted, loudness becomes the main axis of the map (right), and moments from the same recording stay together (middle).
""")
md("![The sound space with pitch subtracted](docs/assets/umap_minus_pitch.png)")
md("""
**Editing by moving through the code.** If loudness is a *direction* in the 24-number space, we can push a performance along it and
decode the result. It does not act like a volume knob: the decoder rebuilds the sound as if the cellist had played harder. Listen to
the Maisky reconstruction followed by the pushed version, played back at the same level.
""")
code("""
listen("11_loudness_push", "Reconstruction, then the loudness push (same playback level)")
display(Image(f"{A}/loudness_push_stats.png"))
""")
md("""
Every one of 89 recordings was pushed the same way, then turned back down by the level it gained, so only *how* the sound is produced
differs:

| After removing the level change | Median across 89 recordings | Recordings changing this way |
|---|---|---|
| upper harmonics (1–4 kHz) vs fundamentals | **+1.3 dB** | 83 / 89 |
| level peaks above average (crest) | **−1.2 dB**: more sustained, pressed tone | 79 / 89 |
| brightness (spectral centroid) | **+9.0%** | 81 / 89 |
| clarity between harmonics | **−0.6 dB**: more bow noise, scratchier | 69 / 89 |
| attack sharpness | no change | n/a |

That is what heavier bow pressure does to a real cello, and the codec learned it from the recordings alone.
""")

md("""
## 5. From written notes to a living performance

Sheet music only says *which* note and *when*. A cellist adds slides between notes, vibrato and swells in loudness. Here the system was
given only the notes of the Maisky passage (orange) and decided how to play them (blue). It never saw this recording.
""")
code('display(Image(f"{A}/notes_to_performance.png"))')
code("""
listen("3_performed_from_notes_maisky", "The system performing the passage from its notes alone")
listen("1_original_maisky", "For comparison: Maisky's real performance")
""")

md("""
## 6. Music the system composed itself

A separate model learned melody, rhythm, key and meter from about a million notes of public-domain melodies (folk airs, Bach chorales)
and the Bach cello suites. You choose a key, a time signature and a style; it writes the notes, and the rest of the system performs them
on its learned cello. Nothing below exists in any training data.
""")
code('display(Image(f"{A}/generated_piece.png"))')
code("""
listen("4_composed_air_dminor", "An air in D minor (3/4)")
listen("5_composed_air_gmajor", "An air in G major (4/4)")
listen("6_composed_chorale_dminor", "A chorale-style melody in D minor (4/4)")
listen("7_composed_suite_style_gmajor", "A busier piece in the style of the Bach cello suites (G major)")
""")

md("""
## 7. How good is it? (the short version)

| Test | Result |
|---|---|
| Pitch accuracy when rebuilding the held-out recording | within about 7 cents (a cent is 1/100 of a semitone) |
| Bit rate of the quantized codec | 1,800 bits per second (a phone call is about 8,000) |
| Pitch accuracy when performing notes from their score | 93–94% of the time within a quarter of a semitone |
| Copying in composed music | never more than 16 notes in a row from any single training melody |

Full details, design decisions and honest limitations are in the [README](README.md) and the lab notebook (`v2/LAB_NOTEBOOK.md`).
""")
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
out = os.path.expanduser("~/music-visualizer/tour.ipynb"); nbf.write(nb, out); print("wrote", out)
