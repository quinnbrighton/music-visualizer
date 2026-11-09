# Cello from Scratch: a learned codec, an expressive performer and a composer for solo cello

An end-to-end system that **compresses, understands, performs and composes** solo cello music. It was built from raw recordings,
judged by ear, and checked against held-out data at every step.

```
 notes (composed or any MIDI)  ──►  expressive curves  ──►  continuous latent  ──►  waveform
   symbolic note model             note-conditioned         rectified-flow           regularized
   (key, meter, style)             curve model (flow)       latent model             adversarial codec
```

Each stage is a separate, inspectable model with a clear job. The audio codec is also usable on its own: it compresses real cello
recordings to a 25 Hz, 24-number-per-frame description, or about 1.8 kbit/s once quantized.

> **Scope and honesty.** Every number below comes from a results file and a held-out evaluation. The test recording (a Mischa Maisky
> performance) was never trained on. Listening
> overruled metrics when they disagreed.

---

## What it does

| Capability | How | Result (held-out) |
|---|---|---|
| **Compress cello audio** | Encoder → 24-dim latent at 25 Hz → harmonic-excitation decoder with an adversarial (GAN) stage | Spectral error 0.776 (multi-scale; the old spectrogram pipeline was far worse); pitch error 7 cents |
| **Quantize it into a real codec** | 12-stage residual vector quantizer (64 entries each) | 1,800 bit/s at 8 cents pitch error; 600 bit/s at 18 cents |
| **Make the latent predictable** | Regularized "sound-space" codec: equal latent step = equal change in sound, plus a temporal-predictability penalty | Unpredictable share **halved** (11.4% → 5.4%, frozen-predictor test) with *better* reconstruction (0.797 → 0.788) |
| **Render a performance from expressive curves** | Non-autoregressive rectified-flow transformer: pitch, loudness, brightness, noisiness and voicing in, latent out | 6.4 cents median pitch error on a 30 s held-out passage, 93% within 50 cents; frame-to-frame motion matches real playing |
| **Perform written notes like a cellist** | Note-conditioned curve model adds vibrato, slides and bow dynamics; explicit per-note vibrato drawn from real statistics | Follows a real cellist's performance of the same notes to 22 cents (79% within 50 c); voicing 99.5% |
| **Compose new music** | Meter-aware symbolic transformer pretrained on about 1 million notes of public-domain melodies, fine-tuned on the Bach cello suites | 95–100% in key; leap statistics matched to the suites; at most 16 notes copied from any single training passage |
| **Play any MIDI** | `midi_to_audio.py` runs the full performance chain | 93–94% of audio within 50 cents of the written notes |

## Listen

| What | File |
|---|---|
| Generated pieces: D minor 3/4, G major 4/4, C major minuet | `colab_results/symlm/pieces/*.mp4` |
| Best-scoring generated pieces (F major 3/8, B♭ major 4/4, G major 4/4, A minor 3/4) | `colab_results/symlm/batch/*.mp4` |
| A held-out Maisky passage performed from its notes alone, old vs new chain | `colab_results/midi/maisky_from_midi_*.wav` |
| The latent space moving with the music (old vs regularized codec, synced audio) | `colab_results/reg/maisky_latent_old_vs_reg_s4000.mp4` |

---

## The design, and why each decision was made

The project went through many design changes, and each one was driven by a measurement. These are the ones that mattered.

### 1. The codec: fix dynamics and excitation, not phase
- The original spectrogram autoencoder plus Griffin-Lim **kept only about 25% of the slow amplitude movement** (dynamics, vibrato) and
  added fast flutter. Phase reconstruction was not the main problem; lost dynamics were.
- A DDSP-style harmonic synthesizer halved the error (log-mel 0.87 → 0.44) and kept about 94% of the slow movement.
- I then moved to a **phase-locked harmonic excitation decoder**: the network predicts magnitude and phase offsets over a sine-bank
  excitation. This sounded clearly better by ear, even where the metrics barely moved.
- Three diagnosed perceptual problems, each with a targeted fix:
  - **Too clean between harmonics** (measured peak-to-valley contrast 15.9 dB vs 13.7 in the original): a learned noise path.
  - **"Doesn't sound like a cello"**: a texture loss on per-second spectral spread, band correlation and roughness.
  - **"Hollow"**: traced to note edges decaying too slowly (−2.4 vs −3.6 dB per 10 ms). Fixed with an adversarial decoder
    (multi-period and multi-band STFT discriminators, the HiFi-GAN / BigVGAN / DAC recipe): edges reached −3.3 dB per 10 ms.
- **Pitch lives inside the latent** instead of being sent as side information, so the bit rate is honest: everything the decoder
  needs is counted.

### 2. Tokens vs continuous latents: follow the evidence
- I built a full token pipeline: residual VQ codebooks (1024 and 64 entries; a k-means study showed 128 and 256 match 1024 per bit
  and overfit less), a MusicGen-style delay-pattern transformer, pitch-matching losses through the frozen codec, and scheduled
  sampling.
- A continuity diagnostic showed generated token sequences moved about 2× more per frame than real ones. The fine quantizer stages
  were close to random (5–5.8 of 6 bits).
- The literature (Kyutai's continuous audio language models, ViolinDiff, MIDI-DDSP) pointed to continuous latents with dense
  controls, so I switched to a **non-autoregressive rectified-flow model** over the continuous latent. Continuity matched real
  playing immediately.
- **Found and fixed a real bug along the way:** generation scripts truncated the controls at the prediction frontier, but the control
  convolution looks ahead, so the model saw zeros. The earlier "the transformer ignores pitch" conclusion was mostly this bug.

### 3. Regularizing the latent space: interpretability vs predictability
- **Measured** how much each latent channel moves the sound and the pitch: equal nudges changed the sound up to 2.3× more on some
  channels and pitch up to 9× more.
- First attempt: decorrelation plus unit variance per channel (the DIP-VAE / β-VAE idea) plus StyleGAN2-style path-length uniformity.
- **Lesson learned and written down:** per-channel unit variance *fights* the sound-space goal. It stretches quiet-but-important
  channels and amplifies noisy ones. Disentanglement and isometry are different objectives, and rate–distortion theory says
  predictability should be measured in the units of the perceptual metric.
- Final version: path-length sound uniformity, latent noise and a **causal temporal-predictability penalty**, all in raw latent
  units. Verified with a **frozen-predictor test** that both latents had to pass the same way: 11.4% → 5.4% unpredictable.
- An earlier, flattering version of this number was caught and retracted. It came from a jointly trained predictor still learning,
  not from the latent improving.

### 4. Performance: separate what is played from how it is played
- Generated pitch and loudness curves feed the latent model, so composition and rendering are decoupled.
- A note-guessing stage turns curves into notes (tuning estimate, vibrato-tolerant semitone snapping, bow re-attack detection,
  removal of pitch-tracker blips and octave errors). The curve model then learns only the *expression*.
- **Vibrato investigation:** six candidate fixes were measured, covering samplers, flow-time sampling, note-relative pitch, guidance
  and an adversarial loss. The models *do* produce realistic vibrato (about 6 Hz, about ±10 cents), but on 2–3% of long notes against
  12% in real playing. The model hedges on whether a note gets vibrato. The fix is explicit: decide per note with the real
  probability and add a real-shaped oscillation.
- **A decoder bug found by measurement:** the decoder's group norm takes statistics over the whole clip, but training only saw
  4-second clips, so long renders gained about 10 dB of hiss above 6.4 kHz. Decoding in 4-second windows with crossfades fixed it,
  and the codec now reproduces real recordings to within 0.0 dB in that band.

### 5. Composition: the data matters more than the model
- The first note model, trained on notes guessed from audio, "jumped around". Measurement showed why: 20% octave-plus leaps and 22%
  sub-80 ms blips were transcription artifacts, and there was no meter.
- I rebuilt it on **symbolic data with real meter**. Header tokens set style, time signature and key, and each note is encoded as bar
  line, position in bar, pitch and duration. Data: about 1 million notes from public-domain folk collections and Bach chorale voices,
  plus the Bach cello suites.
- **Overfitting controls:** transposition augmentation, held-out movements, early stopping. A copy audit (n-gram overlap) is
  calibrated against *real* held-out music, and a single-passage copy guard caps verbatim runs at 16 notes. Generation guards
  prevent repetition loops and runaway leaps.
- **Automatic curation:** candidates are ranked by surprisal under the pretrained model plus penalties for statistics far from the
  real suites (leaps, key, rhythmic monotony, range).

---

## Engineering and process

- **Rigor:** fixed held-out test and validation splits (near-duplicate recordings removed), level-matched listening comparisons, seed
  replicates (sampled decoding reproduces within 1%, so differences above about 2% are real), a stall rule that abandons a line after
  two non-improving experiments.
- **Compute:** everything trains on a single RTX 3060 (12 GB); longer runs optionally on Colab, driven by resumable pipeline scripts.
- **Scale of the work:** about 20 training pipelines (codecs, quantizers, token LMs, flow models, curve models, note models), each
  with in-training evaluations tied to perceptual failure modes (pitch error in cents, note-edge decay, harmonic contrast, texture,
  movement per frame, vibrato statistics).

## Utility and future applications

- **Low-bit-rate instrument audio:** a 1.8 kbit/s cello codec with pitch-accurate decoding, for streaming, archives or embedded use.
- **MIDI-to-performance rendering:** turn any score, sketch or melody into an expressive cello performance, as a composer's tool or a
  practice aid.
- **Interactive composition:** set key, meter and length, or give an opening phrase and let the model continue it. Curation keeps
  the best candidates.
- **Controllable expression:** independent knobs for vibrato, dynamics, brightness and bow noise, because each lives in its own
  curve.
- **Music information research:** an interpretable latent (dedicated pitch channels, measured per-channel sensitivity) for
  studying what makes a performance expressive.
- **Next steps:**
  - learned dynamics and phrasing from performances
  - phrase and section structure (A–B–A form)
  - per-note vibrato and dynamics predicted by the note model
  - quantizing the predictable latent with a predictive coder to turn halved surprise into real bits
  - extending to violin and viola, which use the same pipeline with new recordings

---

## Repository map

| Path | Contents |
|---|---|
| `v2/` | Codec research: harmonic and DDSP decoders, encoders, evaluation tools |
| `colab/cello_gan/train_gan.py`, `train_rvq.py`, `train_reg.py` | Adversarial codec, residual-VQ codec, regularized sound-space codec |
| `colab/cello_gan/train_lat.py`, `gen_lat.py` | Rectified-flow latent model (curves → latent) |
| `colab/cello_gan/guess_notes.py`, `train_curve.py` | Note guessing; note-conditioned expressive curve model |
| `colab/cello_gan/sym_tokens.py`, `train_symlm.py`, `gen_sym.py`, `score_sym.py` | Symbolic composer: tokens, training, generation with guards, automatic ranking |
| `colab/cello_gan/midi_to_audio.py` | Any MIDI file → cello audio (full chain, explicit vibrato, windowed decoding) |
| `colab/cello_gan/latent_sensitivity.py`, `latent_video.py`, `freq_diag.py` | Interpretability and diagnostics |
| `README_v1_vae.md` | The project's first version (spectrogram VAE + Griffin-Lim) |

## Quick start

```bash
cd colab/cello_gan
# compose: style, key, meter, length, tempo
STYLE=cello KEY="D minor" METER=3/4 BARS=32 BPM=66 SEED=3 \
  python gen_sym.py ../../colab_results/symlm/symlm_default.pt piece.mid
# perform: notes -> curves -> latent -> audio  (VIB = share of long notes given vibrato)
VIB=0.3 python midi_to_audio.py <codec.pt> <latent_model.pt> <curve_model.pt> piece.mid piece.wav
```
The codec scripts also read a few environment variables for the model configuration; the full set is in each script's header.

## Data and credits

- **Training audio:** 89 solo-cello performances from YouTube (about 12 hours), used for personal, non-commercial research. One Maisky
  recording is held out as the test piece. The audio is **not redistributed**. Every recording is credited with its performer or
  channel and a link to the original upload in [`docs/SOURCES.md`](SOURCES.md).
- **Symbolic data:**
  - public-domain collections bundled with [music21](https://web.mit.edu/music21/) (Essen folksongs, O'Neill's, Ryan's Mammoth,
    Aird's Airs, Bach chorales), used under their bundled licenses
  - Bach cello suites from the [Mutopia Project](https://www.mutopiaproject.org/): suites 1, 3 and 4 public domain, suites 2, 5 and 6
    CC BY-SA 3.0
- **Prior work this builds on:** DDSP (Engel et al. 2020), RAVE (Caillon & Esling 2021), HiFi-GAN / BigVGAN / DAC discriminators,
  Vocos, MusicGen (Copet et al. 2023), Kyutai's continuous audio language models (2025), MIDI-DDSP, ViolinDiff, StyleGAN2 path-length
  regularization, DIP-VAE, isometric autoencoders. The frozen MERT encoder was tested and found *worse* than a log-mel encoder for
  reconstruction.
