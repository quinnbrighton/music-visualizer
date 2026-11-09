# Cello autoencoder overnight study: what condenses cello audio best?

Written ~07:20 on Nov 7 2026 from `LAB_NOTEBOOK.md` and the result files in `~/music-visualizer/v2/`. Nothing here was pushed anywhere. Numbers are from single training runs unless stated. A seed replicate of the best vocoder config (voc_g vs voc_g_seed1) reproduced the sampled-decoding result almost exactly (val mss .890 vs .890; Maisky .784 vs .787; log-mel .663 vs .658), so at sampled decoding differences above ~2% are probably real and under ~1% are ties. Mean-decode numbers are NOT stable across seeds (val mss 2.04 vs 1.82 for voc_g), so do not rely on them.

## Question
What is the smallest description of solo cello audio that still reconstructs it well, and what does each part of the description carry? Compared: a spectrogram autoencoder + Griffin-Lim (the old pipeline), a DDSP autoencoder, and a neural-vocoder autoencoder, with a frozen MERT-v1-95M encoder or a log-mel conv encoder.

## Setup
- Data: 93 recordings (12.8 h), mono 24 kHz. Test = one never-trained Mischa Maisky recording; validation = 5 recordings (2 near-duplicates of training files were removed). 83 training recordings.
- Latent: 24 dims in 5 groups smoothed over 0.01 / 0.05 / 0.25 / 1 / 3 s, plus an 8-dim per-clip global code, VAE with KL penalty. Rate = KL in bits per second of audio (100 frames/s).
- Metric used throughout: multi-resolution spectral loss ("mss", lower is better) on 20 s held-out windows. Also log-mel distance on a fixed 15 s Maisky window. Neither is a listening test.
- "Sampled" decoding means z = mean + noise * std (as in training); "mean" decoding uses the mean latent.

## Results (held-out validation mss; Maisky test mss in brackets)
| System | Encoder | Given to decoder | Latent b/s | + side info | Val | Test |
|---|---|---|---|---|---|---|
| Old: spectrogram AE + Griffin-Lim | log-mel | nothing | ~1400 | 0 | log-mel err .87* | |
| DDSP v1 | log-mel | f0 + loudness (yin) | 586 | ~330 | **0.826** | 0.783 |
| Vocoder, log-mel enc (voc_f) | log-mel | f0 + loudness | ~830 | ~330 | 0.883 | 0.775 |
| Vocoder, log-mel enc (voc_g) | log-mel | nothing | ~860 | 0 | 0.890 | 0.784 |
| Vocoder, MERT enc (voc_e) | MERT | f0 + loudness | ~775 | ~330 | 0.918 | 0.786 |
| Vocoder, MERT enc (voc_c) | MERT | nothing | 848 | 0 | 1.167 | 0.911 |
| Vocoder, MERT enc (voc_b) | MERT | nothing | 1945 | 0 | 1.170 | 0.892 |
| voc_g + adversarial stage | log-mel | nothing | ~775 | 0 | 1.008 | 0.896 |
\* different protocol; see `compare_systems.json`: on the same 15 s window log-mel error: Griffin-Lim on the true spectrogram .15 (uncompressed ceiling), old pipeline .87, DDSP v1 .44, voc_g .56.
Side-info bits are a rough entropy estimate (f0 in 0.5-semitone steps, loudness in 1 dB steps): ~160 + ~170 b/s.

## Findings that held up
1. **Fast flutter was not the problem; lost dynamics were.** The old spectrogram pipeline kept ~25% of slow amplitude movement and added fast flutter; DDSP kept ~94% of slow movement with no excess flutter (compare_systems.json).
2. **Pitch tracker errors are real.** Raw yin had 6.6% of frames jump >5 semitones (4.9% octave-sized); cleaning cut this to 2.2% / 1.0%; CREPE+Viterbi had 0 on the test window.
3. **Variational decoders need their noise.** Vocoder decoders trained on sampled latents are much worse when decoded from the mean (voc_b: 1.389 mean vs 1.170 sampled; voc_e: 3.40 vs 0.92). The noise supplies fast (10-43 Hz) texture, e.g. bow noise-like variation. DDSP v1 is insensitive (0.826 vs 0.829).
4. **Frozen MERT is worse than a log-mel conv encoder for reconstruction** at ~850 b/s: 1.167 vs 0.890 with nothing given (-24%), 0.918 vs 0.883 with f0+loudness given. Handing f0+loudness to a MERT model recovered some of that gap; it did nothing for the log-mel encoder (0.883 vs 0.890). (An earlier overnight statement that "explicit pitch+loudness is the most efficient condensation" was wrong and is retracted in the notebook.)
5. **Rate-distortion is shallow.** MERT vocoder: 4421 b/s mean-decode 1.316; 1945 b/s 1.389; 848 b/s 1.482 (mean decode).
6. **The adversarial stage trades spectral fidelity for liveliness.** It raises mss (0.890 -> 1.008) but restores slow/mid amplitude movement (slowest band .114 -> .374, target .665 on that window). Which sounds better is a listening question.
7. **A per-clip global code carried recording identity (90% kNN on 6 held-out recordings) but swapping it barely changed the sound** in the spectrogram models.

## What the latent encodes (voc_g, log-mel encoder, nothing handed to the decoder; eval_probe.py)
Ridge probes fit on 10 TRAIN recordings, scored on 6 held-out recordings (out-of-sample R^2):
| Quantity | all 24 dims | g1 (0.01 s) | g2 (0.05 s, dims 6-10) | g3 (0.25 s) | g4 (1 s) | g5 (3 s) |
|---|---|---|---|---|---|---|
| loudness | 0.98 | -0.12 | **0.98** | 0.20 | 0.58 | -0.17 |
| log2 f0 (voiced frames) | 0.62 | -0.12 | **0.74** | -0.18 | 0.23 | 0.21 |
| log brightness (spectral centroid) | 0.81 | 0.06 | 0.39 | -0.15 | 0.51 | 0.14 |
| periodicity (harmonic vs noisy) | 0.40 | -0.02 | 0.42 | 0.33 | 0.37 | 0.09 |
Reading: loudness (and mostly pitch) sit in one 5-dimension, 0.05 s group; brightness is spread over the 0.05 s and 1 s groups; the fastest unsmoothed group carries nothing readable. CAVEAT: this model was trained with a weak auxiliary linear probe from the latent to f0 and loudness (train_voc.py AUXF/AUXL), so the clean pitch/loudness structure is partly imposed; brightness and periodicity were never supervised, so those are emergent.

## Does CREPE pitch remove the DDSP "buzz"? (ddsp_v1c_crepe vs ddsp_v1)
Same config, only the pitch source differs (yin-raw vs CREPE+Viterbi). Held-out val mss .827 vs .826; Maisky test .781 vs .783; fixed 15 s window log-mel err .457 vs .443; per-frame error 99th percentile .646 vs .630, 0% of frames above 1.5 for both. So the metrics show no glitch frames and no gain from cleaner pitch; whether the buzz changes is a listening question (`report/system_comparison4.mp4` clips 2 vs 3).

## Limitations (do not skip)
- Short runs (4000 steps for the sweep, 3000-5000 for others). Only ONE replicate exists (voc_g_seed1, matching the original within 1% at sampled decoding); other configs are single runs.
- Spectral loss and log-mel error are proxies; no listening test beyond Quinn's earlier informal comments (DDSP v1: "synthetic but much better than before", "buzzy, no bow noise").
- ddsp_v1 and the +controls vocoders use yin pitch from the same recordings as their teacher; the pitch-probe error is measured against yin and is unreliable on at least one validation recording (cdbdb38e, yin vs CREPE disagree on 41% of frames).
- DDSP v1 was not trained on CREPE pitch or retrained after pitch cleaning (that run was stopped when the direction changed).
- MERT result is for a frozen 94M model, 4 cached layers, a small trainable head, 4000 steps. It says nothing about fine-tuning MERT or about MERT for interpretability/probing.
- Rates are KL estimates on 20 s windows (not an actual entropy coder).

## Files
`LAB_NOTEBOOK.md` (full log incl. mistakes), `compare_systems.json`, `voc_*.json` (training curves/evals), `eval_tau.py`, `eval_gap.py`, `eval_ddsp_tau.py`, `train_voc.py`, `voc_model.py`, `train_ddsp.py`, `ddsp_model.py`, `report/system_comparison*.mp4` (level-matched listening comparisons of the same 15 s Maisky window).
