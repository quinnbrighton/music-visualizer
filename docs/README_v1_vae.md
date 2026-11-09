# Music Visualizer

A music visualizer built on VAE-based audio encoding. The system learns a compact latent representation of audio (cello performances) using a convolutional Variational Autoencoder on mel spectrograms, then trains a temporal Transformer model on latent sequences for generation.

## Current Pipeline

The project uses a three-phase training approach:

1. **Phase 1 -- Bach-only VAE training** (`train_curriculum.py`): Train the ConvMelVAE on a single high-quality recording (Bach Cello Suite No.1) to learn clean spectral representations.
2. **Phase 2 -- All-data fine-tuning** (`train_curriculum.py`): Fine-tune on the full 23-file cello dataset (~885 minutes of audio) with reduced learning rate and stride-8 windowing.
3. **Phase 3 -- Temporal Transformer** (`train_temporal.py`): Train a Transformer to predict next latent vectors given a context window of 128 frames (~1.5 seconds). Enables autoregressive generation of novel audio.

## Architecture

### ConvMelVAE

- **Input**: 512-bin mel spectrogram, 9-frame context window (4 frames on each side)
- **Latent space**: 12 dimensions (VAE with KL divergence)
- **Encoder**: 3 x Conv1d layers (stride 2) with BatchNorm + LeakyReLU, then FC layers to mu/logvar
- **Decoder**: FC layers then 3 x ConvTranspose1d layers back to mel space
- **Loss**: MSE + Spectral Convergence (SC) + beta-weighted KL divergence (beta=0.001)
- **Parameters**: ~2M
- **Audio params**: SR=44100, N_FFT=4096, HOP=512, N_MEL=512

### Temporal Transformer

- **Input**: Sequences of 128 latent vectors (12D each)
- **Architecture**: 4-layer Transformer with 4 attention heads, d_model=128
- **Output**: Next latent vector prediction
- **Generation**: Autoregressive sampling with temperature control

## Key Files

| File | Purpose |
|------|---------|
| `train_curriculum.py` | Main VAE training script (phase 1 and 2) |
| `train_temporal.py` | Temporal Transformer training (phase 3) |
| `models/mel512_vae_12d_curriculum.pt` | Trained curriculum VAE checkpoint |
| `models/bach_mel_vae_12d_curriculum_latent.npz` | Encoded Bach latent vectors (94,644 x 12) |
| `scripts/export_mel_vae_decoder.py` | Export decoder weights to binary for HTML player |
| `ae3d_bach_12d.html` | 3D latent space visualizer (12D projected to 3D) |
| `ae3d_bach_12d.json` | Latent data for the visualizer |
| `latent_player_12d.html` | Interactive latent space audio player |
| `decoder_12d/` | Exported decoder weights (binary) for browser playback |
| `curriculum_phase1.log` | Phase 1 training log |
| `curriculum_phase2.log` | Phase 2 training log |
| `temporal_training.log` | Phase 3 training log (in progress) |

## Data

- `downloads/` -- Source WAV files (Bach performances, Yo-Yo Ma recordings)
- `cello_dataset/` -- Additional cello recordings + preprocessed mel spectrograms in `preprocessed/`
- `models/` -- All model checkpoints and latent encodings

## How to Train

### Phase 1: Bach-only VAE

```bash
python train_curriculum.py
```

This runs phase 1 (300 epochs on Bach Suite No.1) then phase 2 (100 epochs on all data). The script handles both phases sequentially.

### Phase 3: Temporal model

```bash
python train_temporal.py
```

Encodes all audio files through the trained VAE, then trains the Transformer on latent sequences.

## Training Metrics

### Phase 1 (Bach only, 300 epochs)

- Final validation SC: **0.295**
- Final validation MSE: 0.035
- Best val loss: 0.3297
- Training time: ~27 minutes

### Phase 2 (All data, 100 epochs)

- Final validation SC: **0.43**
- Final validation MSE: 0.091
- SC increases from phase 1 due to greater diversity in the full dataset
- Training time: ~31 minutes

## Lessons Learned

- **Log-magnitude loss instability**: Using log-magnitude spectral loss caused training instability and NaN gradients. Switched to linear-magnitude MSE + spectral convergence which is stable and effective.
- **Curriculum learning**: Training on a single clean recording first (Bach) then fine-tuning on diverse data produces better results than training on everything from the start. The model learns clean spectral structure before adapting to variety.
- **Std floor clamping**: The VAE logvar needs clamping (std floor) to prevent posterior collapse. Using `beta=0.001` for KL weight keeps the latent space regularized without killing reconstruction quality.
- **Stride-8 windowing for large datasets**: Phase 2 uses stride-8 instead of stride-1 to keep the training set manageable (571K windows from 885 minutes of audio vs. 4.5M+ at stride-1).
- **Spectral convergence (SC)** is a more perceptually meaningful metric than raw MSE for audio quality assessment.

## Archive

Old experimental scripts from earlier development iterations are preserved in `archive/`. These include STFT-based autoencoders, oscillator-bank approaches, cello physics modeling, harmonic decomposition, PCA-based methods, and various visualization prototypes. Nothing was deleted -- only moved.
