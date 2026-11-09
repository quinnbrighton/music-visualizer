Cello vocoder: adversarial fine-tuning bundle (built Nov 7 2026)
Start: harm_final_z24lo_s2000.pt (24-channel near-lossless model with temporal prior, pitch coded inside).
Recipe: MPD(2,3,5,7,11) + multi-band complex STFT discriminator, LSGAN adv x1, feature matching x2, DAC multi-scale mel x15,
AdamW 1e-4 betas (0.8, 0.99), per-step decay 0.999996, 1000-step discriminator warm-up, 16384-sample crops.
Env: CELLO_DIR (this folder), OUT_DIR (where checkpoints/clips/log go), RESUME (a *_latest.pt to continue), AMP (bf16 discriminators).
Run: see cello_gan_colab.ipynb. Locally: CELLO_DIR=$PWD OUT_DIR=out INIT=... python train_gan.py NAME 0 30000
