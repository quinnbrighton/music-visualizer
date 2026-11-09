"""Save the continuous (unquantized) 24-channel 25 Hz latent of every recording from the RVQ codec's encoder -> TOK_DIR/latents/{k}.npy (T25,24) float16,
then check how well the decoder reconstructs from it vs 12- and 4-stage quantization (Maisky window).  usage: python encode_latents.py CODEC.pt"""
import os, sys
_codec = sys.argv[1]; sys.argv = ["x", "lat", "1"]; os.environ.setdefault("NINIT", "1")
_s = open(os.path.join(os.environ["CELLO_DIR"], "train_rvq.py")).read(); exec(_s[: _s.index("\n\nif START2 == 1")])
c = torch.load(_codec, map_location=dev, weights_only=False); model.load_state_dict(c["model"]); rvq.load_state_dict(c["rvq"]); model.eval()
import soundfile as sf
OUTT = os.environ["TOK_DIR"]; os.makedirs(f"{OUTT}/latents", exist_ok=True); DS_ = HOP * DSR
with torch.no_grad():
    for k in train + val + test:
        a = A[k]; N = len(a) // DS_ * DS_; CH, CT = 20 * SR, SR; parts = []
        for c0 in range(0, N, CH):
            s, e = max(0, c0 - CT), min(N, c0 + CH + CT); x = torch.from_numpy(np.asarray(a[s:e], dtype=np.float32))[None].to(dev)
            z = to25(feats(x, None), (e - s) // HOP); lo = (c0 - s) // DS_; parts.append(z[0, :, lo: lo + min(CH, N - c0) // DS_].t().cpu())
        Z = torch.cat(parts).numpy(); np.save(f"{OUTT}/latents/{k}.npy", Z.astype(np.float16)); print(k, Z.shape, flush=True)
    k = test[0]; s0 = 5572800; n_ = 12 * SR; T = n_ // HOP; x = torch.from_numpy(np.asarray(A[k][s0:s0 + n_], np.float32))[None].to(dev)
    Z = torch.from_numpy(np.load(f"{OUTT}/latents/{k}.npy").astype(np.float32))[s0 // DS_: s0 // DS_ + n_ // DS_].t()[None].to(dev)
    for tag, n in [("continuous", 0), ("q12", 12), ("q4", 4)]:
        zz = Z if n == 0 else quant(Z, n)[0]; hh = dec_feats(zz, T); y = model.synth(hh, model.pitch_f0(hh), T)
        print(tag, "mss %.4f" % MSS(y, x).item(), "lm %.4f" % logmel_dist(y[0].cpu(), x[0].cpu())["lm"], flush=True); sf.write(f"{OUTT}/maisky_{tag}.wav", y[0].cpu().numpy() / (y.abs().max().item() + 1e-6) * 0.5, SR)
