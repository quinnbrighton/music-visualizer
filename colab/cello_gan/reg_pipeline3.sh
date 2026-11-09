# Sound-space version: path length + latent noise + temporal predictability all in raw latent units (RAW=1), global size term only, no decorrelation; from the original rvq64 codec -> lat4. Resumable.
# Start from the Colab terminal: nohup bash /content/regfiles/reg_pipeline.sh > /dev/null 2>&1 &   then run wait_and_release_reg.py in a notebook cell.
O=/content/drive/MyDrive/cello_gan_out; F=$O/reg3; C=/content/cello_gan; T=/content/tokens; mkdir -p $F $T; rm -f $F/DONE $F/FAILED
fail() { echo "FAILED at: $1 ($(date))" | tee $F/FAILED; exit 1; }
cp /content/regfiles/*.py $C/ || fail copy
tar -xzf $O/final/tokens64.tar.gz -C /content || fail tokens   # tokens/controls (pitch, periodicity, loudness)
tar -xzf /content/regfiles/controls2.tar.gz -C $T || fail controls2   # brightness, noisiness
export CELLO_DIR=$C OUT_DIR=$F INIT=harm_final_z24lo_s2000 INZD=24 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5 BETA=0 CW=0.05 TW=8 PW=1 PRAMP=1
export ADV=1 FMW=2 MSW=15 WARM=0 CROP=16384 DLR=1e-4 AMP=1 AMPG=1 GLR=5e-5 NQ=12 K=64 EVERY=1000
echo "1/4 training the regularized codec ($(date))" > $F/STATUS
if [ ! -f $F/codec_snd_s8000.pt ]; then
  BS=64 RAW=1 PREDW=2 COVW=0 SCALEW=1 REG_INIT=$O/harm_rvq64_s5000.pt REG_RESUME=$F/codec_snd_latest.pt python -u $C/train_reg.py codec_snd 8000 >> $F/codec_snd.log 2>&1 || fail codec
fi
echo "2/4 encoding latents ($(date))" > $F/STATUS
BS=2 TOK_DIR=$T python $C/encode_latents.py $F/codec_snd_s8000.pt > $F/encode.log 2>&1 || fail encode
tar -czf $F/latents_snd.tar.gz -C $T latents || fail tar
echo "3/4 training the latent model lat4 ($(date))" > $F/STATUS
mkdir -p $F/lat4; [ -f $F/lat4/lat4_s20000.pt ] || TOK_DIR=$T OUT=$F/lat4 python -u $C/train_lat.py lat4 20000 $F/codec_snd_s8000.pt > $F/lat4/train.log 2>&1 || fail lat4
echo "4/4 generating and measuring ($(date))" > $F/STATUS
BS=2 TOK_DIR=$T GS=1,2 python $C/gen_lat.py $F/codec_snd_s8000.pt $F/lat4/lat4_s20000.pt $F/lat4/gen > $F/gen.log 2>&1 || fail generate
BS=2 TOK_DIR=$T python $C/latent_sensitivity.py $F/codec_snd_s8000.pt $F/lat4/lat4_s20000.pt $F/sensitivity_snd.json > $F/sens.log 2>&1 || fail sensitivity
echo "done ($(date))" > $F/STATUS; date > $F/DONE
