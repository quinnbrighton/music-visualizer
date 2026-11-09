# Stronger regularization: temporal predictability (PREDW) + stronger decorrelation/scale/noise, from the first regularized run at step 6000 -> lat3. Resumable.
# Start from the Colab terminal: nohup bash /content/regfiles/reg_pipeline.sh > /dev/null 2>&1 &   then run wait_and_release_reg.py in a notebook cell.
O=/content/drive/MyDrive/cello_gan_out; F=$O/reg2; C=/content/cello_gan; T=/content/tokens; mkdir -p $F $T; rm -f $F/DONE $F/FAILED
fail() { echo "FAILED at: $1 ($(date))" | tee $F/FAILED; exit 1; }
cp /content/regfiles/*.py $C/ || fail copy
tar -xzf $O/final/tokens64.tar.gz -C /content || fail tokens   # tokens/controls (pitch, periodicity, loudness)
tar -xzf /content/regfiles/controls2.tar.gz -C $T || fail controls2   # brightness, noisiness
export CELLO_DIR=$C OUT_DIR=$F INIT=harm_final_z24lo_s2000 INZD=24 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5 BETA=0 CW=0.05 TW=8 PW=1 PRAMP=1
export ADV=1 FMW=2 MSW=15 WARM=0 CROP=16384 DLR=1e-4 AMP=1 AMPG=1 GLR=5e-5 NQ=12 K=64 EVERY=1000
echo "1/4 training the regularized codec ($(date))" > $F/STATUS
if [ ! -f $F/codec_pred_s8000.pt ]; then
  BS=64 PREDW=2 COVW=20 SCALEW=2 NOISE=0.15 REG_INIT=$O/reg/codec_reg_s6000.pt REG_DISC=$O/reg/codec_reg_latest.pt REG_RESUME=$F/codec_pred_latest.pt python -u $C/train_reg.py codec_pred 8000 >> $F/codec_pred.log 2>&1 || fail codec
fi
echo "2/4 encoding latents ($(date))" > $F/STATUS
BS=2 TOK_DIR=$T python $C/encode_latents.py $F/codec_pred_s8000.pt > $F/encode.log 2>&1 || fail encode
tar -czf $F/latents_pred.tar.gz -C $T latents || fail tar
echo "3/4 training the latent model lat3 ($(date))" > $F/STATUS
mkdir -p $F/lat3; [ -f $F/lat3/lat3_s20000.pt ] || TOK_DIR=$T OUT=$F/lat3 python -u $C/train_lat.py lat3 20000 $F/codec_pred_s8000.pt > $F/lat3/train.log 2>&1 || fail lat3
echo "4/4 generating and measuring ($(date))" > $F/STATUS
BS=2 TOK_DIR=$T GS=1,2 python $C/gen_lat.py $F/codec_pred_s8000.pt $F/lat3/lat3_s20000.pt $F/lat3/gen > $F/gen.log 2>&1 || fail generate
BS=2 TOK_DIR=$T python $C/latent_sensitivity.py $F/codec_pred_s8000.pt $F/lat3/lat3_s20000.pt $F/sensitivity_pred.json > $F/sens.log 2>&1 || fail sensitivity
echo "done ($(date))" > $F/STATUS; date > $F/DONE
