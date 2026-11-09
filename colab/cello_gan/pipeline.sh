# After the 64-entry RVQ run plateaus: tokenize, train the token transformer, generate test passages, save everything to Drive, mark DONE.
# Started from the Colab terminal with nohup; a notebook cell (cello_wait_and_release.py) releases the runtime when DONE or FAILED appears.
O=/content/drive/MyDrive/cello_gan_out; F=$O/final; C=/content/cello_gan; mkdir -p $F; rm -f $F/DONE $F/FAILED
fail() { echo "FAILED at: $1 ($(date))" | tee $F/FAILED; exit 1; }
cp /content/tokfiles/*.py $C/ || fail copy
export CELLO_DIR=$C OUT_DIR=$O INIT=harm_final_z24lo_s2000 INZD=24 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5 BETA=0 AMPG=0 BS=2 NQ=12 K=64
echo "1/5 waiting for the RVQ plateau ($(date))" > $F/STATUS
python $C/plateau_watch.py $O/harm_rvq64.json $O/harm_rvq64.log >> $F/pipeline.log 2>&1 || fail watch
pkill -f 'train_rvq.py harm_rvq64'; sleep 15
python -c "import torch; c=torch.load('$O/harm_rvq64_latest.pt', map_location='cpu', weights_only=False); torch.save({k: c[k] for k in ('model','rvq','step','args')}, '$F/harm_rvq64_final.pt'); print('RVQ final step', c['step'])" >> $F/pipeline.log 2>&1 || fail save_rvq
cp $O/harm_rvq64.json $O/harm_rvq64.log $F/ 2>/dev/null
echo "2/5 tokenizing ($(date))" > $F/STATUS
TOK_DIR=/content/tokens RVQ_RESUME=$O/harm_rvq64_latest.pt python $C/tokenize_rvq.py > $F/tokenize.log 2>&1 || fail tokenize
tar -czf $F/tokens64.tar.gz -C /content tokens/tokens tokens/controls || fail tar
echo "3/5 training the token transformer ($(date))" > $F/STATUS
TOK_DIR=/content/tokens OUT=$F BS=64 python $C/train_tok2.py tok64 15000 > $F/train_tok2.log 2>&1 || fail transformer
echo "4/5 generating test passages ($(date))" > $F/STATUS
TOK_DIR=/content/tokens RVQ_RESUME=$O/harm_rvq64_latest.pt GS=1,3 python $C/gen_tok2.py $F/tok64_best.pt $F/gen > $F/gen.log 2>&1 || fail generate
echo "5/5 done ($(date))" > $F/STATUS; date > $F/DONE
