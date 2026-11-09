# C major audit pieces from the cello fine-tune of the symbolic note model, every EVERYN steps. Prints one "AUDIT ..." line per piece.
R=$HOME/music-visualizer/colab_results; C=$HOME/music-visualizer/colab/cello_gan; A=$R/symlm/audit; mkdir -p $A; EVERYN=${EVERYN:-1000}; STEPS=${STEPS:-4000}
export CELLO_DIR=$C TOK_DIR=$HOME/music-visualizer/tokens64 OUT_DIR=${OUT_DIR:-/tmp/cello_out}
export INIT=harm_final_z24lo_s2000 INZD=24 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5 BETA=0 AMPG=0 BS=2 NQ=12 K=64 PYTHONPATH=$C
next=$EVERYN; cd $C
while [ $next -le $STEPS ]; do
  if grep -q "^EVAL $next " $R/symlm/cello.log 2>/dev/null; then
    sleep 5; cp $R/symlm/sym_cello_last.pt $A/ck.pt
    G=$(STYLE=cello KEY="C major" METER=3/4 BARS=24 BPM=76 SEED=1 python gen_sym.py $A/ck.pt $A/cmajor_ft$next.mid 2>&1 | grep -E "^wrote|^copying" | sed 's#.*audit/##' | tr '\n' ' ')
    python midi_to_audio.py $R/reg3/codec_snd_s8000.pt $R/reg3/lat4_s20000.pt $R/curve3/curve3_last.pt $A/cmajor_ft$next.mid $A/cmajor_ft$next.wav > /dev/null 2>&1
    ffmpeg -y -loglevel error -i $A/cmajor_ft$next.wav -filter_complex "[0:a]showwaves=s=480x160:mode=cline:colors=white,format=yuv420p[v]" -map "[v]" -map 0:a -c:v libx264 -crf 32 -c:a aac -b:a 160k $A/cmajor_ft$next.mp4
    echo "AUDIT step $next | $A/cmajor_ft$next.mp4 | $G | $(grep "^EVAL $next " $R/symlm/cello.log)"
    next=$((next + EVERYN))
  fi
  sleep 20
done
echo "AUDIT finished"
