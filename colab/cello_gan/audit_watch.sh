# Every EVERYN steps of note-model training: generate a C major piece from the latest checkpoint and render it to audio
# (notes -> curve4 if finished, else curve3 -> lat4 -> sound-space codec). Prints one "AUDIT ..." line per piece.
# usage: bash audit_watch.sh   env: EVERYN (2000) NOTES (120) STEPS (12000)
R=$HOME/music-visualizer/colab_results; C=$HOME/music-visualizer/colab/cello_gan; A=$R/notelm/audit; mkdir -p $A; EVERYN=${EVERYN:-2000}; NOTES=${NOTES:-120}; STEPS=${STEPS:-12000}
export CELLO_DIR=$C TOK_DIR=$HOME/music-visualizer/tokens64 OUT_DIR=${OUT_DIR:-/tmp/cello_out}
export INIT=harm_final_z24lo_s2000 INZD=24 ZDIM=24 RATESSPEC=6x1,5x5,5x25,4x1,4x5 BETA=0 AMPG=0 BS=2 NQ=12 K=64 PYTHONPATH=$C
next=$EVERYN; cd $C
while [ $next -le $STEPS ]; do
  if grep -q "^EVAL $next " $R/notelm/train.log 2>/dev/null; then
    sleep 5; cp $R/notelm/notelm1_last.pt $A/ck.pt
    CURVE=$R/curve3/curve3_last.pt; grep -q "training done" $R/curve4/train.log 2>/dev/null && CURVE=$R/curve4/curve4_last.pt
    G=$(KEY="C major" NOTES=$NOTES SEED=$next python gen_notes.py $A/ck.pt $A/cmajor_s$next.mid 2>&1 | grep -E "^wrote|^copying" | tr '\n' ' ')
    M=$(python midi_to_audio.py $R/reg3/codec_snd_s8000.pt $R/reg3/lat4_s20000.pt $CURVE $A/cmajor_s$next.mid $A/cmajor_s$next.wav 2>&1 | grep -E "^wrote|Error" | tail -1)
    ffmpeg -y -loglevel error -i $A/cmajor_s$next.wav -filter_complex "[0:a]showwaves=s=480x160:mode=cline:colors=white,format=yuv420p[v]" -map "[v]" -map 0:a -c:v libx264 -crf 32 -c:a aac -b:a 160k $A/cmajor_s$next.mp4
    V=$(grep "^EVAL $next " $R/notelm/train.log | cut -c1-120)
    echo "AUDIT step $next | $(basename $CURVE) | $A/cmajor_s$next.mp4 | ${G} | ${V}"
    next=$((next + EVERYN))
  fi
  sleep 30
done
echo "AUDIT finished"
