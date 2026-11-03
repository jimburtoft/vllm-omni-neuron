#!/bin/bash
# usage: serve_multi.sh <B> <TP> <DEV0> <DEV1> <REPLICAS> <tag> "<client session counts>"
B=$1; TP=$2; DEV0=$3; DEV1=$4; R=$5; TAG=$6; COUNTS=$7
bash ~/stop_all.sh > /dev/null
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
cd ~/moshi; export PYTHONPATH=$HOME/moshi NEURON_SKIP_EFA_AFFINITY=1
[ -f ~/q.wav ] || python -c "
import numpy as np, soundfile as sf
sf.write('/home/ubuntu/q.wav', np.load('/home/ubuntu/goldens/clip_utt2.npy'), 24000)"
MOSHI_B=$B MOSHI_TP0=$TP MOSHI_REP0=$R MOSHI_DEV0=$DEV0 MOSHI_DEV1=$DEV1 python -m moshi_neuron.serving.make_deploy /tmp/m_$TAG.yaml > /dev/null || exit 1
MOSHI_NC_COUNT=${MOSHI_NC_COUNT:-4} nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
  --deploy-config /tmp/m_$TAG.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600 > ~/serve_$TAG.log 2>&1 &
for i in $(seq 1 360); do grep -q "Application startup complete" ~/serve_$TAG.log && break; grep -q "initialization failed" ~/serve_$TAG.log && break; sleep 10; done
grep -q "Application startup complete" ~/serve_$TAG.log || { echo NOSTART; grep -iE "Error" ~/serve_$TAG.log | grep -v "Chat\|max_model_len" | head -5 | cut -c1-250; exit 1; }
for N in $COUNTS; do
  python -m moshi_neuron.serving.duplex_client --wav ~/q.wav --sessions $N --out ~/o_${TAG}_$N/ > ~/c_${TAG}_$N.log 2>&1
  grep -h '^{' ~/c_${TAG}_$N.log | python3 -c "
import sys, json
r = [json.loads(l) for l in sys.stdin]
print('$TAG n=$N sessions', len(r), 'min frames', min(x['chunks'] for x in r), 'first audio max', round(max(x['first_audio_s'] for x in r), 3),
      'p50', round(min(x['lag_p50_ms'] for x in r)), '-', round(max(x['lag_p50_ms'] for x in r)), 'p99 max', round(max(x['lag_p99_ms'] for x in r)), flush=True)"
  grep -hE "MoshiS0\] 200 steps" ~/serve_$TAG.log | tail -1 | cut -c95-300
  sleep 20
done
echo DONE_$TAG
