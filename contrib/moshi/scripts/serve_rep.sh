#!/bin/bash
# one server: stage-0 replicas on $DEV0 (B each), shared decoder on $DEV1; client $N sessions
B=$1; DEV0=$2; DEV1=$3; N=$4; TAG=$5
R=$(echo $DEV0 | tr ',' '\n' | wc -l)
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill -9 $p 2>/dev/null; done; sleep 2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
cd ~/moshi; export PYTHONPATH=$HOME/moshi NEURON_SKIP_EFA_AFFINITY=1
MOSHI_B=$B MOSHI_REP0=$R MOSHI_DEV0=$DEV0 MOSHI_DEV1=$DEV1 python -m moshi_neuron.serving.make_deploy /tmp/r_$TAG.yaml > /dev/null || exit 1
grep -n "num_replicas\|devices\|max_num_seqs\|max_sessions" /tmp/r_$TAG.yaml
MOSHI_NC_COUNT=4 nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
  --deploy-config /tmp/r_$TAG.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600 > ~/serve_$TAG.log 2>&1 &
for i in $(seq 1 240); do grep -q "Application startup complete" ~/serve_$TAG.log && break; sleep 10; done
grep -q "Application startup complete" ~/serve_$TAG.log || { echo NOSTART; grep -iE "Traceback|Error" ~/serve_$TAG.log | grep -v "Chat\|base.py" | tail -5 | cut -c1-300; exit 1; }
python -m moshi_neuron.serving.duplex_client --wav ~/q.wav --sessions $N --out ~/o_$TAG/ > ~/c_$TAG.log 2>&1
grep -h '^{' ~/c_$TAG.log | python3 -c "
import sys, json
r = [json.loads(l) for l in sys.stdin]
print('sessions', len(r), 'min frames', min(x['chunks'] for x in r), 'first audio max', round(max(x['first_audio_s'] for x in r), 3),
      'p50', round(min(x['lag_p50_ms'] for x in r)), '-', round(max(x['lag_p50_ms'] for x in r)), 'p99 max', round(max(x['lag_p99_ms'] for x in r)), 'max', round(max(x['lag_max_ms'] for x in r)))"
echo DONE_$TAG
