#!/bin/bash
# usage: serve_test.sh <B> <sessions> <tag>   (MOSHI_* env passed through to the server)
B=$1; N=$2; TAG=$3
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill -9 $p 2>/dev/null; done; sleep 2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
cd ~/moshi; export PYTHONPATH=$HOME/moshi NEURON_SKIP_EFA_AFFINITY=1
MOSHI_B=$B python -m moshi_neuron.serving.make_deploy /tmp/m_$TAG.yaml > /dev/null
MOSHI_NC_COUNT=4 nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
  --deploy-config /tmp/m_$TAG.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600 > ~/serve_$TAG.log 2>&1 &
for i in $(seq 1 180); do grep -q "Application startup complete" ~/serve_$TAG.log && break; sleep 10; done
grep -q "Application startup complete" ~/serve_$TAG.log || { echo "NOSTART"; grep -iE "Traceback|Error" ~/serve_$TAG.log | tail -5; exit 1; }
python - <<PY
import numpy as np, soundfile as sf
sf.write("/home/ubuntu/q.wav", np.load("/home/ubuntu/goldens/clip_utt2.npy"), 24000)
PY
python -m moshi_neuron.serving.duplex_client --wav ~/q.wav --sessions $N --out ~/o_$TAG/ 2>&1 | grep '^{' | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); d.pop('text', None); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})"
echo DONE_$TAG
