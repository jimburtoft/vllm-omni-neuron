#!/bin/bash
# two B=$1 servers (cores 0/1 port 8000, cores 2/3 port 8001), $1 sessions each, all concurrent
B=$1; TAG=two$B
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill -9 $p 2>/dev/null; done; sleep 2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
cd ~/moshi; export PYTHONPATH=$HOME/moshi NEURON_SKIP_EFA_AFFINITY=1
MOSHI_B=$B python -m moshi_neuron.serving.make_deploy /tmp/a.yaml > /dev/null
MOSHI_B=$B MOSHI_DEV0=2 MOSHI_DEV1=3 python -m moshi_neuron.serving.make_deploy /tmp/b.yaml > /dev/null
MOSHI_NC_COUNT=4 nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 --deploy-config /tmp/a.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600 > ~/serve_${TAG}a.log 2>&1 &
sleep 60
MOSHI_NC_COUNT=4 nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 --deploy-config /tmp/b.yaml --port 8001 --omni --init-timeout 3600 --stage-init-timeout 3600 > ~/serve_${TAG}b.log 2>&1 &
for s in a b; do for i in $(seq 1 180); do grep -q "Application startup complete" ~/serve_${TAG}$s.log && break; sleep 10; done; done
grep -q "Application startup complete" ~/serve_${TAG}b.log || { echo NOSTART; exit 1; }
( python -m moshi_neuron.serving.duplex_client --url "ws://127.0.0.1:8000/v1/realtime?duplex=1" --wav ~/q.wav --sessions $B --out ~/o_${TAG}a/ > ~/c_${TAG}a.log 2>&1 &
  python -m moshi_neuron.serving.duplex_client --url "ws://127.0.0.1:8001/v1/realtime?duplex=1" --wav ~/q.wav --sessions $B --out ~/o_${TAG}b/ > ~/c_${TAG}b.log 2>&1; wait )
grep -h '^{' ~/c_${TAG}a.log ~/c_${TAG}b.log | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); d.pop('text', None); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})"
echo DONE_$TAG
