#!/bin/bash
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill -9 $p 2>/dev/null; done; sleep 2
cd ~ && cp kv_amax*.json /tmp/ 2>/dev/null; rm -rf moshi && tar xzf moshi_contrib.tgz && cp /tmp/kv_amax*.json moshi/ 2>/dev/null
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
python - <<PY
import numpy as np, soundfile as sf
a = np.load("/home/ubuntu/goldens/clip_utt2.npy"); s = np.load("/home/ubuntu/goldens/clip_utt1.npy")
x = np.concatenate([a, s] * 7)[: 24000 * 300]   # 5 minutes
sf.write("/home/ubuntu/q5min.wav", x, 24000); print(len(x) / 24000, "s")
PY
cd ~/moshi; export PYTHONPATH=$HOME/moshi NEURON_SKIP_EFA_AFFINITY=1
MOSHI_B=5 python -m moshi_neuron.serving.make_deploy /tmp/long.yaml > /dev/null; grep -n "max_model_len" /tmp/long.yaml
MOSHI_NC_COUNT=4 nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 --deploy-config /tmp/long.yaml --port 8000 --omni --init-timeout 3600 --stage-init-timeout 3600 > ~/serve_long.log 2>&1 &
for i in $(seq 1 240); do grep -q "Application startup complete" ~/serve_long.log && break; sleep 10; done
grep -q "Application startup complete" ~/serve_long.log || { echo NOSTART; grep -iE "Traceback|Error" ~/serve_long.log | tail -5; exit 1; }
python -m moshi_neuron.serving.duplex_client --wav ~/q5min.wav --sessions 2 --out ~/o_long/ > ~/c_long.log 2>&1
grep -h '^{' ~/c_long.log | python3 -c "
import sys, json
for l in sys.stdin:
    d = json.loads(l); d.pop('text', None); print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()})"
echo LONG_DONE
