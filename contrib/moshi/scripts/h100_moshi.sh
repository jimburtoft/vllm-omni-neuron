#!/bin/bash
# H100 side of the Moshi comparison -- INTERNAL ONLY.
# upstream vllm-omni PR #9319 (moshiko on the PersonaPlex duplex pipeline), vLLM 0.32.0 container, CUDA graphs on
# (deploy default), same duplex client (drift-free pacing) + 28.8 s clip as the Trn2 runs.
# usage: bash h100_moshi.sh <sessions list, e.g. "1 4 8 12 16"> [max_sessions]
set -e
NS=${1:-"1 4 8 12 16"}; MAXS=${2:-16}
IMG=vllm/vllm-openai:v0.32.0
cd ~/h100
[ -d vllm-omni ] || git clone -q https://github.com/vllm-project/vllm-omni.git
cd vllm-omni && git fetch -q origin pull/9319/head:moshi 2>/dev/null || true; git checkout -q moshi && git log -1 --format='%h %s' && cd ..
cat > overlay.yaml <<Y
base_config: personaplex.yaml
duplex_session:
  max_sessions: $MAXS
stages:
  - stage_id: 0
    max_num_seqs: $MAXS
  - stage_id: 1
    max_num_seqs: $((MAXS + 2))
Y
docker pull -q $IMG > /dev/null
docker rm -f moshi > /dev/null 2>&1 || true
docker run -d --name moshi --gpus all --ipc=host --network host -v ~/h100:/w -v ~/.cache/huggingface:/root/.cache/huggingface \
  --entrypoint bash $IMG -c "pip install -q -e /w/vllm-omni > /w/pip.log 2>&1; pip install -q soundfile librosa websockets >> /w/pip.log 2>&1; \
   cp /w/overlay.yaml /w/vllm-omni/vllm_omni/deploy/moshi_overlay.yaml; \
   vllm serve kyutai/moshiko-pytorch-bf16 --omni --deploy-config /w/vllm-omni/vllm_omni/deploy/moshi_overlay.yaml --port 8000 > /w/serve.log 2>&1" > /dev/null
for i in $(seq 1 160); do grep -q "Application startup complete" serve.log 2>/dev/null && break; sleep 15; done
grep -q "Application startup complete" serve.log || { echo NOSTART; tail -60 serve.log; tail -20 pip.log; exit 1; }
echo "server up"
for n in $NS; do
  docker exec moshi python /w/duplex_client.py --wav /w/clip_utt2.npy --sessions $n --out /w/o_$n/ > c_$n.log 2>&1 || true
  grep -h '^{' c_$n.log | python3 -c "
import sys, json
r = [json.loads(l) for l in sys.stdin]
if r: print('H100 n=$n sessions', len(r), 'min frames', min(x['chunks'] for x in r), 'first audio max', round(max(x['first_audio_s'] for x in r), 3),
      'p50', round(min(x['lag_p50_ms'] for x in r)), '-', round(max(x['lag_p50_ms'] for x in r)), 'p99 max', round(max(x['lag_p99_ms'] for x in r)), 'max', round(max(x['lag_max_ms'] for x in r)))
else: print('H100 n=$n FAILED'); print(open('c_$n.log').read()[-1500:])"
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
done
echo H100_DONE
