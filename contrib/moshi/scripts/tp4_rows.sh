#!/bin/bash
# Throughput layout (TP4, codec on TP ranks) + rows-per-step measurement. usage: tp4_rows.sh <tag> "<session counts>"
# extra env (e.g. MOSHI_FRAMESYNC=1) is passed through to the server.
TAG=$1; NS=$2
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc"); do kill -9 $p 2>/dev/null; done; sleep 2
source /opt/aws_neuronx_venv_pytorch_inference_vllm_omni_0_29_0_0_2_0/bin/activate
cd ~/moshi; export PYTHONPATH=$HOME/moshi:$HOME/moshi/scripts/fs_site NEURON_SKIP_EFA_AFFINITY=1
[ -f ~/q.wav ] || python -c "
import numpy as np, soundfile as sf
sf.write('/home/ubuntu/q.wav', np.load('/home/ubuntu/goldens/clip_utt2.npy'), 24000)"
R=~/runs/$TAG; mkdir -p $R
MOSHI_B=${MOSHI_B:-32} MOSHI_TP0=4 MOSHI_DEV0=0,1,2,3 MOSHI_DEV1=0 MOSHI_S0_DECODE=1 MOSHI_MLP_XW=1 \
  python -m moshi_neuron.serving.make_deploy $R/deploy.yaml > /dev/null || exit 1
export MOSHI_S0_DECODE=1 MOSHI_MLP_XW=1 MOSHI_BUCKETS=${MOSHI_BUCKETS:-1,2,4,8,16,24,32} MOSHI_CTX_BUCKETS=512,1024,2048 \
  MOSHI_NC_COUNT=4 MOSHI_STEP_STATS=1 MOSHI_TRACE=$R/trace MOSHI_PHASES=${MOSHI_PHASES-$R/ph} MOSHI_PARTS=${MOSHI_PARTS-1} MOSHI_CONS_LOG=${MOSHI_CONS_LOG-1}
MOSHI_DEP_SHARD=${MOSHI_DEP_SHARD:-1} nohup python -m moshi_neuron.serving.serve_launcher serve kyutai/moshiko-pytorch-bf16 \
  --deploy-config $R/deploy.yaml --port 8000 --omni --init-timeout 10800 --stage-init-timeout 10800 > $R/serve.log 2>&1 &
for i in $(seq 1 1100); do grep -q "Application startup complete" $R/serve.log && break; grep -q "initialization failed" $R/serve.log && break; sleep 10; done
grep -q "Application startup complete" $R/serve.log || { echo NOSTART; grep -iE "Error|Traceback" $R/serve.log | head -20 | cut -c1-300; exit 1; }
echo "$TAG up"
for n in $NS; do
  for f in $R/trace.s0.*; do [ -f $f ] && : > $f; done
  pidstat -u -t -h 5 > $R/pidstat_$n.log 2>&1 & PS=$!
  for f in $R/ph.ec.*; do [ -f $f ] && : > $f; done
  python -m moshi_neuron.serving.duplex_client --wav ${WAV:-$HOME/q.wav} --sessions $n --out $R/o_$n/ > $R/c_$n.log 2>&1 &
  CL=$!
  if [ "${SPY:-0}" = "1" ]; then
    sleep 14
    for p in $(pgrep -f "StageEngineCoreProc_stage0|Worker_TP0|VLLM::Worker_TP0"); do
      sudo env "PATH=$PATH" py-spy record -p $p -d 8 -r 200 -f raw -o $R/spy_${n}_$p.txt --nonblocking > /dev/null 2>&1 &
    done
    for p in $(pgrep -f "serve_launcher"); do
      sudo env "PATH=$PATH" py-spy record -p $p -d 8 -r 200 -f raw -o $R/spy_${n}_api_$p.txt --nonblocking > /dev/null 2>&1 &
    done
    ps -eo pid,comm,args | grep -E "VLLM|Stage|serve_launcher" | grep -v grep > $R/ps_$n.txt
  fi
  wait $CL
  kill $PS
  mkdir -p $R/ph_$n; cp $R/ph.ec.* $R/ph_$n/ 2>/dev/null
  mkdir -p $R/tr_$n; cp $R/trace.s0.* $R/tr_$n/ 2>/dev/null
  python3 ~/rows.py $R $n
done
echo "DONE_$TAG"
