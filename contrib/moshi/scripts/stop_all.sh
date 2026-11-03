#!/bin/bash
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc|prof_tp|prof_lm|run_e2e|torchrun|dep_shard|bucket_test"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc|prof_tp|prof_lm|run_e2e|torchrun|VLLM::|dep_shard|bucket_test"); do kill -9 $p 2>/dev/null; done
for i in $(seq 1 30); do pgrep -f "VLLM::|serve_launcher|StageEngineCoreProc" > /dev/null || break; sleep 2; done
sleep 3
echo stopped
