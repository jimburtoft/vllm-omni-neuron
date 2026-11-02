#!/bin/bash
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc|prof_tp|prof_lm|run_e2e|torchrun"); do kill $p 2>/dev/null; done; sleep 8
for p in $(pgrep -f "serve_launcher|StageEngineCoreProc|prof_tp|prof_lm|run_e2e|torchrun|VLLM::"); do kill -9 $p 2>/dev/null; done; sleep 3
echo stopped
