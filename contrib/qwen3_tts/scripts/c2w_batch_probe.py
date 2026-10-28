#!/usr/bin/env python3
"""Code2Wav crop-window (25 ctx + 25 new frames) device time vs batch size; checks batched == per-row output.
usage: c2w_batch_probe.py <model_dir> [B,...]"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.code2wav import Code2Wav, _Crop
dev = torch.device("neuron:0")
m = Code2Wav(os.path.join(sys.argv[1], "speech_tokenizer"), dtype=torch.bfloat16).to(dev)
c = compile_fn()
Bs = [int(x) for x in (sys.argv[2] if len(sys.argv) > 2 else "1,2,4,8").split(",")]
torch.manual_seed(0)
codes = torch.randint(0, 2048, (max(Bs), 16, 50))
ref = None
with torch.inference_mode():
    for B in Bs:
        fn = c(_Crop(m, 25))
        x = codes[:B].to(dev)
        for _ in range(3): y = fn(x); y.cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); y = fn(x); y = y.cpu(); ts.append(time.perf_counter() - t0)
        if ref is None:
            ref = y
        same = torch.equal(y[:1], ref[:1])
        diff = (y[:1].float() - ref[:1].float()).abs().max().item()
        t = sorted(ts)[10] * 1000
        torch.save(y.float(), f"/home/ubuntu/c2w_B{B}.pt")
        print(f"c2w crop B={B}: {t:.2f} ms ({t/B:.2f} ms/window) row0 equal to B=1: {same} maxdiff {diff:.2e}", flush=True)
