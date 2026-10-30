#!/usr/bin/env python3
"""Device time of an R-row prefill call vs R single-row calls (B-slot talker). usage: prefill_rows_probe.py <model> B Tb ctx"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.fused import FusedEngine
from qwen3_tts_neuron.batched import _M
B, Tb, ctx = int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dev = torch.device("neuron:0"); dt = torch.bfloat16; S = 2048
eng = TTSEngine(sys.argv[1], device=dev, max_ctx=S, dtype=dt, compile=False, prefill_buckets=(64, 128, 256))
fe = FusedEngine(eng, B, S=S, vec_cp=False, compile=False)
bt = fe.bt; c = compile_fn(); H = eng.talker.cfg.hidden
one = c(_M(lambda x, r, p: bt.prefill(x, r, p, ctx=ctx), bt))
many = c(_M(lambda x, r, p: bt.prefill_rows(x, r, p, ctx=ctx), bt))
def bench(f, n=15):
    for _ in range(3): f()[1].cpu()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f()[1].cpu(); ts.append(time.perf_counter() - t0)
    return 1000 * sorted(ts)[n // 2]
with torch.inference_mode():
    x1 = torch.zeros(1, Tb, H, dtype=dt).to(dev); p1 = torch.arange(Tb).to(dev)
    print(f"[pr] B={B} Tb={Tb} ctx={ctx} 1 row: {bench(lambda: one(x1, torch.tensor([3], device=dev), p1)):.2f} ms", flush=True)
    for R in (2, 4, 8):
        xr = torch.zeros(R, Tb, H, dtype=dt).to(dev); pr = torch.arange(Tb).repeat(R, 1).to(dev)
        rows = torch.arange(R).to(dev)
        print(f"[pr] B={B} Tb={Tb} ctx={ctx} {R} rows: {bench(lambda: many(xr, rows, pr)):.2f} ms", flush=True)
