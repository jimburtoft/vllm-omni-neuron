#!/usr/bin/env python3
"""Device time of the fused frame's parts at batch B: talker decode alone, code predictor alone, full frame.
usage: frame_split.py <model_dir> [B] [ctx]"""
import os, sys, time
import torch
import torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.fused import FusedEngine, FusedFrameRows, cp_frame_vec
from qwen3_tts_neuron.batched import _M
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
ctx = int(sys.argv[3]) if len(sys.argv) > 3 else 256
dev, dt, S = torch.device("neuron:0"), torch.bfloat16, 2048
eng = TTSEngine(sys.argv[1], device=dev, max_ctx=S, dtype=dt, compile=False)
fe = FusedEngine(eng, B, S=S, vec_cp=True, compile=False)
c = compile_fn()
ff = FusedFrameRows(eng, fe.bt, vec_cp=True, ctx=ctx).to(dev)
bt = fe.bt
dec = c(_M(lambda x, p: bt.decode(x, p, ctx=ctx), bt))
cpf = c(_M(lambda h, e0, g, st, sk: cp_frame_vec(eng.cp, h, e0, True, g, rows=(st, sk)), eng.cp, ff))
frame = c(ff)
H, V, Vc = eng.talker.cfg.hidden, eng.talker.cfg.vocab, eng.cp.cfg.vocab
h = torch.zeros(B, 1, H, dtype=dt, device=dev); lg = torch.zeros(B, V, device=dev)
seen = torch.zeros(B, V, dtype=torch.bool, device=dev); pad = torch.zeros(B, 1, H, dtype=dt, device=dev)
pos = torch.full((B,), 50, device=dev); z = torch.zeros(B, device=dev); zk = torch.zeros(B, dtype=torch.long, device=dev)
rep = torch.ones(B, device=dev); allow = torch.ones(B, dtype=torch.bool, device=dev)
g0 = torch.zeros(B, V, device=dev); gs = torch.zeros(B, 15, 1, Vc, device=dev)
def bench(name, fn, n=30):
    for _ in range(3): out = fn()
    (out[-1] if isinstance(out, tuple) else out).cpu()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); out = fn(); (out[-1] if isinstance(out, tuple) else out).cpu(); ts.append(time.perf_counter() - t0)
    print(f"{name}: {1000*sorted(ts)[n//2]:.2f} ms")
with torch.inference_mode():
    bench(f"talker decode B={B} ctx={ctx}", lambda: dec(h, pos))
    bench(f"code predictor B={B}", lambda: cpf(h, h, gs, z, zk))
    bench(f"fused frame B={B}", lambda: frame(h, lg, seen, allow, pad, pos, g0, gs, rep, z, zk, z, zk))
    import qwen3_tts_neuron.fused as fu
    cpg = c(_M(lambda h, e0: cp_frame_vec(eng.cp, h, e0, True, None, 0.0, 0), eng.cp))  # static greedy sampler
    bench(f"code predictor B={B} static-greedy (no per-row topk)", lambda: cpg(h, h))
    cps = c(_M(lambda h, e0: fu.cp_frame_vec(eng.cp, h, e0, False, None, 0.0, 0), eng.cp))
    bench(f"code predictor B={B} matmul attention", lambda: cps(h, h))
