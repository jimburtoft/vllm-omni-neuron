#!/usr/bin/env python3
"""Separate prefill call + frame call vs one fused graph (prefill one row, splice its state, run the frame).
usage: fused_prefill_probe.py <model> B Tb ctx"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.fused import FusedEngine, FusedFrameRows
from qwen3_tts_neuron.batched import _M
P, B, Tb, ctx = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
dev = torch.device("neuron:0"); dt = torch.bfloat16; S = 1024
eng = TTSEngine(P, device=dev, max_ctx=S, dtype=dt, compile=False, prefill_buckets=(64, 128))
fe = FusedEngine(eng, B, S=S, vec_cp=False, compile=False)
bt = fe.bt; c = compile_fn(); H, V, Vc = eng.talker.cfg.hidden, eng.talker.cfg.vocab, eng.cp.cfg.vocab
frame_m = FusedFrameRows(eng, bt, vec_cp=False, ctx=ctx).to(dev)
prefill = c(_M(lambda x, r, p: bt.prefill(x, r, p, ctx=256), bt))
frame = c(frame_m)
class Fused(torch.nn.Module):
    def __init__(s):
        super().__init__(); s.f = frame_m
    def forward(s, x, row, ppos, h, lg, seen, allow, pad, pos, g0, gs, rep, t, k, st, sk):
        hl, l1 = bt.prefill(x, row, ppos, ctx=256)
        h = h.index_copy(0, row, hl); lg = lg.index_copy(0, row, l1)
        seen = seen.index_copy(0, row, torch.zeros(1, V, dtype=torch.bool, device=x.device))
        return s.f(h, lg, seen, allow, pad, pos, g0, gs, rep, t, k, st, sk)
fused = c(Fused().to(dev))
z = lambda *sh, d=dt: torch.zeros(*sh, dtype=d, device=dev)
x = (torch.randn(1, Tb, H) * 0.3).to(dt).to(dev); row = torch.tensor([3], device=dev); ppos = torch.arange(Tb, device=dev)
h = z(B, 1, H); lg = z(B, V, d=torch.float32); seen = z(B, V, d=torch.bool); allow = torch.ones(B, dtype=torch.bool, device=dev)
pad = z(B, 1, H); pos = torch.full((B,), 100, device=dev); g0 = z(B, V, d=torch.float32); gs = z(B, 15, 1, Vc, d=torch.float32)
rep = torch.ones(B, device=dev); t = torch.full((B,), 0.9, device=dev); k = torch.full((B,), 50, dtype=torch.long, device=dev)
def sep():
    hl, l1 = prefill(x, row, ppos)
    hh = h.index_copy(0, row, hl); ll = lg.index_copy(0, row, l1); ss = seen.index_copy(0, row, torch.zeros(1, V, dtype=torch.bool, device=dev))
    return frame(hh, ll, ss, allow, pad, pos, g0, gs, rep, t, k, t, k)[3]
def fus():
    return fused(x, row, ppos, h, lg, seen, allow, pad, pos, g0, gs, rep, t, k, t, k)[3]
def fr():
    return frame(h, lg, seen, allow, pad, pos, g0, gs, rep, t, k, t, k)[3]
def bench(name, f, n=20):
    with torch.inference_mode():
        for _ in range(3): f().cpu()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); f().cpu(); ts.append(time.perf_counter() - t0)
    print(f"[fp] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} B={B} Tb={Tb} {name}: {1000 * sorted(ts)[n // 2]:.2f} ms", flush=True)
bench("frame only", fr); bench("prefill + frame (2 calls)", sep); bench("fused prefill+frame (1 call)", fus)
