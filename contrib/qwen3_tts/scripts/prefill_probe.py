#!/usr/bin/env python3
"""Time the AR worker's slot-prefill path pieces on device: prefill graph per bucket, state index_copy, and
the build-prompt host step. usage: prefill_probe.py <model_dir> [B]"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.fused import FusedEngine
B = int(sys.argv[2]) if len(sys.argv) > 2 else 32
dev = torch.device("neuron:0"); dt = torch.bfloat16; S = 2048
eng = TTSEngine(sys.argv[1], device=dev, max_ctx=S, dtype=dt, compile=False, prefill_buckets=(64, 128, 256))
fe = FusedEngine(eng, B, S=S, vec_cp=True, compile=False)
from qwen3_tts_neuron.batched import _M
PCTX = int(os.environ.get("PCTX", "0"))
bt = fe.bt
pf = compile_fn()(_M(lambda x, r, p: bt.prefill(x, r, p, ctx=PCTX), bt) if PCTX else fe.prefill)
print("PCTX", PCTX)
H, V = eng.talker.cfg.hidden, eng.talker.cfg.vocab
h = torch.zeros(B, 1, H, dtype=dt, device=dev); lg = torch.zeros(B, V, device=dev)
def sync(t): t.cpu()
with torch.inference_mode():
    for Tb in (64, 128, 256):
        x = torch.zeros(1, Tb, H, dtype=dt); p = torch.arange(Tb)
        for _ in range(3):
            hl, l1 = pf(x.to(dev), torch.tensor([3], device=dev), p.to(dev)); sync(l1)
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); hl, l1 = pf(x.to(dev), torch.tensor([3], device=dev), p.to(dev)); sync(l1); ts.append(time.perf_counter() - t0)
        print(f"prefill Tb={Tb}: {1000*sorted(ts)[5]:.2f} ms (synced)")
        ts = []
        for _ in range(10):
            t0 = time.perf_counter(); hl, l1 = pf(x.to(dev), torch.tensor([3], device=dev), p.to(dev)); ts.append(time.perf_counter() - t0)
        sync(l1); print(f"prefill Tb={Tb}: {1000*sorted(ts)[5]:.2f} ms (dispatch only)")
    bi = torch.tensor([3], device=dev)
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        h = h.index_copy(0, bi, hl); lg = lg.index_copy(0, bi, l1); sync(lg)
        ts.append(time.perf_counter() - t0)
    print(f"state index_copy x2 (eager, synced): {1000*sorted(ts)[5]:.2f} ms")
    ts = []
    for _ in range(10):
        t0 = time.perf_counter(); y = torch.zeros(1, 64, H, dtype=dt).to(dev); sync(y); ts.append(time.perf_counter() - t0)
    print(f"H2D 64xH: {1000*sorted(ts)[5]:.2f} ms")
