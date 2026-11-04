#!/usr/bin/env python3
"""Run the code-predictor frame alone at batch B (current LNC) so its NEFF can be profiled.
usage: cp_profile.py <model> B"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
import qwen3_tts_neuron.fused as fu
from qwen3_tts_neuron.batched import _M
P, B = sys.argv[1], int(sys.argv[2])
dev = torch.device("neuron:0")
eng = TTSEngine(P, device=dev, max_ctx=256, dtype=torch.bfloat16, compile=False)
cp = eng.cp; Vc = cp.cfg.vocab; H = eng.talker.cfg.hidden
torch.manual_seed(0)
h = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev); e0 = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev)
gs = (-torch.log(-torch.log(torch.rand(B, 15, 1, Vc).clamp(1e-10, 1 - 1e-7)))).to(dev)
st = torch.full((B,), 0.9, device=dev); sk = torch.full((B,), 50, dtype=torch.long, device=dev)
if os.environ.get("QTTS_CP_NKI") == "1":
    fu.prepare_cp_nki(cp)
fn = compile_fn()(_M(lambda a, b, g, s, k: fu.cp_frame_vec(cp, a, b, False, g, rows=(s, k))[0], cp))
with torch.inference_mode():
    for _ in range(3): fn(h, e0, gs, st, sk).cpu()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); fn(h, e0, gs, st, sk).cpu(); ts.append(time.perf_counter() - t0)
    o = fn(h, e0, gs, st, sk).cpu()
torch.save(o, f"/home/ubuntu/cp_codes_{os.environ.get('QTTS_CP_NKI','0')}_B{B}.pt")
print(f"[cpp] nki={os.environ.get('QTTS_CP_NKI','0')} B={B} cp frame {1000 * sorted(ts)[10]:.2f} ms", flush=True)
