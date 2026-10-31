#!/usr/bin/env python3
"""On-device check of sample_rows (per-row top-k / temperature Gumbel sampling) vs CPU, same inputs.
usage: NEURON_LOGICAL_NC_CONFIG=<1|2> sample_probe.py"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.fused import sample_rows
dev = torch.device("neuron:0")
torch.manual_seed(0)
B, V = 48, 3072
for name, shape in (("talker", (B, V)), ("cp", (B, 1, 2048))):
    lg = torch.randn(*shape) * 3
    g = -torch.log(-torch.log(torch.rand(*shape).clamp(1e-10, 1 - 1e-7)))
    temp = torch.full((B,), 0.9); temp[:4] = 0.0
    topk = torch.full((B,), 50, dtype=torch.long); topk[4:8] = 0
    ref = sample_rows(lg, g, temp, topk)
    f = compile_fn()(lambda a, b, c, d: sample_rows(a, b, c, d))
    with torch.inference_mode():
        out = f(lg.to(dev), g.to(dev), temp.to(dev), topk.to(dev)).cpu()
        # pieces: topk values and k-th threshold
        kv = compile_fn()(lambda a: torch.topk(a, 64, dim=-1).values)(lg.to(dev)).cpu()
    kv_ref = torch.topk(lg, 64, dim=-1).values
    print(f"[sample] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} {name}: rows equal {int((out == ref).sum())}/{out.numel()} | "
          f"topk values maxdiff {(kv - kv_ref).abs().max().item():.3e} sorted-desc {bool((kv[..., :-1] >= kv[..., 1:]).all())}", flush=True)
