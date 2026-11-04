#!/usr/bin/env python3
"""Ablate cp_step sections on device (B, pos fixed) to see where the per-step time goes. Variants via a kernel arg
`skip`: 0 full, 1 no attention core (scores/softmax/PV), 2 no MLP, 3 projections only (no attention core, no norms/rope).
usage: cp_kernel_parts.py <model> B pos"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.kernels.cp_nki import cp_params, cp_step
from qwen3_tts_neuron.batched import _M
P, B, pos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("neuron:0"); dt = torch.bfloat16
eng = TTSEngine(P, device=dev, max_ctx=64, dtype=dt, compile=False)
cp = eng.cp; cfg = cp.cfg; S = cp.S; L = len(cp.layers); lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
x = (torch.randn(B, cfg.hidden) * 0.5).to(dt).to(dev)
# also time just the 4 projections' weight streaming for one step (5 layers) as the floor
kcl = (torch.randn(L, B, S, 1024) * 0.3).to(dt).to(dev); vcl = (torch.randn(L, B, S, 1024) * 0.3).to(dt).to(dev)
pp = {k: v.to(dev) for k, v in cp_params(cp).items()}
k = wrap_nki(cp_step)
for skip in (0, 5, 1):
    fn = compile_fn()(_M(lambda a, b, c, sk=skip: k[lnc](a, b, c, pp["wqkv"], pp["wo"], pp["wgu"], pp["wd"], pp["ln1"], pp["ln2"],
                                                         pp["qn"], pp["kn"], pp["norm"], pp["cos"], pp["sin"], pos, cfg.eps, skip=sk), cp))
    with torch.inference_mode():
        for _ in range(3): fn(x, kcl, vcl).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); fn(x, kcl, vcl).cpu(); ts.append(time.perf_counter() - t0)
    names = {0: "full", 1: "no attn", 2: "no MLP", 3: "Pool+DVE split", 4: "bf16", 5: "attention in 4 big ops"}
    print(f"[ckp] B={B} pos={pos} skip={skip} ({names[skip]}): {1000 * sorted(ts)[10]:.2f} ms", flush=True)
