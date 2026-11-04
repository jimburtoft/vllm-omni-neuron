#!/usr/bin/env python3
"""Code-predictor ablations at batch B: which pieces cost what in the compiled graph.
  full      : current CP frame
  noattn    : attention replaced by v (no QK/softmax/PV) -- cost of attention
  nomlp     : MLP skipped -- cost of MLP matmuls
  dense     : layers only, no sampling/heads/embeds (h stays) -- cost of heads+sampling+feedback
usage: cp_ablate.py <model> B"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine, rms
import qwen3_tts_neuron.engine as en
import qwen3_tts_neuron.fused as fu
from qwen3_tts_neuron.batched import _M
P, B = sys.argv[1], int(sys.argv[2])
dev = torch.device("neuron:0")
eng = TTSEngine(P, device=dev, max_ctx=256, dtype=torch.bfloat16, compile=False)
cp = eng.cp; Vc = cp.cfg.vocab; H = eng.talker.cfg.hidden
h = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev); e0 = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev)
gs = (-torch.log(-torch.log(torch.rand(B, 15, 1, Vc).clamp(1e-10, 1 - 1e-7)))).to(dev)
st = torch.full((B,), 0.9, device=dev); sk = torch.full((B,), 50, dtype=torch.long, device=dev)
L0 = en.Qwen3Layer
orig_attn, orig_mlp = L0._attn, L0._mlp
def run(name):
    if name == "noattn":
        L0._attn = lambda self, q, k, v, m: q.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    if name == "nomlp":
        L0._mlp = lambda self, h: h * 0
    f = (lambda a, b, g, s, k: fu.cp_frame_vec(cp, a, b, False, g, rows=(s, k))[0])
    fn = compile_fn()(_M(f, cp))
    with torch.inference_mode():
        for _ in range(3): fn(h, e0, gs, st, sk).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); fn(h, e0, gs, st, sk).cpu(); ts.append(time.perf_counter() - t0)
    L0._attn, L0._mlp = orig_attn, orig_mlp
    print(f"[abl] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} B={B} {name}: {1000 * sorted(ts)[10]:.2f} ms", flush=True)
for n in os.environ.get("ONLY", "full,noattn,nomlp").split(","):
    run(n)
