#!/usr/bin/env python3
"""Greedy code-predictor (matmul form) on device vs CPU, same dtype (bf16 or fp32): per group, cosine of the
hidden state and logits, to see whether device CP numerics diverge (beyond bf16 noise) at LNC=2 vs LNC=1."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine, rms
from qwen3_tts_neuron.batched import _M
DT = getattr(torch, sys.argv[2] if len(sys.argv) > 2 else "float32")
B = 4; dev = torch.device("neuron:0")
engd = TTSEngine(sys.argv[1], device=dev, max_ctx=256, dtype=DT, compile=False)
engc = TTSEngine(sys.argv[1], device="cpu", max_ctx=256, dtype=DT, compile=False)
H = engc.talker.cfg.hidden
torch.manual_seed(0)
h = (torch.randn(B, 1, H) * 0.5).to(DT); e0 = (torch.randn(B, 1, H) * 0.5).to(DT)
def first_logits(cp, h, e0):
    cfg = cp.cfg; dev_ = h.device; L = len(cp.layers)
    kc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=dev_) for _ in range(L)]
    vc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=dev_) for _ in range(L)]
    j = torch.arange(cp.S, device=dev_); pos = torch.arange(2, device=dev_)
    cos, sin = cp.cos.index_select(0, pos), cp.sin.index_select(0, pos)
    m = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32)
    x = torch.cat([h, e0], 1) @ cp.proj_w + cp.proj_b
    outs = [x]
    for i, l in enumerate(cp.layers):
        x, kc[i], vc[i] = l.forward_local(x, cos, sin, kc[i], vc[i], pos, m)
        outs.append(x)
    hh = rms(x[:, -1:, :], cp.norm, cfg.eps)
    outs.append((hh @ cp.heads[0]).float())
    return outs
ref = first_logits(engc.cp, h, e0)
with torch.inference_mode():
    n = len(ref)
    fn = compile_fn()(_M(lambda a, b: torch.cat([o.float().reshape(B, -1) for o in first_logits(engd.cp, a, b)], 1), engd.cp))
    flat = fn(h.to(dev), e0.to(dev)).cpu()
off = 0
for i, r in enumerate(ref):
    k = r.reshape(B, -1).shape[1]; o = flat[:, off:off + k]; off += k
    cos = F.cosine_similarity(o.float(), r.reshape(B, -1).float(), dim=-1).min().item()
    print(f"[cpg] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} {str(DT)[6:]} stage {i} ({'in' if i == 0 else 'logits' if i == n - 1 else 'layer'+str(i-1)}): min cos {cos:.6f}")
r = ref[-1].reshape(B, -1); o = flat[:, off - r.shape[1]:off]
print(f"[cpg] argmax agree {int((o.argmax(-1) == r.argmax(-1)).sum())}/{B}")
