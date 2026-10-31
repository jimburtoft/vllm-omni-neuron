#!/usr/bin/env python3
"""cp_frame_vec with per-row sampling: device (current LNC) vs CPU fp32 on identical inputs and noise.
usage: cp_probe.py <model> [variant]   variant: rows (default) | static | greedy"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.fused import cp_frame_vec
from qwen3_tts_neuron.batched import _M
var = sys.argv[2] if len(sys.argv) > 2 else "rows"
B = 4
dev = torch.device("neuron:0")
engd = TTSEngine(sys.argv[1], device=dev, max_ctx=256, dtype=torch.bfloat16, compile=False)
engc = TTSEngine(sys.argv[1], device="cpu", max_ctx=256, dtype=torch.float32, compile=False)
H, Vc = engc.talker.cfg.hidden, engc.cp.cfg.vocab
torch.manual_seed(0)
h = torch.randn(B, 1, H) * 0.5; e0 = torch.randn(B, 1, H) * 0.5
gs = -torch.log(-torch.log(torch.rand(B, 15, 1, Vc).clamp(1e-10, 1 - 1e-7)))
st = torch.full((B,), 0.9); sk = torch.full((B,), 50, dtype=torch.long)
if var == "greedy":
    st = torch.zeros(B); sk = torch.zeros(B, dtype=torch.long)
import qwen3_tts_neuron.fused as fu
OLD = fu.sample_rows
def sample_rows_old(logits, gumbel, temp, topk):
    B = logits.shape[0]
    shape = (B,) + (1,) * (logits.dim() - 1)
    t = temp.view(shape)
    greedy = t <= 0
    l = logits / torch.where(greedy, 1.0, t)
    kmax = min(fu.TOPK_MAX, l.shape[-1])
    vals = torch.topk(l, kmax, dim=-1).values
    k = topk.view(shape).clamp(min=0, max=kmax)
    kth = torch.gather(vals, -1, (torch.where(k > 0, k, kmax) - 1).long())
    l = torch.where((k > 0) & (l < kth), float("-inf"), l)
    noisy = torch.where(greedy, l, l + gumbel)
    return torch.argmax(noisy, dim=-1)
if var == "rows_old":
    fu.sample_rows = sample_rows_old
def sample_rows_ws(logits, gumbel, temp, topk):
    # gumbel added before the top-k mask (no where on -inf afterwards)
    B = logits.shape[0]
    shape = (B,) + (1,) * (logits.dim() - 1)
    t = temp.view(shape)
    greedy = t <= 0
    kmax = min(fu.TOPK_MAX, logits.shape[-1])
    vals = torch.topk(logits, kmax, dim=-1).values
    k = topk.view(shape).clamp(min=0, max=kmax)
    kth = torch.gather(vals, -1, (torch.where(k > 0, k, kmax) - 1).long())
    keep = (k <= 0) | (logits >= kth)
    score = logits / torch.where(greedy, 1.0, t) + torch.where(greedy, 0.0, 1.0) * gumbel
    return torch.argmax(score + (~keep).to(score.dtype) * -3.0e38, dim=-1)
if var == "rows_ws":
    fu.sample_rows = sample_rows_ws
if var == "rows_unroll_off":
    pass
def run(eng, h, e0, gs, st, sk):
    if var == "static":
        return cp_frame_vec(eng.cp, h, e0, False, gs, 0.9, 50)[0]
    return cp_frame_vec(eng.cp, h, e0, False, gs, rows=(st, sk))[0]
# reference: CPU bf16 with the same function (isolates compile issues from bf16-vs-fp32 drift)
engc = TTSEngine(sys.argv[1], device="cpu", max_ctx=256, dtype=torch.bfloat16, compile=False)
ref = run(engc, h.to(torch.bfloat16), e0.to(torch.bfloat16), gs, st, sk)
fn = compile_fn()(_M(lambda a, b, c, d, e: run(engd, a, b, c, d, e), engd.cp))
with torch.inference_mode():
    out = fn(*(t.to(dev) for t in (h.to(torch.bfloat16), e0.to(torch.bfloat16), gs, st, sk))).cpu()
print(f"[cp] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} var={var} zeros dev {int((out == 0).sum())}/{out.numel()} ref {int((ref == 0).sum())}/{ref.numel()}; first group equal {bool((out[:, 0] == ref[:, 0]).all())}")
print("[cp] dev row0", out[0].tolist()); print("[cp] ref row0", ref[0].tolist())
# single-sampler graph on the first group's logits for comparison (outside the CP loop)
with torch.inference_mode():
    lg = torch.randn(B, 1, Vc) * 3
    s1 = compile_fn()(lambda a, b, c, d: fu.sample_rows(a, b, c, d))(lg.to(dev), gs[:, 0].to(dev), st.to(dev), sk.to(dev)).cpu()
print(f"[cp] standalone sampler equal {bool((s1 == fu.sample_rows(lg, gs[:, 0], st, sk)).all())}")
