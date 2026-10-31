#!/usr/bin/env python3
"""Inside the code-predictor loop at the current LNC: dump the k-th threshold, kept count and sampled index of group 0..2
(device vs CPU bf16) to find which piece goes wrong."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
import qwen3_tts_neuron.fused as fu
from qwen3_tts_neuron.batched import _M
B = 4
dev = torch.device("neuron:0")
engd = TTSEngine(sys.argv[1], device=dev, max_ctx=256, dtype=torch.bfloat16, compile=False)
engc = TTSEngine(sys.argv[1], device="cpu", max_ctx=256, dtype=torch.bfloat16, compile=False)
H, Vc = engc.talker.cfg.hidden, engc.cp.cfg.vocab
torch.manual_seed(0)
lgs = torch.randn(3, B, 1, Vc) * 3
gs = -torch.log(-torch.log(torch.rand(3, B, 1, Vc).clamp(1e-10, 1 - 1e-7)))
st = torch.full((B,), 0.9); sk = torch.full((B,), 50, dtype=torch.long)
def body(lgs, gs, st, sk, emb):
    outs = []
    acc = torch.zeros(B, 1, emb.shape[-1], dtype=emb.dtype, device=lgs.device)
    for g in range(3):
        lg = lgs[g] + (acc.float().sum(-1, keepdim=True) * 0.0)  # dependency chain like the CP loop
        shape = (B, 1, 1)
        vals = torch.topk(lg, 64, dim=-1).values
        k = sk.view(shape).clamp(0, 64)
        sel = torch.where(k > 0, k, 64) - 1
        oh = (torch.arange(64, device=lg.device).view(1, 1, 64) == sel)
        kth = (vals * oh.to(vals.dtype)).sum(-1, keepdim=True)
        keep = (lg >= kth).sum(-1)
        outs += [lg.max(-1).values.view(B), (lg - kth).max(-1).values.view(B)]
        c = fu.sample_rows(lg, gs[g], st, sk)
        outs += [kth.view(B), keep.view(B).float(), c.view(B).float()]
        acc = acc + F.embedding(c, emb)
    return torch.stack(outs)
emb_c = engc.cp.embeds[0]; emb_d = engd.cp.embeds[0]
ref = body(lgs, gs, st, sk, emb_c)
with torch.inference_mode():
    out = compile_fn()(_M(lambda a, b, c, d: body(a, b, c, d, emb_d), engd.cp))(lgs.to(dev), gs.to(dev), st.to(dev), sk.to(dev)).cpu()
for g in range(3):
    o = out[5*g:5*g+5]; r = ref[5*g:5*g+5]
    print(f"[cp2] g{g} lg.max dev {[round(x,3) for x in o[0].tolist()]} ref {[round(x,3) for x in r[0].tolist()]} | max(lg-kth) dev {[round(x,3) for x in o[1].tolist()]} ref {[round(x,3) for x in r[1].tolist()]}")
    print(f"[cp2]     kth dev {[round(x,3) for x in o[2].tolist()]} kept dev {o[3].tolist()} ref {r[3].tolist()} | code dev {o[4].tolist()} ref {r[4].tolist()}")
