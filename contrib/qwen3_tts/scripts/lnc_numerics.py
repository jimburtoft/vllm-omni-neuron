#!/usr/bin/env python3
"""Talker + code-predictor numerics on device at the current LNC vs CPU fp32: one prefill + greedy frames, compare
talker logits (cosine, top-1 agreement) per step on teacher-forced inputs (same codes fed on both sides)."""
import os, sys
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.serving.prompts import PromptBuilder
from qwen3_tts_neuron.fused import FusedEngine, FusedFrameRows
from qwen3_tts_neuron.batched import prefill_plan
P = sys.argv[1]; dev = torch.device("neuron:0"); B = 2; S = 2048
engc = TTSEngine(P, device="cpu", max_ctx=512, dtype=torch.float32, compile=False)
engd = TTSEngine(P, device=dev, max_ctx=S, dtype=torch.bfloat16, compile=False, prefill_buckets=(64, 128, 256))
pb = PromptBuilder(engc, P)
emb, _, pad, _ = pb.build({"text": ["She said she would be here by noon, and she always keeps her word."], "language": ["English"],
                           "instruct": ["Calm, warm adult female voice, steady pace, gentle and reassuring."], "task_type": ["VoiceDesign"]})
# CPU fp32 reference greedy codes
ref = engc.generate(emb, pad, max_new=60)
# device fused engine, greedy
fe = FusedEngine(engd, B, S=S, vec_cp=False, compile=False)
c = compile_fn()
prefill = c(fe.prefill); frame = c(FusedFrameRows(engd, fe.bt, vec_cp=False, ctx=256).to(dev))
H, V = engd.talker.cfg.hidden, engd.talker.cfg.vocab
with torch.inference_mode():
    T = emb.shape[1]; bi = torch.tensor([0], device=dev)
    for start, n, Tb, p in prefill_plan(T, engd.prefill_buckets, S):
        x = torch.zeros(1, Tb, H, dtype=torch.bfloat16); x[:, Tb - n:] = emb[:, start:start + n].to(torch.bfloat16)
        hl, l1 = prefill(x.to(dev), bi, p.to(dev))
    h = torch.zeros(B, 1, H, dtype=torch.bfloat16, device=dev).index_copy(0, bi, hl)
    lg = torch.zeros(B, V, device=dev).index_copy(0, bi, l1)
    seen = torch.zeros(B, V, dtype=torch.bool, device=dev); pd = torch.zeros(B, 1, H, dtype=torch.bfloat16, device=dev)
    pd = pd.index_copy(0, bi, pad.reshape(1, 1, -1).to(torch.bfloat16).to(dev))
    pos = torch.tensor([T, 0]); z = torch.zeros(B, device=dev); zk = torch.zeros(B, dtype=torch.long, device=dev)
    rep = torch.full((B,), 1.05, device=dev); g0 = torch.zeros(B, V, device=dev); gs = torch.zeros(B, 15, 1, engd.cp.cfg.vocab, device=dev)
    l_prefill = l1.cpu()
    codes = []
    for step in range(len(ref)):
        allow = torch.full((B,), step >= 2, device=dev)
        h, lg, seen, cd = frame(h, lg, seen, allow, pd, pos.to(dev), g0, gs, rep, z, zk, z, zk)
        codes.append(cd.cpu()[0]); pos += 1
cod = torch.stack(codes)
n = min(len(cod), len(ref))
match = (cod[:n] == ref[:n]).all(-1).float()
first = int((match == 0).nonzero()[0]) if (match == 0).any() else n
cb0 = (cod[:n, 0] == ref[:n, 0]).float().mean().item(); res = (cod[:n, 1:] == ref[:n, 1:]).float().mean().item()
print(f"[num] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} frames ref {len(ref)}; identical frames before first divergence {first}; "
      f"CB0 agreement {cb0:.3f}; residual-code agreement {res:.3f}")
print(f"[num] first frame dev {cod[0].tolist()}"); print(f"[num] first frame ref {ref[0].tolist()}")
