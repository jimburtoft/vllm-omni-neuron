#!/usr/bin/env python3
"""Standalone probe of the served talker graphs (slot prefill + FusedFrameRows) at a given batch size.

usage: frame_probe.py <model_dir> [--B 32] [--S 2048] [--ctx 256] [--steps 200] [--device neuron:0|cpu]
Runs greedy decoding of 3 reference prompts replicated over B slots, through the same calls the AR worker makes,
and prints frames per row, per-row agreement with row 0 of the same prompt, and the frame time.
Env: QTTS_DTYPE (bfloat16 default on device), QTTS_EAGER=1 (no compile).
"""
import argparse, os, sys, time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--B", type=int, default=32); ap.add_argument("--S", type=int, default=2048)
ap.add_argument("--ctx", type=int, default=256); ap.add_argument("--steps", type=int, default=200)
ap.add_argument("--device", default="neuron:0"); ap.add_argument("--save", default=None)
a = ap.parse_args()
cpu = a.device == "cpu"
if cpu:
    os.environ["VLLM_NEURON_CPU_MODE"] = "1"
else:
    from qwen3_tts_neuron.serving.common import neuron_runtime_env
    neuron_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine
from qwen3_tts_neuron.fused import FusedEngine, FusedFrameRows
from qwen3_tts_neuron.batched import prefill_plan
from qwen3_tts_neuron.serving.common import compile_fn, model_dtype
from qwen3_tts_neuron.serving.prompts import PromptBuilder

dt = torch.float32 if cpu else model_dtype()
dev = torch.device(a.device)
eng = TTSEngine(a.model, device=dev, max_ctx=a.S, dtype=dt, compile=False, prefill_buckets=(64, 128, 256))
VEC = os.environ.get("QTTS_CP_VEC", "0") == "1"
fe = FusedEngine(eng, a.B, S=a.S, vec_cp=VEC, compile=False)
c = compile_fn()
prefill = c(fe.prefill)
frame = c(FusedFrameRows(eng, fe.bt, vec_cp=VEC, ctx=a.ctx).to(dev))
pb = PromptBuilder(eng, a.model)
CASES = [("en_calm", "She said she would be here by noon, and she always keeps her word.", "English",
          "Calm, warm adult female voice, steady pace, gentle and reassuring."),
         ("en_excited", "No way! Did you really win the whole tournament on your first try?", "English",
          "Energetic young male voice, bright timbre, fast and excited."),
         ("zh_soft", "哥哥，你回来啦，人家等了你好久好久了。", "Chinese", "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。")]
prompts = [pb.build({"text": [t], "language": [l], "instruct": [i], "task_type": ["VoiceDesign"]}) for _, t, l, i in CASES]
B, H, V = a.B, eng.talker.cfg.hidden, eng.talker.cfg.vocab
h = torch.zeros(B, 1, H, dtype=dt, device=dev); lg = torch.zeros(B, V, device=dev)
seen = torch.zeros(B, V, dtype=torch.bool, device=dev); pad = torch.zeros(B, 1, H, dtype=dt, device=dev)
pos = torch.zeros(B, dtype=torch.long)
with torch.inference_mode():
    for b in range(B):
        emb, _, p_, _ = prompts[b % 3]
        T = emb.shape[1]
        bi = torch.tensor([b], device=dev)
        for start, n, Tb, pp in prefill_plan(T, eng.prefill_buckets, a.S):
            x = torch.zeros(1, Tb, H, dtype=dt); x[:, Tb - n:] = emb[:, start:start + n]
            hl, l1 = prefill(x.to(dev), bi, pp.to(dev))
        h = h.index_copy(0, bi, hl); lg = lg.index_copy(0, bi, l1)
        pad = pad.index_copy(0, bi, p_.reshape(1, 1, -1).to(dt).to(dev)); pos[b] = T
    SAMP = os.environ.get("PROBE_SAMPLED", "0") == "1"
    MODE = os.environ.get("PROBE_MODE", "both" if SAMP else "greedy")  # both | talker_only | cp_only | greedy
    st_on = MODE in ("both", "talker_only"); sp_on = MODE in ("both", "cp_only")
    temp = torch.full((B,), 0.9 if st_on else 0.0, device=dev); topk = torch.full((B,), 50 if st_on else 0, dtype=torch.long, device=dev)
    stemp = torch.full((B,), 0.9 if sp_on else 0.0, device=dev); stopk = torch.full((B,), 50 if sp_on else 0, dtype=torch.long, device=dev)
    KONLY = os.environ.get("PROBE_KONLY")  # "temp": topk=0 with temperature; "topk": temperature tiny... (sub only)
    if KONLY == "temp_only":
        stopk = torch.zeros(B, dtype=torch.long, device=dev)
    if KONLY == "topk_greedyish":  # top-k on, temperature ~0 (argmax of masked logits)
        stemp = torch.full((B,), 1e-3, device=dev)
    if KONLY == "topk64":
        stopk = torch.full((B,), 64, dtype=torch.long, device=dev)
    if KONLY == "topk1":
        stopk = torch.full((B,), 1, dtype=torch.long, device=dev)
    z, zk = temp, topk
    rep = torch.full((B,), 1.05, device=dev)
    gen = torch.Generator().manual_seed(0)
    gum = lambda *sh: -torch.log(-torch.log(torch.rand(*sh, generator=gen).clamp(1e-10, 1 - 1e-7)))
    g0 = gum(B, V).to(dev) if SAMP else torch.zeros(B, V, device=dev)
    gs = gum(B, 15, 1, eng.cp.cfg.vocab).to(dev) if SAMP else torch.zeros(B, 15, 1, eng.cp.cfg.vocab, device=dev)
    codes_all, done, ts = [], torch.zeros(B, dtype=torch.bool), []
    for step in range(a.steps):
        allow = torch.full((B,), step >= 2, device=dev)
        t0 = time.perf_counter()
        h, lg, seen, codes = frame(h, lg, seen, allow, pad, pos.to(dev), g0, gs, rep, z, zk, stemp, stopk)
        codes = codes.cpu()
        ts.append(time.perf_counter() - t0)
        pos += 1
        codes_all.append(codes)
        done |= codes[:, 0] == eng.eos
        if bool(done.all()):
            break
C = torch.stack(codes_all, 1)  # [B, steps, 16]
frames = [int((C[b, :, 0] == eng.eos).nonzero()[0]) if (C[b, :, 0] == eng.eos).any() else C.shape[1] for b in range(B)]
K = min(3, B)
print("frames per prompt:", {CASES[k][0]: sorted(set(frames[k::3])) for k in range(K)})
for k in range(K):
    ref = C[k, :frames[k]]
    agree = [float((C[b, :frames[k]] == ref).all(-1).float().mean()) for b in range(k, B, 3)]
    print(CASES[k][0], "row agreement vs row", k, ":", [round(x, 3) for x in agree])
ts = sorted(ts[3:]) or ts
print("residual-zero fraction per row", [round(float((C[b, :frames[b], 1:] == 0).float().mean()), 3) for b in range(B)])
print(f"B={B} ctx={a.ctx} frame p50 {1000 * ts[len(ts) // 2]:.2f} ms")
if a.save:
    torch.save(dict(codes=C, frames=frames), a.save)
