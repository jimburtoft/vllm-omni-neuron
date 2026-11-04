#!/usr/bin/env python3
"""Code-predictor attention formulations at batch B (T=1 steps, 17-slot KV):
  base : current (K cache [B,nkv,S,D], scores = q @ K^T, p @ V)
  kt   : K cache stored transposed [B,nkv,D,S] (no per-step K transpose)
  ew   : elementwise QK and PV (multiply + reduce over D / S), no small matmuls
  ewkt : elementwise with K transposed [B,nkv,D,S] (reduce over the partition-friendly axis)
Checks codes vs base for identical noise. usage: cp_attn_variants.py <model> B"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from qwen3_tts_neuron.engine import TTSEngine, rms
import qwen3_tts_neuron.fused as fu
from qwen3_tts_neuron.batched import _M
P, B = sys.argv[1], int(sys.argv[2])
dev = torch.device("neuron:0")
eng = TTSEngine(P, device=dev, max_ctx=256, dtype=torch.bfloat16, compile=False)
cp = eng.cp; cfg = cp.cfg; Vc = cfg.vocab; H = eng.talker.cfg.hidden

def layer(l, x, cos, sin, kc, vc, pos, m, var):
    q, k, v = l._qkv(x, cos, sin)              # q [B,nh,T,D], k/v [B,nkv,T,D]
    Bq, _, T, D = q.shape
    if var in ("kt", "ewkt"):
        kc = kc.index_copy(3, pos, k.transpose(-1, -2))   # [B,nkv,D,S]
    else:
        kc = kc.index_copy(2, pos, k)                      # [B,nkv,S,D]
    vc = vc.index_copy(2, pos, v)
    qg = q.reshape(Bq, l.nkv, l.rep * T, D)
    S = vc.shape[2]
    if var == "base":
        sc = (qg @ kc.transpose(-1, -2)).float()
    elif var == "kt":
        sc = (qg @ kc).float()
    elif var == "ew":
        sc = (qg.unsqueeze(-2).float() * kc.unsqueeze(2).float()).sum(-1)     # [B,nkv,rT,S]
    elif var == "ewkt":
        sc = (qg.unsqueeze(-1).float() * kc.unsqueeze(2).float()).sum(-2)     # [B,nkv,rT,S]
    sc = sc * l.scale
    sc = sc.view(Bq, l.nkv, l.rep, T, S) + m.view(1, 1, 1, T, S)
    p = torch.softmax(sc, -1).view(Bq, l.nkv, l.rep * T, S)
    if var in ("ew", "ewkt"):
        o = (p.unsqueeze(-1) * vc.unsqueeze(2).float()).sum(-2).to(q.dtype)   # [B,nkv,rT,D]
    else:
        o = p.to(q.dtype) @ vc
    o = o.view(Bq, l.nh, T, D).transpose(1, 2).reshape(Bq, T, l.nh * D)
    h = x + o @ l.w_o
    return h + l._mlp(h), kc, vc

def cp_frame(th, e0, gumbel, st, sk, var):
    L = len(cp.layers)
    kshape = (B, cfg.kv, cfg.head_dim, cp.S) if var in ("kt", "ewkt") else (B, cfg.kv, cp.S, cfg.head_dim)
    kc = [torch.zeros(*kshape, dtype=cp.dtype, device=th.device) for _ in range(L)]
    vc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=th.device) for _ in range(L)]
    j = torch.arange(cp.S, device=th.device)
    def run(x, pos):
        cos, sin = cp.cos.index_select(0, pos), cp.sin.index_select(0, pos)
        m = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32)
        h = x
        for i, l in enumerate(cp.layers):
            h, kc[i], vc[i] = layer(l, h, cos, sin, kc[i], vc[i], pos, m, var)
        return rms(h[:, -1:, :], cp.norm, cfg.eps)
    h = run(torch.cat([th, e0], 1) @ cp.proj_w + cp.proj_b, torch.arange(2, device=th.device))
    codes = []
    for g in range(cfg.groups - 1):
        c = fu.sample_rows((h @ cp.heads[g]).float(), gumbel[:, g], st, sk)
        codes.append(c)
        if g < cfg.groups - 2:
            h = run(F.embedding(c, cp.embeds[g]) @ cp.proj_w + cp.proj_b, torch.full((1,), g + 2, device=th.device, dtype=torch.long))
    return torch.cat(codes, -1)

torch.manual_seed(0)
h = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev); e0 = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev)
gs = (-torch.log(-torch.log(torch.rand(B, 15, 1, Vc).clamp(1e-10, 1 - 1e-7)))).to(dev)
st = torch.full((B,), 0.9, device=dev); sk = torch.full((B,), 50, dtype=torch.long, device=dev)
ref = None
for var in os.environ.get("ONLY", "base,kt,ew,ewkt").split(","):
    fn = compile_fn()(_M(lambda a, b, g, s, k, v=var: cp_frame(a, b, g, s, k, v), cp))
    with torch.inference_mode():
        for _ in range(3): o = fn(h, e0, gs, st, sk).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); o = fn(h, e0, gs, st, sk); o = o.cpu(); ts.append(time.perf_counter() - t0)
    eq = "" if ref is None else f" | codes equal to base {int((o == ref).sum())}/{o.numel()}"
    if var == "base": ref = o
    print(f"[cav] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} B={B} {var}: {1000 * sorted(ts)[10]:.2f} ms{eq}", flush=True)
