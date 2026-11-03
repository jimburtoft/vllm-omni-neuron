#!/usr/bin/env python3
"""Talker decode attention: torch (current) vs vllm-neuron NF.attention_decode (fused NKI) for one layer and for the
28-layer attention-only stack, at batch B and ctx S. Also checks numerics vs the torch path.
usage: attn_kernel_probe.py <model> B ctx"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
import vllm_neuron.functional as NF
from qwen3_tts_neuron.engine import TTSEngine, rms, rotate_half
from qwen3_tts_neuron.batched import _qkv_b, _attn_b, _M
P, B, S = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("neuron:0"); dt = torch.bfloat16
eng = TTSEngine(P, device=dev, max_ctx=S, dtype=dt, compile=False)
t = eng.talker; c = t.cfg
nkv, D, nh, Hd = c.kv, c.head_dim, c.heads, c.hidden
L = int(os.environ.get("NLAYERS", "1"))
BLK = int(os.environ.get("KV_BLOCK", "128")); NB = S // BLK
torch.manual_seed(0)
x = (torch.randn(B, 1, Hd) * 0.5).to(dt).to(dev)
pos = torch.randint(10, S - 2, (B,)).to(dev)
kc = [(torch.randn(B * nkv * S, D) * 0.3).to(dt).to(dev) for _ in range(L)]
vc = [(torch.randn(B * nkv * S, D) * 0.3).to(dt).to(dev) for _ in range(L)]
def torch_path(x, pos, *kv):
    j = torch.arange(S, device=x.device)
    mask = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32).view(B, 1, S)
    cos = t.cos.index_select(0, pos).view(B, 1, -1); sin = t.sin.index_select(0, pos).view(B, 1, -1)
    h = x
    for i in range(L):
        l = t.layers[i]
        q, k, v = _qkv_b(l, h, cos, sin)
        K = kv[2 * i].view(B, nkv, S, D); V = kv[2 * i + 1].view(B, nkv, S, D)
        h = h + _attn_b(l, q, K, V, mask) @ l.w_o
    return h
def kernel_path(x, pos, *kv):
    half = D // 2
    cos = t.cos.index_select(0, pos)[:, :half].view(B, 1, half).permute(2, 0, 1).contiguous().to(dt)
    sin = t.sin.index_select(0, pos)[:, :half].view(B, 1, half).permute(2, 0, 1).contiguous().to(dt)
    mask = NF.gen_attention_decode_mask(pos_ids=pos.view(1, B).to(torch.float32), bs=B, q_head=nh // nkv * 0 + nh, s_active=1,
                                        s_prior=S, start_pos=None, block_len=BLK)
    # block table: row b, head h -> block ids ((b*nkv + h)*NB + j) in a [B*nkv*NB, BLK, D] cache view
    bt = ((torch.arange(B, device=x.device).view(B, 1, 1) * nkv + torch.arange(nkv, device=x.device).view(1, nkv, 1)) * NB
          + torch.arange(NB, device=x.device).view(1, 1, NB)).to(torch.int32)
    h = x
    for i in range(L):
        l = t.layers[i]
        out, Kn, Vn = NF.attention_decode(
            X=h, X_hidden_dim_actual=Hd, rmsnorm_X_enabled=True, rmsnorm_X_eps=l.eps, rmsnorm_X_gamma=l.ln1.view(1, -1),
            W_qkv=l.w_qkv, bias_qkv=None,
            rmsnorm_QK_pre_rope_enabled=True, rmsnorm_QK_pre_rope_eps=l.eps,
            rmsnorm_QK_pre_rope_W_Q=l.qn.view(1, -1), rmsnorm_QK_pre_rope_W_K=l.kn.view(1, -1),
            rmsnorm_QK_post_rope_enabled=False, cos=cos, sin=sin, rope_contiguous_layout=True,
            K_cache_transposed=False, active_blocks_table=bt,
            K_cache=kv[2 * i].view(B * nkv * NB, BLK, D).view(B * NB, nkv, BLK, D) if False else kv[2 * i].view(B, nkv, NB, BLK, D).permute(0, 2, 1, 3, 4).reshape(B * NB, nkv, BLK, D),
            V_cache=kv[2 * i + 1].view(B, nkv, NB, BLK, D).permute(0, 2, 1, 3, 4).reshape(B * NB, nkv, BLK, D),
            attention_mask=mask, softmax_scale=D ** -0.5, sink=None, update_cache=False,
            W_out=l.w_o, bias_out=None, transposed_out=False, out_in_sb=False)
        h = h + out.view(B, 1, Hd)
    return h
def bench(name, f, *a, n=20):
    fn = compile_fn()(_M(f, t))
    with torch.inference_mode():
        for _ in range(3): o = fn(*a).cpu()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); o = fn(*a); o = o.cpu(); ts.append(time.perf_counter() - t0)
    print(f"[ak] {name} B={B} S={S} layers={L}: {1000 * sorted(ts)[n // 2]:.2f} ms", flush=True)
    return o
kvs = [z for p in zip(kc, vc) for z in p]
a = bench("torch", torch_path, x, pos, *kvs)
try:
    b = bench("nki attention_decode", kernel_path, x, pos, *kvs)
    cos_ = F.cosine_similarity(a.float().view(B, -1), b.float().view(B, -1), dim=-1).min().item()
    print(f"[ak] min cosine torch vs kernel: {cos_:.5f}", flush=True)
except Exception as e:
    print("[ak] kernel failed:", repr(e)[:600], flush=True)
