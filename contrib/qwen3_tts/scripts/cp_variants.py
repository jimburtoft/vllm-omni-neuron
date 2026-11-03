#!/usr/bin/env python3
"""Code-predictor frame variants at batch B, device time (current LNC):
  base    : current cp_frame_vec (matmul attention, 17-slot KV, sampler rows)
  kv16    : KV width trimmed to the used positions only (static per step: slot s+1)
  nosamp  : greedy argmax instead of the per-row top-k sampler (cost of sampling)
  noemb   : skip the embedding lookup feedback (upper bound on the head/embed cost)
usage: cp_variants.py <model> B"""
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
cp = eng.cp; cfg = cp.cfg; H = eng.talker.cfg.hidden
torch.manual_seed(0)
h = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev); e0 = (torch.randn(B, 1, H) * 0.5).to(torch.bfloat16).to(dev)
gs = (-torch.log(-torch.log(torch.rand(B, 15, 1, cfg.vocab).clamp(1e-10, 1 - 1e-7)))).to(dev)
st = torch.full((B,), 0.9, device=dev); sk = torch.full((B,), 50, dtype=torch.long, device=dev)

def cp_trim(cp, th, e0, gumbel, rows, sample=True, emb=True):
    """KV kept as a python list of per-position K/V (concatenated), attention over exactly s+1 positions."""
    L = len(cp.layers); dev_ = th.device
    Ks = [[] for _ in range(L)]; Vs = [[] for _ in range(L)]
    def run(x, pos0):
        T = x.shape[1]
        pos = torch.arange(pos0, pos0 + T, device=dev_)
        cos, sin = cp.cos.index_select(0, pos), cp.sin.index_select(0, pos)
        hh = x
        for i, l in enumerate(cp.layers):
            q, k, v = l._qkv(hh, cos, sin)
            Ks[i].append(k); Vs[i].append(v)
            K = torch.cat(Ks[i], 2); V = torch.cat(Vs[i], 2)
            S = K.shape[2]
            m = torch.where(torch.arange(S, device=dev_).view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32)
            o = l._attn(q, K, V, m)
            hh = hh + o @ l.w_o
            hh = hh + l._mlp(hh)
        return rms(hh[:, -1:, :], cp.norm, cfg.eps)
    x = run(torch.cat([th, e0], 1) @ cp.proj_w + cp.proj_b, 0)
    codes, es = [], torch.zeros_like(th)
    for g in range(cfg.groups - 1):
        lg = (x @ cp.heads[g]).float()
        c = fu.sample_rows(lg, gumbel[:, g], rows[0], rows[1]) if sample else lg.argmax(-1)
        codes.append(c)
        e = F.embedding(c, cp.embeds[g]) if emb else th
        es = es + e
        if g < cfg.groups - 2:
            x = run(e @ cp.proj_w + cp.proj_b, g + 2)
    return torch.cat(codes, -1), es

def sample_cand_k(logits, gumbel, temp, topk, kmax):
    """Same as fused._sample_rows_candidates but with a smaller candidate width kmax (topk <= kmax)."""
    B, V = logits.shape[0], logits.shape[-1]
    l0 = logits.reshape(B, V); g = gumbel.reshape(B, V); t = temp.view(B, 1)
    greedy = t <= 0
    inv_t = 1.0 / torch.where(greedy, 1.0, t); noise_on = torch.where(greedy, 0.0, 1.0)
    vals, idx = torch.topk(l0, kmax, dim=-1)
    k = topk.view(B, 1).clamp(min=0, max=kmax)
    pos = torch.arange(kmax, device=logits.device).view(1, kmax)
    score = torch.where(pos < k, vals * inv_t + noise_on * torch.gather(g, -1, idx), float("-inf"))
    j = torch.argmax(score, dim=-1, keepdim=True)
    return torch.gather(idx, -1, j).view(logits.shape[:-1])

def cp_with_sampler(sampler):
    def f(a, b, g, s, k):
        orig = fu.sample_rows
        fu.sample_rows = sampler
        try:
            return fu.cp_frame_vec(cp, a, b, False, g, rows=(s, k))[0]
        finally:
            fu.sample_rows = orig
    return f

variants = {
    "cand64": cp_with_sampler(lambda l, g, t, k: sample_cand_k(l, g, t, k, 64)),
    "cand50": cp_with_sampler(lambda l, g, t, k: sample_cand_k(l, g, t, k, 50)),
    "threshold": cp_with_sampler(fu._sample_rows_threshold),
    "base": lambda a, b, g, s, k: fu.cp_frame_vec(cp, a, b, False, g, rows=(s, k))[0],
    "trim": lambda a, b, g, s, k: cp_trim(cp, a, b, g, (s, k))[0],
    "trim_greedy": lambda a, b, g, s, k: cp_trim(cp, a, b, g, (s, k), sample=False)[0],
    "base_greedy_rows0": lambda a, b, g, s, k: fu.cp_frame_vec(cp, a, b, False, g, rows=(s * 0, k * 0))[0],
}
only = os.environ.get("ONLY")
ref = None
with torch.inference_mode():
    for name, f in variants.items():
        if only and name not in only.split(","):
            continue
        fn = compile_fn()(_M(f, cp))
        for _ in range(3): o = fn(h, e0, gs, st, sk).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); o = fn(h, e0, gs, st, sk); o = o.cpu(); ts.append(time.perf_counter() - t0)
        same = "" if ref is None else f" codes equal to base {int((o == ref).sum())}/{o.numel()}"
        if name == "base": ref = o
        print(f"[cpv] lnc={os.environ.get('NEURON_LOGICAL_NC_CONFIG')} B={B} {name}: {1000 * sorted(ts)[10]:.2f} ms{same}", flush=True)
