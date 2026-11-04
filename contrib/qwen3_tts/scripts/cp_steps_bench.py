#!/usr/bin/env python3
"""Device: 15 consecutive code-predictor steps (positions 2..16) in ONE compiled graph -- torch layers vs NKI step
kernel -- without heads/sampling, to isolate the per-step cost inside a real multi-step graph.
usage: cp_steps_bench.py <model> B"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from qwen3_tts_neuron.engine import TTSEngine, rms
from qwen3_tts_neuron.kernels.cp_nki import cp_params, cp_step
from qwen3_tts_neuron.batched import _M
P, B = sys.argv[1], int(sys.argv[2])
dev = torch.device("neuron:0"); dt = torch.bfloat16
eng = TTSEngine(P, device=dev, max_ctx=64, dtype=dt, compile=False)
cp = eng.cp; cfg = cp.cfg; S = cp.S; L = len(cp.layers); lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
pp = {k: v.to(dev) for k, v in cp_params(cp).items()}
k = wrap_nki(cp_step)
x0 = (torch.randn(B, cfg.hidden) * 0.5).to(dt).to(dev)
def torch_steps(x):
    kc = [torch.zeros(B, cfg.kv, S, cfg.head_dim, dtype=dt, device=x.device) for _ in range(L)]
    vc = [torch.zeros_like(kc[0]) for _ in range(L)]
    j = torch.arange(S, device=x.device); h = x.view(B, 1, -1)
    for p in range(2, 17):
        pt = torch.full((1,), p, device=x.device, dtype=torch.long)
        cos, sin = cp.cos.index_select(0, pt), cp.sin.index_select(0, pt)
        m = torch.where(j.view(1, -1) <= pt.view(-1, 1), 0.0, -1e30).float()
        hh = h
        for i, l in enumerate(cp.layers):
            hh, kc[i], vc[i] = l.forward_local(hh, cos, sin, kc[i], vc[i], pt, m)
        h = rms(hh[:, -1:, :], cp.norm, cfg.eps) * 0.5 + h * 0.5
    return h
def nki_steps(x):
    kc = torch.zeros(L, B, cfg.kv, S, cfg.head_dim, dtype=dt, device=x.device); vc = torch.zeros_like(kc); h = x
    for p in range(2, 17):
        y = k[lnc](h.contiguous(), kc, vc, pp["wqkv"], pp["wo"], pp["wgu"], pp["wd"], pp["ln1"], pp["ln2"], pp["qn"],
                   pp["kn"], pp["norm"], pp["cos"], pp["sin"], p, cfg.eps)
        h = y * 0.5 + h * 0.5
    return h
def nki_mlp_only(x):
    """isolate: kernel with attention skipped (skip=1) -- cost of the projections + norms alone"""
    kc = torch.zeros(L, B, cfg.kv, S, cfg.head_dim, dtype=dt, device=x.device); vc = torch.zeros_like(kc); h = x
    for p in range(2, 17):
        y = k[lnc](h.contiguous(), kc, vc, pp["wqkv"], pp["wo"], pp["wgu"], pp["wd"], pp["ln1"], pp["ln2"], pp["qn"],
                   pp["kn"], pp["norm"], pp["cos"], pp["sin"], p, cfg.eps, skip=1)
        h = y * 0.5 + h * 0.5
    return h
def torch_noattn(x):
    import qwen3_tts_neuron.engine as en
    o = en.Qwen3Layer._attn
    en.Qwen3Layer._attn = lambda self, q, k_, v, m: q.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
    try:
        return torch_steps(x)
    finally:
        en.Qwen3Layer._attn = o
def nki_v(skip):
    def f(x):
        kc = torch.zeros(L, B, cfg.kv, S, cfg.head_dim, dtype=dt, device=x.device); vc = torch.zeros_like(kc); h = x
        for p in range(2, 17):
            y = k[lnc](h.contiguous(), kc, vc, pp["wqkv"], pp["wo"], pp["wgu"], pp["wd"], pp["ln1"], pp["ln2"], pp["qn"],
                       pp["kn"], pp["norm"], pp["cos"], pp["sin"], p, cfg.eps, skip=skip)
            h = y * 0.5 + h * 0.5
        return h
    return f
def nki_a(attn):
    def f(x):
        kc = torch.zeros(L, B, cfg.kv, S, cfg.head_dim, dtype=dt, device=x.device); vc = torch.zeros_like(kc); h = x
        for p in range(2, 17):
            y = k[lnc](h.contiguous(), kc, vc, pp["wqkv"], pp["wo"], pp["wgu"], pp["wd"], pp["ln1"], pp["ln2"], pp["qn"],
                       pp["kn"], pp["norm"], pp["cos"], pp["sin"], p, cfg.eps, attn=attn, wdge=int(os.environ.get("CPK_WDGE", "0")))
            h = y * 0.5 + h * 0.5
        return h
    return f
for name, f in ((f"nki wdge={os.environ.get('CPK_WDGE', '0')}", nki_a(1)),):
    fn = compile_fn()(_M(f, cp))
    with torch.inference_mode():
        for _ in range(3): fn(x0).cpu()
        ts = []
        for _ in range(15):
            t0 = time.perf_counter(); fn(x0).cpu(); ts.append(time.perf_counter() - t0)
    print(f"[css] lnc={lnc} B={B} 15 steps {name}: {1000 * sorted(ts)[7]:.2f} ms ({1000 * sorted(ts)[7] / 15:.2f} ms/step)", flush=True)
