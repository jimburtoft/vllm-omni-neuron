#!/usr/bin/env python3
"""Device: one code-predictor step (5 layers, T=1) torch vs cp_step NKI kernel, at batch B and position pos.
usage: cp_kernel_bench.py <model> B pos"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
from qwen3_tts_neuron.engine import TTSEngine, rms
from qwen3_tts_neuron.kernels.cp_nki import cp_params, cp_step
from qwen3_tts_neuron.batched import _M
P, B, pos = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("neuron:0"); dt = torch.bfloat16
eng = TTSEngine(P, device=dev, max_ctx=64, dtype=dt, compile=False)
cp = eng.cp; cfg = cp.cfg; S = cp.S; L = len(cp.layers)
lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
torch.manual_seed(0)
x = (torch.randn(B, 1, cfg.hidden) * 0.5).to(dt).to(dev)
kc0 = [(torch.randn(B, cfg.kv, S, cfg.head_dim) * 0.3).to(dt) for _ in range(L)]
vc0 = [(torch.randn(B, cfg.kv, S, cfg.head_dim) * 0.3).to(dt) for _ in range(L)]
kc = [k.to(dev) for k in kc0]; vc = [v.to(dev) for v in vc0]
kcl = torch.stack([k.permute(0, 2, 1, 3).reshape(B, S, -1) for k in kc0]).contiguous().to(dev)
vcl = torch.stack([v.permute(0, 2, 1, 3).reshape(B, S, -1) for v in vc0]).contiguous().to(dev)
pp = {k: v.to(dev) for k, v in cp_params(cp).items()}
pt = torch.tensor([pos], device=dev)
def torch_step(x, *kv):
    cos, sin = cp.cos.index_select(0, pt), cp.sin.index_select(0, pt)
    m = torch.where(torch.arange(S, device=x.device).view(1, -1) <= pt.view(-1, 1), 0.0, -1e30).float()
    h = x
    for i, l in enumerate(cp.layers):
        h, _, _ = l.forward_local(h, cos, sin, kv[2 * i], kv[2 * i + 1], pt, m)
    return rms(h[:, -1:, :], cp.norm, cfg.eps)[:, 0]
k = wrap_nki(cp_step)
def kern_step(x, kcl, vcl):
    return k[lnc](x.view(B, -1), kcl, vcl, pp["wqkv"], pp["wo"], pp["wgu"], pp["wd"], pp["ln1"], pp["ln2"], pp["qn"],
                  pp["kn"], pp["norm"], pp["cos"], pp["sin"], pos, cfg.eps)
def bench(name, fn, *args, n=20):
    with torch.inference_mode():
        for _ in range(3): o = fn(*args).cpu()
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); o = fn(*args); o = o.cpu(); ts.append(time.perf_counter() - t0)
    print(f"[ckb] lnc={lnc} B={B} pos={pos} {name}: {1000 * sorted(ts)[n // 2]:.2f} ms", flush=True)
    return o
kvs = [z for p in zip(kc, vc) for z in p]
for lay in ("qkv", "attn", "mlp"):
    pass
a = bench("torch step", compile_fn()(_M(torch_step, cp)), x, *kvs)
b = bench("nki step", compile_fn()(_M(kern_step, cp)), x, kcl, vcl)
print(f"[ckb] min cosine torch vs kernel: {F.cosine_similarity(a.float(), b.float(), dim=-1).min().item():.5f}")
