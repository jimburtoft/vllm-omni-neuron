#!/usr/bin/env python3
"""Device A/B of one whole decoder block's 3 residual units (stock graph vs 3 NKI kernel calls), same input.
usage: c2w_block_ab.py <model_dir> <block> <T>"""
import os, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
import qwen3_tts_neuron.code2wav as cw
bi, T = int(sys.argv[2]), int(sys.argv[3])
dev = torch.device("neuron:0")
m = cw.Code2Wav(os.path.join(sys.argv[1], "speech_tokenizer"), dtype=torch.bfloat16).to(dev)
m.prepare_nki({bi})
Bk = m.blocks[bi]
C = Bk["r0w1"].shape[0]


WITH_CT = os.environ.get("WITH_CT", "0") == "1"  # include the block's snake + transposed conv before the units


def head(x):
    if not WITH_CT:
        return x
    x = cw._snake(x, Bk["a"], Bk["b"])
    return m._ctconv(x, Bk["tw"], Bk["tb"], m.rates[bi])


def stock(x):
    x = head(x)
    for ri, dil in enumerate((1, 3, 9)):
        res = x
        y = cw._snake(x, Bk[f"r{ri}a1"], Bk[f"r{ri}b1"])
        y = m._cconv(y, Bk[f"r{ri}w1"], Bk[f"r{ri}c1"], dilation=dil)
        y = cw._snake(y, Bk[f"r{ri}a2"], Bk[f"r{ri}b2"])
        y = m._cconv(y, Bk[f"r{ri}w2"], Bk[f"r{ri}c2"])
        x = res + y
    return x


def kern(x):
    if WITH_CT:
        x = m._nki_k_head[m._nki_grid](x.contiguous(), *[getattr(m, f"_nkih_{bi}_{k}") for k in ("wt", "bias", "a", "b")],
                                       m.rates[bi], int(os.environ.get("QTTS_C2W_NKI_TILEQ", "256")), m._snake_impl)
    for ri, dil in enumerate((1, 3, 9)):
        x = m._nki_unit(x, bi, ri, dil)
    return x


torch.manual_seed(0)
Cin = Bk["tw"].shape[0] if WITH_CT else C
Tin = T // m.rates[bi] if WITH_CT else T
x = (torch.randn(1, Cin, Tin) * 0.3).to(torch.bfloat16).to(dev)
c = compile_fn()
with torch.inference_mode():
    for name, f in (("stock", stock), ("nki", kern)):
        fn = c(f)
        for _ in range(3): y = fn(x).cpu()
        ts = []
        for _ in range(30):
            t0 = time.perf_counter(); y = fn(x).cpu(); ts.append(time.perf_counter() - t0)
        print(f"[block] {bi} C={C} T={T} with_ct={WITH_CT} {name}: {1000 * sorted(ts)[15]:.2f} ms", flush=True)
