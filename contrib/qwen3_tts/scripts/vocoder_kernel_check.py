#!/usr/bin/env python3
"""Simulator / device check of the fused vocoder residual-unit kernel against the fp32 torch reference.

usage: vocoder_kernel_check.py <model_dir> [--device sim|neuron] [--block 3] [--T 4096]
sim: nki.simulate on numpy inputs (CPU). neuron: wrap_nki under torch.compile on neuron:0 (lite backend).
"""
import argparse, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("--device", default="sim"); ap.add_argument("--block", type=int, default=3)
ap.add_argument("--T", type=int, default=4096); ap.add_argument("--B", type=int, default=1)
ap.add_argument("--units", default="0,1,2"); ap.add_argument("--tile", type=int, default=512)
a = ap.parse_args()

from qwen3_tts_neuron.code2wav import Code2Wav, _snake
from qwen3_tts_neuron.kernels.vocoder_nki import resunit_params, snake_resunit

m = Code2Wav(os.path.join(a.model, "speech_tokenizer"), dtype=torch.float32)
Bk = m.blocks[a.block]
C = Bk["r0w1"].shape[0]
torch.manual_seed(0)
x = (torch.randn(a.B, C, a.T) * 0.3).to(torch.bfloat16)


def ref_unit(x, ri, dil):
    xf = x.float()
    y = _snake(xf, Bk[f"r{ri}a1"], Bk[f"r{ri}b1"])
    y = F.conv1d(F.pad(y, ((7 - 1) * dil, 0)), Bk[f"r{ri}w1"], Bk[f"r{ri}c1"], dilation=dil)
    y = _snake(y, Bk[f"r{ri}a2"], Bk[f"r{ri}b2"])
    y = F.conv1d(y, Bk[f"r{ri}w2"], Bk[f"r{ri}c2"])
    return xf + y


def stats(o, r):
    o, r = o.float(), r.float()
    err = (o - r)
    snr = 10 * torch.log10(r.pow(2).sum() / err.pow(2).sum()).item()
    cos = F.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
    return f"maxabs {err.abs().max().item():.3e} snr {snr:.1f} dB cos {cos:.6f}"


for ri in [int(u) for u in a.units.split(",")]:
    dil = (1, 3, 9)[ri]
    p = resunit_params(Bk, ri)
    r = ref_unit(x, ri, dil)
    # bf16-convs reference (what the stock bf16 graph does): snake fp32, convs in bf16
    xb = x.float()
    y = _snake(xb, Bk[f"r{ri}a1"], Bk[f"r{ri}b1"]).to(torch.bfloat16)
    y = F.conv1d(F.pad(y, ((7 - 1) * dil, 0)), Bk[f"r{ri}w1"].to(torch.bfloat16), Bk[f"r{ri}c1"].to(torch.bfloat16),
                 dilation=dil)
    y = _snake(y.float(), Bk[f"r{ri}a2"], Bk[f"r{ri}b2"]).to(torch.bfloat16)
    y = F.conv1d(y, Bk[f"r{ri}w2"].to(torch.bfloat16), Bk[f"r{ri}c2"].to(torch.bfloat16))
    rb = (xb + y.float()).to(torch.bfloat16)
    if a.device == "sim":
        import nki
        args = [x.float().numpy().astype(np.float32)]  # simulator: pass fp32 views of the bf16 data
        try:
            import ml_dtypes
            bf = ml_dtypes.bfloat16
            args = [x.float().numpy().astype(bf)]
            cv = lambda t: t.float().numpy().astype(bf) if t.dtype == torch.bfloat16 else t.numpy()
        except ImportError:
            cv = lambda t: t.float().numpy()
        t0 = time.time()
        out = nki.simulate(snake_resunit)(args[0], cv(p["w1t"]), cv(p["c1"]), cv(p["w2t"]), cv(p["c2"]), cv(p["a1"]),
                                           cv(p["b1"]), cv(p["a2"]), cv(p["b2"]), dil, a.tile)
        o = torch.from_numpy(np.asarray(out).astype(np.float32))
        print(f"[sim] block {a.block} unit {ri} dil {dil} C {C} T {a.T}: kernel vs fp32 {stats(o, r)} | "
              f"stock-bf16 vs fp32 {stats(rb, r)} | {time.time() - t0:.1f}s", flush=True)
    else:
        from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
        neuron_runtime_env()
        import libtorch_neuronx_lite  # noqa: F401
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        dev = torch.device("neuron:0")
        k = wrap_nki(snake_resunit)
        lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
        dv = {kk: v.to(dev) for kk, v in p.items()}
        fn = compile_fn()(lambda xx: k[lnc](xx, dv["w1t"], dv["c1"], dv["w2t"], dv["c2"], dv["a1"], dv["b1"], dv["a2"],
                                            dv["b2"], dil, a.tile))
        xd = x.to(dev)
        with torch.inference_mode():
            o = fn(xd).cpu()
            ts = []
            for _ in range(20):
                t0 = time.perf_counter(); fn(xd).cpu(); ts.append(time.perf_counter() - t0)
        # stock graph for the same unit (bf16 convs, fp32 snake), same device
        Bd = {kk: v.to(dev) for kk, v in Bk.items()}
        def stock(xx):
            xf = xx.to(torch.bfloat16)
            y = _snake(xf, Bd[f"r{ri}a1"].to(torch.bfloat16), Bd[f"r{ri}b1"].to(torch.bfloat16))
            y = F.conv1d(F.pad(y, ((7 - 1) * dil, 0)), Bd[f"r{ri}w1"].to(torch.bfloat16), Bd[f"r{ri}c1"].to(torch.bfloat16), dilation=dil)
            y = _snake(y, Bd[f"r{ri}a2"].to(torch.bfloat16), Bd[f"r{ri}b2"].to(torch.bfloat16))
            y = F.conv1d(y, Bd[f"r{ri}w2"].to(torch.bfloat16), Bd[f"r{ri}c2"].to(torch.bfloat16))
            return xf + y
        sfn = compile_fn()(stock)
        with torch.inference_mode():
            so = sfn(xd).cpu()
            ts2 = []
            for _ in range(20):
                t0 = time.perf_counter(); sfn(xd).cpu(); ts2.append(time.perf_counter() - t0)
        print(f"[neuron] stock graph same unit: vs fp32 {stats(so, r)} | {1000 * sorted(ts2)[10]:.2f} ms", flush=True)
        print(f"[neuron] block {a.block} unit {ri} dil {dil} C {C} T {a.T}: kernel vs fp32 {stats(o, r)} | "
              f"stock-bf16 vs fp32 {stats(rb, r)} | {1000 * sorted(ts)[10]:.2f} ms", flush=True)
