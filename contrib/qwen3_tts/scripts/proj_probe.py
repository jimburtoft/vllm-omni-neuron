#!/usr/bin/env python3
"""Device: one [B,1024] x [1024,6144] projection as an NKI kernel, two layouts (weights moving vs stationary)."""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
import nki, nki.isa as nisa, nki.language as nl
from qwen3_tts_neuron.kernels.cp_nki import _proj, _proj_ws, _transpose_to_T, P
B = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 6144; K = 1024
@nki.jit
def k_mov(x, w, ver=1):
    o = nl.ndarray((B, N), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((B, K), dtype=nl.bfloat16, buffer=nl.sbuf); nisa.dma_copy(dst=xs, src=x[0:B, 0:K])
    xT = nl.ndarray((P, K // P, B), dtype=nl.bfloat16, buffer=nl.sbuf); _transpose_to_T(xT, xs, B, K // P)
    out = nl.ndarray((B, N), dtype=nl.float32, buffer=nl.sbuf); _proj(out, xT, w, 0, B, N)
    nisa.dma_copy(dst=o[0:B, 0:N], src=out); return o
@nki.jit
def k_ws(x, w, ver=1):
    o = nl.ndarray((P, N // P, B), dtype=nl.float32, buffer=nl.shared_hbm)
    xs = nl.ndarray((B, K), dtype=nl.bfloat16, buffer=nl.sbuf); nisa.dma_copy(dst=xs, src=x[0:B, 0:K])
    xT = nl.ndarray((P, K // P, B), dtype=nl.bfloat16, buffer=nl.sbuf); _transpose_to_T(xT, xs, B, K // P)
    outT = nl.ndarray((P, N // P, B), dtype=nl.float32, buffer=nl.sbuf); _proj_ws(outT, xT, w, 0, B, N)
    nisa.dma_copy(dst=o, src=outT); return o
dev = torch.device("neuron:0"); lnc = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
x = (torch.randn(B, K) * 0.3).to(torch.bfloat16).to(dev); w = (torch.randn(1, K, N) * 0.03).to(torch.bfloat16).to(dev)
ref = (x.cpu().float() @ w.cpu().float()[0])
@nki.jit
def k_dma(x, w, ver=1):
    """weight bytes only: DMA the whole [K, N] weight into SBUF blocks, no compute."""
    o = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    wv = w[0].reshape((K // P, P, N)).rearrange(("c", "p", "n"), ("p", "c", "n"))
    acc = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=acc, value=0.0)
    for ob in range(N // 512):
        wt = nl.ndarray((P, K // P, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=wv[:, :, ob * 512:(ob + 1) * 512])
        nisa.tensor_tensor(dst=acc[0:B, 0:1], data1=acc[0:B, 0:1], data2=wt[0:B, 0, 0:1], op=nl.add)
    nisa.dma_copy(dst=o[0:B, 0:128], src=acc); return o
@nki.jit
def k_dma_q(x, w, ver=1):
    """DMA spread over 4 queues (sync / scalar / gpsimd / default): weight bytes only."""
    o = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    wv = w[0].reshape((K // P, P, N)).rearrange(("c", "p", "n"), ("p", "c", "n"))
    acc = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=acc, value=0.0)
    engs = [nisa.sync_engine, nisa.scalar_engine, nisa.gpsimd_engine, nisa.vector_engine]
    for ob in range(N // 512):
        wt = nl.ndarray((P, K // P, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=wv[:, :, ob * 512:(ob + 1) * 512], engine=engs[ob % 4])
        nisa.tensor_tensor(dst=acc[0:B, 0:1], data1=acc[0:B, 0:1], data2=wt[0:B, 0, 0:1], op=nl.add)
    nisa.dma_copy(dst=o[0:B, 0:128], src=acc); return o
@nki.jit
def k_dma_big(x, w, ver=1):
    """weight bytes only, 2 MB per DMA (2048 columns)."""
    o = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    wv = w[0].reshape((K // P, P, N)).rearrange(("c", "p", "n"), ("p", "c", "n"))
    acc = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=acc, value=0.0)
    CW = 2048
    for ob in range(N // CW):
        wt = nl.ndarray((P, K // P, CW), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=wv[:, :, ob * CW:(ob + 1) * CW])
        nisa.tensor_tensor(dst=acc[0:B, 0:1], data1=acc[0:B, 0:1], data2=wt[0:B, 0, 0:1], op=nl.add)
    nisa.dma_copy(dst=o[0:B, 0:128], src=acc); return o
@nki.jit
def k_dma_rows(x, w, ver=1):
    """weight bytes only, contiguous rows: [128 partitions, 8 rows of N each] per DMA chunk of 512 columns? no: whole
    rows -> each partition reads K/128 full rows (contiguous N*2 bytes)."""
    o = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.shared_hbm)
    acc = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=acc, value=0.0)
    RW = 4096
    for ob in range(N // RW):
        wt = nl.ndarray((P, K // P, RW), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=w[0].reshape((P, K // P, N))[:, :, ob * RW:(ob + 1) * RW])
        nisa.tensor_tensor(dst=acc[0:B, 0:1], data1=acc[0:B, 0:1], data2=wt[0:B, 0, 0:1], op=nl.add)
    nisa.dma_copy(dst=o[0:B, 0:128], src=acc); return o
def mk(CW):
    @nki.jit
    def k(x, w, ver=CW):
        o = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.shared_hbm)
        wv = w[0].reshape((K // P, P, N)).rearrange(("c", "p", "n"), ("p", "c", "n"))
        acc = nl.ndarray((B, 128), dtype=nl.float32, buffer=nl.sbuf); nisa.memset(dst=acc, value=0.0)
        for ob in range(N // CW):
            wt = nl.ndarray((P, K // P, CW), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt, src=wv[:, :, ob * CW:(ob + 1) * CW])
            nisa.tensor_tensor(dst=acc[0:B, 0:1], data1=acc[0:B, 0:1], data2=wt[0:B, 0, 0:1], op=nl.add)
        nisa.dma_copy(dst=o[0:B, 0:128], src=acc); return o
    return k
for name, kk in [(f"dma {cw} cols ({cw * 2 * 1024 // 1024} KB)", mk(cw)) for cw in (1024, 2048, 4096, 8192)]:
    wk = wrap_nki(kk)
    fn = compile_fn()(lambda a, b, wk=wk: wk[lnc](a, b))
    with torch.inference_mode():
        for _ in range(3): o = fn(x, w).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); o = fn(x, w).cpu(); ts.append(time.perf_counter() - t0)
    if name.startswith("dma"):
        print(f"[pj] B={B} N={N} {name}: {1000 * sorted(ts)[10]:.3f} ms ({K * N * 2 / (sorted(ts)[10]) / 1e9:.0f} GB/s incl. launch)", flush=True); continue
    o = o if o.dim() == 2 else o.permute(1, 0, 2).reshape(N, B).t()
    print(f"[pj] B={B} N={N} {name}: {1000 * sorted(ts)[10]:.3f} ms maxerr {(o.float() - ref).abs().max().item():.3e}", flush=True)
mm = compile_fn()(lambda a, b: a @ b[0])
with torch.inference_mode():
    for _ in range(3): mm(x, w).cpu()
    ts = []
    for _ in range(20):
        t0 = time.perf_counter(); mm(x, w).cpu(); ts.append(time.perf_counter() - t0)
print(f"[pj] B={B} N={N} torch matmul: {1000 * sorted(ts)[10]:.3f} ms", flush=True)
