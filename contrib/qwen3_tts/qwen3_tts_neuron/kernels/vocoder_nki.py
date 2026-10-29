"""NKI kernels for the Code2Wav (vocoder) decoder blocks.

`snake_resunit`: one fused decoder residual unit, the hot loop of the high-rate decoder blocks.

    y   = snake(x; a1, b1)                 # x + b * sin(a * x)^2, b = 1 / (exp(beta) + 1e-9), a = exp(alpha)
    y   = causal_conv1d(y, w1, c1, k=7, dilation=d)
    y   = snake(y; a2, b2)
    y   = conv1d(y, w2, c2, k=1)
    out = x + y

The reference graph runs this over [C, T] tensors (C = 96..768, T up to 52k samples per window) that do not fit SBUF,
so the compiler materialises every intermediate in HBM. This kernel streams time tiles with a 6*d-sample causal halo:
one HBM read of x and one write of out per unit, everything else stays in SBUF/PSUM.

Layout: channels on partitions (tiles of <= 128 channels: C = 96 -> 1, 192 -> 2, 384 -> 3, 768 -> 6), time
on the free dimension. Conv taps are 7 accumulating matmuls on shifted SBUF slices.

Precision: x / out / weights bf16; snake math in fp32; matmuls bf16 with fp32 PSUM accumulation. The reference
(`code2wav._snake` with SNAKE_FP32) computes snake in fp32 and the convs in bf16.

sin on the scalar engine accepts [-pi, pi], so the argument is range-reduced first: sin^2 has period pi, and
fmod(a*x, pi) lies in (-pi, pi).
"""
from __future__ import annotations

import math

import nki
import nki.isa as nisa
import nki.language as nl

PI = math.pi


def _snake_tile(dst, src, a_t, b_t, rows, cols, scratch_r, scratch_s):
    """dst[rows, cols] = src + b * sin(a * src)^2  (src fp32 or bf16 in SBUF; a/b fp32 [rows, 1])."""
    nisa.tensor_scalar(dst=scratch_r[0:rows, 0:cols], data=src, op0=nl.multiply, operand0=a_t[0:rows, 0:1],
                       op1=nl.fmod, operand1=PI)
    nisa.activation(dst=scratch_s[0:rows, 0:cols], op=nl.sin, data=scratch_r[0:rows, 0:cols])
    nisa.activation(dst=scratch_r[0:rows, 0:cols], op=nl.square, data=scratch_s[0:rows, 0:cols])
    nisa.scalar_tensor_tensor(dst=dst, data=scratch_r[0:rows, 0:cols], op0=nl.multiply, operand0=b_t[0:rows, 0:1],
                              op1=nl.add, operand1=src)


@nki.jit
def snake_resunit(x, w1t, c1, w2t, c2, a1, b1, a2, b2, dilation, tile_t=512):
    """One fused residual unit. C channels are split into channel tiles of <= 128 (C = 96 / 192 / 384 / 768).

    Args:
        x:   [B, C, T] bf16 (HBM)
        w1t: [7, C, C] bf16, w1t[k, cin, cout] = conv1.weight[cout, cin, k]
        c1:  [C, 1] fp32 conv1 bias
        w2t: [C, C] bf16, w2t[cin, cout] = conv2.weight[cout, cin, 0]
        c2:  [C, 1] fp32 conv2 bias
        a1, b1, a2, b2: [C, 1] fp32 snake params (a = exp(alpha), b = 1 / (exp(beta) + 1e-9))
        dilation: int (static)
    Returns:
        out: [B, C, T] bf16
    """
    B, C, T = x.shape
    K = 7
    d = dilation
    H = (K - 1) * d  # causal halo
    out = nl.ndarray((B, C, T), dtype=x.dtype, buffer=nl.shared_hbm)
    CT = 128 if C > 128 else C
    NC = (C + CT - 1) // CT

    # weights / params resident in SBUF, split by input-channel tile (partition dim = cin)
    w1_sb = nl.ndarray((CT, NC, K, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    w2_sb = nl.ndarray((CT, NC, C), dtype=nl.bfloat16, buffer=nl.sbuf)
    prm = nl.ndarray((CT, NC, 6), dtype=nl.float32, buffer=nl.sbuf)
    for ci in range(NC):
        cs = min(CT, C - ci * CT)
        for k in range(K):
            nisa.dma_copy(dst=w1_sb[0:cs, ci, k, 0:C], src=w1t[k, ci * CT:ci * CT + cs, 0:C])
        nisa.dma_copy(dst=w2_sb[0:cs, ci, 0:C], src=w2t[ci * CT:ci * CT + cs, 0:C])
        for i, p in enumerate((c1, c2, a1, b1, a2, b2)):
            nisa.dma_copy(dst=prm[0:cs, ci, i:i + 1], src=p[ci * CT:ci * CT + cs, 0:1])

    TT = tile_t
    n_tiles = (T + TT - 1) // TT
    W = H + TT
    for b in range(B):
        for ti in range(n_tiles):
            t0 = ti * TT
            n = min(TT, T - t0)
            lo = t0 - H
            z0 = -lo if lo < 0 else 0  # leading causal zero-pad columns
            xr = nl.ndarray((CT, NC, W), dtype=x.dtype, buffer=nl.sbuf)
            sn = nl.ndarray((CT, NC, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            sr = nl.ndarray((CT, W), dtype=nl.float32, buffer=nl.sbuf)
            ss = nl.ndarray((CT, W), dtype=nl.float32, buffer=nl.sbuf)
            # load x (+ halo) and apply snake 1, per input-channel tile
            for ci in range(NC):
                cs = min(CT, C - ci * CT)
                if z0 > 0:
                    nisa.memset(dst=xr[0:cs, ci, 0:z0], value=0.0)
                    nisa.memset(dst=sn[0:cs, ci, 0:z0], value=0.0)
                nisa.dma_copy(dst=xr[0:cs, ci, z0:H + n], src=x[b, ci * CT:ci * CT + cs, lo + z0:t0 + n])
                _snake_tile(sn[0:cs, ci, z0:H + n], xr[0:cs, ci, z0:H + n], prm[0:cs, ci, 2:3], prm[0:cs, ci, 3:4],
                            cs, H + n - z0, sr, ss)
            # dilated causal conv per output-channel tile: 7 taps x NC input tiles accumulate in PSUM
            sn2 = nl.ndarray((CT, NC, TT), dtype=nl.bfloat16, buffer=nl.sbuf)
            for co in range(NC):
                os_ = min(CT, C - co * CT)
                ps = nl.ndarray((CT, TT), dtype=nl.float32, buffer=nl.psum)
                for ci in range(NC):
                    cs = min(CT, C - ci * CT)
                    for k in range(K):
                        nisa.nc_matmul(dst=ps[0:os_, 0:n], stationary=w1_sb[0:cs, ci, k, co * CT:co * CT + os_],
                                       moving=sn[0:cs, ci, k * d:k * d + n])
                z = nl.ndarray((CT, TT), dtype=nl.float32, buffer=nl.sbuf)
                nisa.activation(dst=z[0:os_, 0:n], op=nl.copy, data=ps[0:os_, 0:n], bias=prm[0:os_, co, 0:1])
                _snake_tile(sn2[0:os_, co, 0:n], z[0:os_, 0:n], prm[0:os_, co, 4:5], prm[0:os_, co, 5:6], os_, n,
                            sr, ss)
            # 1x1 conv + bias + residual per output-channel tile
            for co in range(NC):
                os_ = min(CT, C - co * CT)
                ps2 = nl.ndarray((CT, TT), dtype=nl.float32, buffer=nl.psum)
                for ci in range(NC):
                    cs = min(CT, C - ci * CT)
                    nisa.nc_matmul(dst=ps2[0:os_, 0:n], stationary=w2_sb[0:cs, ci, co * CT:co * CT + os_],
                                   moving=sn2[0:cs, ci, 0:n])
                o = nl.ndarray((CT, TT), dtype=x.dtype, buffer=nl.sbuf)
                nisa.scalar_tensor_tensor(dst=o[0:os_, 0:n], data=ps2[0:os_, 0:n], op0=nl.add,
                                          operand0=prm[0:os_, co, 1:2], op1=nl.add, operand1=xr[0:os_, co, H:H + n])
                nisa.dma_copy(dst=out[b, co * CT:co * CT + os_, t0:t0 + n], src=o[0:os_, 0:n])
    return out


def resunit_params(Bk, ri):
    """Host-side packing of one residual unit's parameters from a Code2Wav block dict (torch tensors)."""
    import torch
    w1 = Bk[f"r{ri}w1"]  # [Cout, Cin, 7]
    w2 = Bk[f"r{ri}w2"]  # [Cout, Cin, 1]
    f32 = lambda t: t.detach().float().reshape(-1, 1).contiguous()
    return dict(
        w1t=w1.detach().permute(2, 1, 0).contiguous().to(torch.bfloat16),
        c1=f32(Bk[f"r{ri}c1"]),
        w2t=w2.detach()[:, :, 0].t().contiguous().to(torch.bfloat16),
        c2=f32(Bk[f"r{ri}c2"]),
        a1=f32(Bk[f"r{ri}a1"]), b1=f32(Bk[f"r{ri}b1"]), a2=f32(Bk[f"r{ri}a2"]), b2=f32(Bk[f"r{ri}b2"]),
    )
