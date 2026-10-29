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

sin on the scalar engine accepts [-pi, pi], so the argument is range-reduced first (sin^2 has period pi; see
_snake_tile). Snake params are passed as a/pi.
"""
from __future__ import annotations

import math

import nki
import nki.isa as nisa
import nki.language as nl

PI = math.pi


import os as _os

# NOTE: libtorch-neuronx-lite keys its NKI cache on the TOP-LEVEL kernel source + args only. Variants that live in
# helpers must be selected through a kernel argument (snake_impl), never a module global, or a stale NEFF is reused.
SNAKE_IMPL = int(_os.environ.get("QTTS_C2W_SNAKE_IMPL", "1"))  # 1 = 6-instruction snake (default), 0 = 7
ROUND_C = 12582912.0  # 1.5 * 2**23: (v + C) - C rounds an fp32 |v| < 2**22 to the nearest integer


def _snake_tile(dst, src, ap_t, b_t, rows, cols, s_v, s_k, impl=0):
    """dst[rows, cols] = src + b * sin(a * src)^2  (src fp32 or bf16 in SBUF; ap = a / pi, b fp32 [rows, 1]).

    sin on the scalar engine accepts [-pi, pi] only, and the ALU has no mod/floor, so range-reduce with the fp32
    rounding trick: v = a*x/pi, k = round(v), sin^2(a*x) = sin^2(pi * (v - k)) with pi*(v-k) in [-pi/2, pi/2].
    """
    if impl == 1:
        # 4 instructions: v = a*x/pi (DVE) | k = round(v) in 1 two-op tensor_scalar (Pool) | sin(pi*(v-k)) via
        # scalar_tensor_tensor (DVE: v - k, then ACT sin+scale) | square + b*() + x fused in one STT
        nisa.tensor_scalar(dst=s_v[0:rows, 0:cols], data=src, op0=nl.multiply, operand0=ap_t)
        nisa.tensor_scalar(dst=s_k[0:rows, 0:cols], data=s_v[0:rows, 0:cols], op0=nl.add, operand0=ROUND_C,
                           op1=nl.subtract, operand1=ROUND_C)
        nisa.tensor_tensor(dst=s_v[0:rows, 0:cols], data1=s_v[0:rows, 0:cols], data2=s_k[0:rows, 0:cols],
                           op=nl.subtract)
        nisa.activation(dst=s_k[0:rows, 0:cols], op=nl.sin, data=s_v[0:rows, 0:cols], scale=PI)
        # (sin * sin) * b + x : scalar_tensor_tensor computes (data op0 operand0) op1 operand1
        nisa.tensor_tensor(dst=s_v[0:rows, 0:cols], data1=s_k[0:rows, 0:cols], data2=s_k[0:rows, 0:cols],
                           op=nl.multiply)
        nisa.scalar_tensor_tensor(dst=dst, data=s_v[0:rows, 0:cols], op0=nl.multiply, operand0=b_t,
                                  op1=nl.add, operand1=src)
        return
    nisa.tensor_scalar(dst=s_v[0:rows, 0:cols], data=src, op0=nl.multiply, operand0=ap_t)            # v
    nisa.tensor_scalar(dst=s_k[0:rows, 0:cols], data=s_v[0:rows, 0:cols], op0=nl.add, operand0=ROUND_C)
    nisa.tensor_scalar(dst=s_k[0:rows, 0:cols], data=s_k[0:rows, 0:cols], op0=nl.subtract, operand0=ROUND_C)  # k
    nisa.tensor_tensor(dst=s_v[0:rows, 0:cols], data1=s_v[0:rows, 0:cols], data2=s_k[0:rows, 0:cols],
                       op=nl.subtract)                                                                  # v - k
    nisa.activation(dst=s_k[0:rows, 0:cols], op=nl.sin, data=s_v[0:rows, 0:cols], scale=PI)
    nisa.activation(dst=s_v[0:rows, 0:cols], op=nl.square, data=s_k[0:rows, 0:cols])
    nisa.scalar_tensor_tensor(dst=dst, data=s_v[0:rows, 0:cols], op0=nl.multiply, operand0=b_t,
                              op1=nl.add, operand1=src)


@nki.jit
def snake_resunit(x, w1t, c1, w2t, c2, a1, b1, a2, b2, dilation, tile_t=512, snake_impl=0):
    """One fused residual unit. C channels are split into channel tiles of <= 128 (C = 96 / 192 / 384 / 768).

    Args:
        x:   [B, C, T] bf16 (HBM)
        w1t: [7, C, C] bf16, w1t[k, cin, cout] = conv1.weight[cout, cin, k]
        c1:  [C, 1] fp32 conv1 bias
        w2t: [C, C] bf16, w2t[cin, cout] = conv2.weight[cout, cin, 0]
        c2:  [C, 1] fp32 conv2 bias
        a1, b1, a2, b2: [C, 1] fp32 snake params, a passed as exp(alpha) / pi, b = 1 / (exp(beta) + 1e-9)
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
                            cs, H + n - z0, sr, ss, snake_impl)
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
                            sr, ss, snake_impl)
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


@nki.jit
def snake_ctconv(x, wt, bias, a, b, rate, tile_q=256, snake_impl=0):
    """Block head: snake, then causal transposed conv (kernel 2*rate, stride rate, right-trimmed by rate).

    y[:, q*r + j] = W_j^T s[:, q] + W_{j+r}^T s[:, q-1] + bias   (s = snake(x), s[:, -1] = 0)

    Args:
        x:    [B, Cin, Tin] bf16
        wt:   [2r, Cin, Cout] bf16, wt[k, cin, cout] = ConvTranspose1d.weight[cin, cout, k]
        bias: [Cout, 1] fp32
        a, b: [Cin, 1] fp32 snake params (a passed as exp(alpha) / pi)
        rate: int r (static)
    Returns:
        y: [B, Cout, Tin * r] bf16
    """
    B, Cin, Tin = x.shape
    K, _, Cout = wt.shape
    r = rate
    y = nl.ndarray((B, Cout, Tin * r), dtype=x.dtype, buffer=nl.shared_hbm)
    CI = 128 if Cin > 128 else Cin
    NCI = (Cin + CI - 1) // CI
    CO = 128 if Cout > 128 else Cout
    NCO = (Cout + CO - 1) // CO
    w_sb = nl.ndarray((CI, NCI, K, Cout), dtype=nl.bfloat16, buffer=nl.sbuf)
    prm = nl.ndarray((CI, NCI, 2), dtype=nl.float32, buffer=nl.sbuf)
    for ci in range(NCI):
        cs = min(CI, Cin - ci * CI)
        for k in range(K):
            nisa.dma_copy(dst=w_sb[0:cs, ci, k, 0:Cout], src=wt[k, ci * CI:ci * CI + cs, 0:Cout])
        nisa.dma_copy(dst=prm[0:cs, ci, 0:1], src=a[ci * CI:ci * CI + cs, 0:1])
        nisa.dma_copy(dst=prm[0:cs, ci, 1:2], src=b[ci * CI:ci * CI + cs, 0:1])
    bs = nl.ndarray((CO, NCO, 1), dtype=nl.float32, buffer=nl.sbuf)
    for co in range(NCO):
        os_ = min(CO, Cout - co * CO)
        nisa.dma_copy(dst=bs[0:os_, co, 0:1], src=bias[co * CO:co * CO + os_, 0:1])

    TQ = tile_q
    n_tiles = (Tin + TQ - 1) // TQ
    for bb in range(B):
        for ti in range(n_tiles):
            q0 = ti * TQ
            n = min(TQ, Tin - q0)
            xr = nl.ndarray((CI, NCI, TQ + 1), dtype=x.dtype, buffer=nl.sbuf)
            sn = nl.ndarray((CI, NCI, TQ + 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            sv = nl.ndarray((CI, TQ + 1), dtype=nl.float32, buffer=nl.sbuf)
            sk = nl.ndarray((CI, TQ + 1), dtype=nl.float32, buffer=nl.sbuf)
            z0 = 1 if q0 == 0 else 0
            for ci in range(NCI):
                cs = min(CI, Cin - ci * CI)
                if z0:
                    nisa.memset(dst=sn[0:cs, ci, 0:1], value=0.0)
                nisa.dma_copy(dst=xr[0:cs, ci, z0:n + 1], src=x[bb, ci * CI:ci * CI + cs, q0 - 1 + z0:q0 + n])
                _snake_tile(sn[0:cs, ci, z0:n + 1], xr[0:cs, ci, z0:n + 1], prm[0:cs, ci, 0:1], prm[0:cs, ci, 1:2],
                            cs, n + 1 - z0, sv, sk, snake_impl)
            for co in range(NCO):
                os_ = min(CO, Cout - co * CO)
                yo = nl.ndarray((CO, TQ, r), dtype=x.dtype, buffer=nl.sbuf)
                for j in range(r):
                    ps = nl.ndarray((CO, TQ), dtype=nl.float32, buffer=nl.psum)
                    for ci in range(NCI):
                        cs = min(CI, Cin - ci * CI)
                        nisa.nc_matmul(dst=ps[0:os_, 0:n], stationary=w_sb[0:cs, ci, j, co * CO:co * CO + os_],
                                       moving=sn[0:cs, ci, 1:n + 1])
                        nisa.nc_matmul(dst=ps[0:os_, 0:n], stationary=w_sb[0:cs, ci, j + r, co * CO:co * CO + os_],
                                       moving=sn[0:cs, ci, 0:n])
                    nisa.activation(dst=yo[0:os_, 0:n, j], op=nl.copy, data=ps[0:os_, 0:n], bias=bs[0:os_, co, 0:1])
                nisa.dma_copy(dst=y[bb, co * CO:co * CO + os_, q0 * r:(q0 + n) * r], src=yo[0:os_, 0:n, 0:r])
    return y


@nki.jit
def snake_conv_out(x, wt, bias, a, b, tile_t=512, snake_impl=0):
    """Decoder output: snake, causal conv (k=7, Cout=1), clamp to [-1, 1].

    Args:
        x:    [B, C, T] bf16
        wt:   [7, C, 1] bf16, wt[k, cin, 0] = conv.weight[0, cin, k]
        bias: [1, 1] fp32
        a, b: [C, 1] fp32 snake params (a passed as exp(alpha) / pi)
    Returns:
        y: [B, 1, T] fp32
    """
    B, C, T = x.shape
    K = 7
    H = K - 1
    y = nl.ndarray((B, 1, T), dtype=nl.float32, buffer=nl.shared_hbm)
    w_sb = nl.ndarray((C, K, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
    for k in range(K):
        nisa.dma_copy(dst=w_sb[0:C, k, 0:1], src=wt[k, 0:C, 0:1])
    prm = nl.ndarray((C, 2), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=prm[0:C, 0:1], src=a[0:C, 0:1])
    nisa.dma_copy(dst=prm[0:C, 1:2], src=b[0:C, 0:1])
    bs = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=bs[0:1, 0:1], src=bias[0:1, 0:1])
    TT = tile_t
    W = H + TT
    for bb in range(B):
        for ti in range((T + TT - 1) // TT):
            t0 = ti * TT
            n = min(TT, T - t0)
            lo = t0 - H
            z0 = -lo if lo < 0 else 0
            xr = nl.ndarray((C, W), dtype=x.dtype, buffer=nl.sbuf)
            sn = nl.ndarray((C, W), dtype=nl.bfloat16, buffer=nl.sbuf)
            sv = nl.ndarray((C, W), dtype=nl.float32, buffer=nl.sbuf)
            sk = nl.ndarray((C, W), dtype=nl.float32, buffer=nl.sbuf)
            if z0 > 0:
                nisa.memset(dst=sn[0:C, 0:z0], value=0.0)
            nisa.dma_copy(dst=xr[0:C, z0:H + n], src=x[bb, 0:C, lo + z0:t0 + n])
            _snake_tile(sn[0:C, z0:H + n], xr[0:C, z0:H + n], prm[0:C, 0:1], prm[0:C, 1:2], C, H + n - z0, sv, sk, snake_impl)
            ps = nl.ndarray((1, TT), dtype=nl.float32, buffer=nl.psum)
            for k in range(K):
                nisa.nc_matmul(dst=ps[0:1, 0:n], stationary=w_sb[0:C, k, 0:1], moving=sn[0:C, k:k + n])
            o = nl.ndarray((1, TT), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=o[0:1, 0:n], op=nl.copy, data=ps[0:1, 0:n], bias=bs[0:1, 0:1])
            nisa.tensor_scalar(dst=o[0:1, 0:n], data=o[0:1, 0:n], op0=nl.maximum, operand0=-1.0, op1=nl.minimum,
                               operand1=1.0)
            nisa.dma_copy(dst=y[bb, 0:1, t0:t0 + n], src=o[0:1, 0:n])
    return y


def resunit_params(Bk, ri):
    """Host-side packing of one residual unit's parameters from a Code2Wav block dict (torch tensors)."""
    import torch
    Bk = {k: v.detach().cpu() for k, v in Bk.items()}  # pack on host (device tensors may be non-contiguous)
    w1 = Bk[f"r{ri}w1"]  # [Cout, Cin, 7]
    w2 = Bk[f"r{ri}w2"]  # [Cout, Cin, 1]
    f32 = lambda t: t.detach().float().reshape(-1, 1).contiguous()
    return dict(
        w1t=w1.detach().permute(2, 1, 0).contiguous().to(torch.bfloat16),
        c1=f32(Bk[f"r{ri}c1"]),
        w2t=w2.detach()[:, :, 0].t().contiguous().to(torch.bfloat16),
        c2=f32(Bk[f"r{ri}c2"]),
        a1=f32(Bk[f"r{ri}a1"]) / math.pi, b1=f32(Bk[f"r{ri}b1"]), a2=f32(Bk[f"r{ri}a2"]) / math.pi,
        b2=f32(Bk[f"r{ri}b2"]),
    )


def ctconv_params(Bk):
    """Host-side packing of a decoder block's head (snake + transposed conv)."""
    import torch
    Bk = {k: v.detach().cpu() for k, v in Bk.items()}
    f32 = lambda t: t.float().reshape(-1, 1).contiguous()
    return dict(wt=Bk["tw"].permute(2, 0, 1).contiguous().to(torch.bfloat16), bias=f32(Bk["tb"]),
                a=f32(Bk["a"]) / math.pi, b=f32(Bk["b"]))


def out_params(fw, fcb, fa, fb):
    """Host-side packing of the decoder output (snake + conv to 1 channel)."""
    import torch
    f32 = lambda t: t.detach().cpu().float().reshape(-1, 1).contiguous()
    return dict(wt=fw.detach().cpu().permute(2, 1, 0).contiguous().to(torch.bfloat16), bias=f32(fcb),
                a=f32(fa) / math.pi, b=f32(fb))
