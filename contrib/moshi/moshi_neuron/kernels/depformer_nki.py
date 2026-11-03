"""NKI kernel: one Moshi depformer step (6 transformer layers) for B rows, everything SBUF-resident.

The depformer is a latency-bound chain (8 steps x 6 layers, d=1024, 16 heads x 64, SiLU-GLU 2816, attention over
<= 8 positions). The torch graph spends most of its time on per-op launch/sync of ~150 tiny ops per step, with
engines <= 40% busy (profile 2026-11-01). This kernel runs the 6 layers of ONE step in one launch:

    x [B, 1024] (fp32 residual in SBUF, transposed layout xT [128, 8, B]: d on partitions)
    for layer in 6:
        h = rms_norm(x) * n1
        q, k, v = h @ Wqkv[step]            (Wqkv^T resident per step: [128, 8, 3072])
        K_cache[layer][:, s] = k ; V_cache[layer][:, s] = v      (cache [B, <=8, 1024] in HBM, written by caller)
        attn over positions 0..s (elementwise: tiny S, B on partitions)
        x += attn @ Wo[step]
        h = rms_norm(x) * n2
        x += (silu(h @ Wg) * (h @ Wu)) @ Wd

Layout choice: the matmuls stream the weights as the MOVING operand (N up to 512 columns per instruction) against a
stationary activation tile hT [128 (d chunk), B] -> PSUM [B, N]; K = 1024 = 8 chunks of 128 accumulate in PSUM.
That puts B on the PSUM partition axis, which is also the natural layout for the per-row softmax over <= 8 slots.
Activations are transposed back to [128, B] chunks with nc_transpose for the next matmul.

Inputs (all HBM):
    x      [B, 1024]   bf16    step input (dep_in projection + previous-code embedding), computed by the caller
    kc, vc [6, B, 8, 1024] bf16  per-layer K/V of this frame's earlier steps (positions 0..s-1 valid)
    wqkvT  [6, 1024, 3072] bf16  (step s weights, already transposed: [in, out])
    woT    [6, 1024, 1024], wgT/wuT [6, 1024, 2816], wdT [6, 2816, 1024] bf16
    n1, n2 [6, 1024] fp32
    s      static int: step index (0..7), number of valid earlier positions
Outputs:
    y      [B, 1024] bf16  last-layer output (caller applies the step head + sampling)
    k_new, v_new [6, B, 1024] bf16  this step's K/V per layer (caller writes them into kc/vc at position s)
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

D = 1024
NH = 16
HD = 64
FF = 2816
L = 6
P = 128
DC = D // P            # 8 chunks of the model dim
FC = (FF + P - 1) // P  # 22 chunks of the FFN dim (2816 = 22 * 128)
NMAX = 512


def _rms_T(dstT, srcT, w_sb, B, eps):
    """dstT[128, DC, B] (bf16) = rms_norm over d of srcT[128, DC, B] (fp32) * w; w_sb [128, DC] fp32 (d on parts).
    Sum of squares across partitions + chunks: square, reduce chunks on free axis, then all-partition sum via a
    ones-matmul (stationary ones [128,128] -> every partition gets the total)."""
    sq = nl.ndarray((P, DC, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=sq, data1=srcT, data2=srcT, op=nl.multiply)
    part = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=part, data=sq.rearrange(("p", "c", "b"), ("p", "b", "c")), op=nl.add, axis=(2,))
    ones = nl.ndarray((P, P), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=ones, value=1.0 / D)
    tot_ps = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tot_ps, stationary=ones, moving=part)
    rs = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=rs, data=tot_ps, op0=nl.add, operand0=eps)
    nisa.activation(dst=rs, op=nl.rsqrt, data=rs)
    for c in range(DC):
        t = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=srcT[:, c, :], data2=rs, op=nl.multiply)
        nisa.tensor_scalar(dst=dstT[:, c, :], data=t, op0=nl.multiply, operand0=w_sb[:, c:c + 1])


def _proj(out_sb, hT, w_hbm, layer, B, n_out):
    """out_sb[B, n_out] (fp32) = h @ W  with hT [128, DC_in, B] bf16 (stationary per K chunk) and W [in, out] in HBM
    streamed in [128, <=512] tiles as the moving operand."""
    kin = hT.shape[1]
    for n0 in range(0, n_out, NMAX):
        nn = min(NMAX, n_out - n0)
        ps = nl.ndarray((B, NMAX), dtype=nl.float32, buffer=nl.psum)
        for c in range(kin):
            wt = nl.ndarray((P, NMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=wt[:, 0:nn], src=w_hbm[layer, c * P:(c + 1) * P, n0:n0 + nn])
            nisa.nc_matmul(dst=ps[:, 0:nn], stationary=hT[:, c, :], moving=wt[:, 0:nn])
        nisa.tensor_copy(dst=out_sb[:, n0:n0 + nn], src=ps[:, 0:nn])


def _transpose_to_T(dstT, src, B, nchunks):
    """src [B, nchunks*128] (fp32 SBUF) -> dstT [128, nchunks, B] (bf16) via PE transposes."""
    for c in range(nchunks):
        tb = nl.ndarray((B, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tb, src=src[:, c * P:(c + 1) * P])
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=tb)
        nisa.tensor_copy(dst=dstT[:, c, :], src=tp)


@nki.jit
def depformer_step(x, kc, vc, wqkvT, woT, wgT, wuT, wdT, n1, n2, s, eps):
    B = x.shape[0]
    y = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    k_new = nl.ndarray((L, B, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_new = nl.ndarray((L, B, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # residual x in [B, D] fp32 (B on partitions) and transposed copy for RMSNorm
    xr = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
    xb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:B, 0:D])
    nisa.tensor_copy(dst=xr, data=xb) if False else nisa.tensor_copy(dst=xr, src=xb)

    for l in range(L):
        # --- attention sub-block
        xT = nl.ndarray((P, DC, B), dtype=nl.float32, buffer=nl.sbuf)
        _transpose_to_T(xT, xr, B, DC)
        w1 = nl.ndarray((P, DC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w1, src=n1[l].reshape((DC, P)).rearrange(("c", "p"), ("p", "c")))
        hT = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _rms_T(hT, xT, w1, B, eps)
        qkv = nl.ndarray((B, 3 * D), dtype=nl.float32, buffer=nl.sbuf)
        _proj(qkv, hT, wqkvT, l, B, 3 * D)
        # this step's k, v -> HBM for the caller's cache
        kb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        vb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kb, src=qkv[:, D:2 * D])
        nisa.tensor_copy(dst=vb, src=qkv[:, 2 * D:3 * D])
        nisa.dma_copy(dst=k_new[l, 0:B, 0:D], src=kb)
        nisa.dma_copy(dst=v_new[l, 0:B, 0:D], src=vb)
        # scores over positions 0..s: elementwise on [B, D] tiles, per-head sums on the free axis
        npos = s + 1
        sc = nl.ndarray((B, NH, 8), dtype=nl.float32, buffer=nl.sbuf)
        vall = nl.ndarray((B, 8, D), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(npos):
            kj = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
            if j < s:
                kjb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                vjb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
                nisa.dma_copy(dst=kjb, src=kc[l, 0:B, j, 0:D])
                nisa.dma_copy(dst=vjb, src=vc[l, 0:B, j, 0:D])
                nisa.tensor_copy(dst=kj, src=kjb)
                nisa.tensor_copy(dst=vall[:, j, :], src=vjb)
            else:
                nisa.tensor_copy(dst=kj, src=kb)
                nisa.tensor_copy(dst=vall[:, j, :], src=vb)
            prod = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=qkv[:, 0:D], data2=kj, op=nl.multiply)
            nisa.tensor_reduce(dst=sc[:, :, j], data=prod.reshape((B, NH, HD)), op=nl.add, axis=(2,))
        # softmax over j (scale 1/8)
        mx = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, data=sc[:, :, 0:npos], op=nl.maximum, axis=(2,))
        ex = nl.ndarray((B, NH, 8), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(npos):
            dj = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dj, data1=sc[:, :, j], data2=mx, op=nl.subtract)
            nisa.activation(dst=ex[:, :, j], op=nl.exp, data=dj, scale=HD ** -0.5)
        den = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=den, data=ex[:, :, 0:npos], op=nl.add, axis=(2,))
        nisa.reciprocal(dst=den, data=den)
        att = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=att, value=0.0)
        for j in range(npos):
            pj = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pj, data1=ex[:, :, j], data2=den, op=nl.multiply)
            tj = nl.ndarray((B, NH, HD), dtype=nl.float32, buffer=nl.sbuf)
            for h in range(NH):
                nisa.tensor_scalar(dst=tj[:, h, :], data=vall[:, j, h * HD:(h + 1) * HD], op0=nl.multiply,
                                   operand0=pj[:, h:h + 1])
            nisa.tensor_tensor(dst=att, data1=att, data2=tj.reshape((B, D)), op=nl.add)
        attT = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _transpose_to_T(attT, att, B, DC)
        o = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        _proj(o, attT, woT, l, B, D)
        nisa.tensor_tensor(dst=xr, data1=xr, data2=o, op=nl.add)
        # --- MLP sub-block
        xT2 = nl.ndarray((P, DC, B), dtype=nl.float32, buffer=nl.sbuf)
        _transpose_to_T(xT2, xr, B, DC)
        w2 = nl.ndarray((P, DC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w2, src=n2[l].reshape((DC, P)).rearrange(("c", "p"), ("p", "c")))
        h2T = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _rms_T(h2T, xT2, w2, B, eps)
        g = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
        u = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
        _proj(g, h2T, wgT, l, B, FF)
        _proj(u, h2T, wuT, l, B, FF)
        sg = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=g)
        nisa.tensor_tensor(dst=sg, data1=sg, data2=u, op=nl.multiply)
        sgT = nl.ndarray((P, FC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _transpose_to_T(sgT, sg, B, FC)
        dn = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        _proj(dn, sgT, wdT, l, B, D)
        nisa.tensor_tensor(dst=xr, data1=xr, data2=dn, op=nl.add)

    yb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yb, src=xr)
    nisa.dma_copy(dst=y[0:B, 0:D], src=yb)
    return y, k_new, v_new
