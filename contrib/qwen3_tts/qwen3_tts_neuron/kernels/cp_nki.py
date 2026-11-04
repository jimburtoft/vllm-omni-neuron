"""NKI kernel for the Qwen3-TTS code predictor: one decode step (T=1) of all 5 layers for B rows (B <= 128).

The code predictor runs 15 sequential steps per audio frame (5-layer Qwen3 decoder, d=1024, 16 heads x 128 with
8 KV heads, SiLU-GLU 3072, attention over <= 17 positions). In the torch graph each step is ~100 small ops, and
at B=80 on an LNC=1 core it measures 16-19 ms per frame with every engine < 60 % busy. This kernel runs one step of
all 5 layers in one launch:

    layout: activations [B, *] with B on partitions (rows independent); matmul operands transposed with the PE.
    for layer in 5:
        h  = rms(x) * ln1
        q,k,v = h @ Wqkv                       (weights streamed as the MOVING operand, 512 columns per matmul)
        q,k = rms_head(q|k) * qn|kn ; rope(q,k)  (per-head RMS over d=128 on the free axis)
        K,V[:, pos] = k, v                      (caller-owned cache [L, B, S, 8*128] in HBM; written here)
        scores over positions 0..pos (elementwise, per head), softmax, weighted V sum   (S <= 17, all on B partitions)
        x += o @ Wo
        h  = rms(x) * ln2
        x += (silu(h @ Wg) * (h @ Wu)) @ Wd
    y = rms(x) * norm                           (final norm; caller applies the group head + sampling)

Inputs (HBM): x [B, 1024] bf16; kc, vc [L, B, 8, S, 128] bf16 (positions < pos valid; this step's k/v written at pos);
weights per layer, transposed to [in, out]: wqkv [L, 1024, 4096], wo [L, 2048, 1024], wgu [L, 1024, 6144],
wd [L, 3072, 1024] (bf16); ln1, ln2 [L, 1024], qn, kn [L, 128], norm [1024] fp32; cos, sin [S, 128] fp32;
pos: static int (step position).
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

D = 1024
NH = 16
NKV = 8
HD = 128
QS = NH * HD      # 2048
KS = NKV * HD     # 1024
FF = 3072
L = 5
P = 128
NMAX = 512


def _transpose_to_T(dstT, src, B, nchunks):
    """src [B, nchunks*128] (SBUF, any float) -> dstT [128, nchunks, B] bf16."""
    for c in range(nchunks):
        tb = nl.ndarray((B, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tb, src=src[:, c * P:(c + 1) * P])
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=tb)
        nisa.tensor_copy(dst=dstT[:, c, :], src=tp)


WDGE = int(__import__('os').environ.get('CPK_WDGE', '0'))  # weight-stream DGE mode (timing A/B)
DMA_COLS = 512    # columns per weight DMA. 4096-column (8 MB) blocks measured faster in isolation (236 vs 138 GB/s)
                  # but slower in the full kernel (1.97 vs 1.50 ms per step): they serialize DMA with the matmuls


def _proj(out_sb, hT, w_hbm, layer, B, n_out):
    """out_sb[B, n_out] fp32 = h @ W; hT [128, kin, B] bf16 stationary per K chunk; W [L, in, out] bf16 in HBM.
    The weight is DMA'd in large column blocks (DMA_COLS) and consumed by 512-column matmuls."""
    kin = hT.shape[1]
    wv = w_hbm[layer].reshape((kin, P, n_out)).rearrange(("c", "p", "n"), ("p", "c", "n"))
    dc = min(DMA_COLS, n_out)
    for d0 in range(0, n_out, dc):
        dn = min(dc, n_out - d0)
        wt = nl.ndarray((P, kin, dc), dtype=nl.bfloat16, buffer=nl.sbuf)
        if WDGE == 1:
            nisa.dma_copy(dst=wt[:, :, 0:dn], src=wv[:, :, d0:d0 + dn], dge_mode=nisa.dge_mode.hwdge)
        elif WDGE == 2:
            nisa.dma_copy(dst=wt[:, :, 0:dn], src=wv[:, :, d0:d0 + dn], dge_mode=nisa.dge_mode.none)
        else:
            nisa.dma_copy(dst=wt[:, :, 0:dn], src=wv[:, :, d0:d0 + dn])
        for n0 in range(0, dn, NMAX):
            nn = min(NMAX, dn - n0)
            ps = nl.ndarray((B, NMAX), dtype=nl.float32, buffer=nl.psum)
            for c in range(kin):
                nisa.nc_matmul(dst=ps[:, 0:nn], stationary=hT[:, c, :], moving=wt[:, c, n0:n0 + nn])
            nisa.tensor_copy(dst=out_sb[:, d0 + n0:d0 + n0 + nn], src=ps[:, 0:nn])


def _proj_ws(outT_sb, hT, w_hbm, layer, B, n_out):
    """outT_sb[128, n_out/128, B] fp32 = (h @ W)^T with the WEIGHTS stationary (128x128 tiles) and the activations
    streamed (moving = hT[:, c, :] [128, B]). One LDWEIGHTS per 128x128 weight tile instead of one per K chunk per
    512-column block; the output comes out transposed (feature on partitions), which is what the next matmul needs."""
    kin = hT.shape[1]
    nout = n_out // P
    wv = w_hbm[layer].reshape((kin, P, n_out)).rearrange(("c", "p", "n"), ("p", "c", "n"))
    for ob in range(nout):
        wt = nl.ndarray((P, kin, P), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt, src=wv[:, :, ob * P:(ob + 1) * P])
        ps = nl.ndarray((P, B), dtype=nl.float32, buffer=nl.psum)
        for c in range(kin):
            nisa.nc_matmul(dst=ps, stationary=wt[:, c, :], moving=hT[:, c, :])
        nisa.tensor_copy(dst=outT_sb[:, ob, :], src=ps)


def _rms_rows(dst, src, w_sb, B, n, eps):
    """dst[B, n] = src * rsqrt(mean(src^2) + eps) * w (w_sb [B, n] fp32, rows on partitions)."""
    sq = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    ss = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=sq, op=nl.square, data=src, reduce_op=nl.add, reduce_res=ss, reduce_cmd=nisa.reduce_cmd.reset_reduce)
    nisa.tensor_scalar(dst=ss, data=ss, op0=nl.multiply, operand0=1.0 / n, op1=nl.add, operand1=eps)
    nisa.activation(dst=ss, op=nl.rsqrt, data=ss)
    nisa.tensor_scalar(dst=sq, data=src, op0=nl.multiply, operand0=ss)
    nisa.tensor_tensor(dst=dst, data1=sq, data2=w_sb, op=nl.multiply)


def _bcast_rows(dst, w_hbm_row, B, n):
    """dst[B, n] fp32 = w (a [n] HBM vector) replicated on every partition (stride-0 partition read)."""
    nisa.dma_copy(dst=dst, src=w_hbm_row.ap(pattern=[[0, B], [1, n]]))


def _attention(att, qr, kall, vall, B, npos, skip):
    """att[B, 2048] fp32 = softmax(q k^T / sqrt(d)) v over positions 0..npos-1 for 16 query heads sharing 8 kv heads.
    q [B, 2048] fp32, kall/vall [B, npos, 1024] bf16. Rows on partitions; all on Vector/Scalar engines.
    skip 5: whole-tensor ops (one multiply + reduce per GQA pair member); skip 7: scores only (timing ablation)."""
    sc = nl.ndarray((B, NH, npos), dtype=nl.float32, buffer=nl.sbuf)
    q4 = qr.reshape((B, NKV, 2, HD))
    if skip == 5:
        kvv = kall.reshape((B, npos, NKV, HD)).rearrange(("b", "j", "g", "d"), ("b", "g", "j", "d"))  # [B,8,npos,128]
        for r in range(2):
            prod = nl.ndarray((B, NKV, npos, HD), dtype=nl.float32, buffer=nl.sbuf)
            qv = q4[:, :, r, :].reshape((B, NKV, 1, HD)).broadcast(2, npos)
            nisa.tensor_tensor(dst=prod, data1=qv, data2=kvv, op=nl.multiply)
            nisa.tensor_reduce(dst=sc.reshape((B, NKV, 2, npos))[:, :, r, :], data=prod, op=nl.add, axis=(3,))
    else:
        for j in range(npos):
            kj = kall[:, j, :].reshape((B, NKV, 1, HD)).broadcast(2, 2)  # stride-0 over the head pair
            prod = nl.ndarray((B, NKV, 2, HD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod, data1=q4, data2=kj, op=nl.multiply)
            nisa.tensor_reduce(dst=sc[:, :, j], data=prod.reshape((B, NH, HD)), op=nl.add, axis=(2,))
    if skip == 7:
        nisa.tensor_copy(dst=att.reshape((B, NH, HD))[:, :, 0:npos], src=sc)
        return
    mx = nl.ndarray((B, NH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=mx.reshape((B, NH)), data=sc, op=nl.maximum, axis=(2,))
    dj = nl.ndarray((B, NH, npos), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=dj, data1=sc, data2=mx.broadcast(2, npos), op=nl.subtract)
    ex = nl.ndarray((B, NH, npos), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=ex, op=nl.exp, data=dj, scale=HD ** -0.5)
    den = nl.ndarray((B, NH, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=den.reshape((B, NH)), data=ex, op=nl.add, axis=(2,))
    nisa.reciprocal(dst=den, data=den)
    pr = nl.ndarray((B, NH, npos), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pr, data1=ex, data2=den.broadcast(2, npos), op=nl.multiply)
    a4 = att.reshape((B, NKV, 2, HD))
    pr4 = pr.reshape((B, NKV, 2, npos))
    if skip == 5:
        vvv = vall.reshape((B, npos, NKV, HD)).rearrange(("b", "j", "g", "d"), ("b", "g", "d", "j"))  # [B,8,128,npos]
        for r in range(2):
            pv = nl.ndarray((B, NKV, HD, npos), dtype=nl.float32, buffer=nl.sbuf)
            pb = pr4[:, :, r, :].reshape((B, NKV, 1, npos)).broadcast(2, HD)
            nisa.tensor_tensor(dst=pv, data1=pb, data2=vvv, op=nl.multiply)
            nisa.tensor_reduce(dst=a4[:, :, r, :], data=pv, op=nl.add, axis=(3,))
        return
    for j in range(npos):
        pj = pr4[:, :, :, j:j + 1].broadcast(3, HD)
        vj = vall[:, j, :].reshape((B, NKV, 1, HD)).broadcast(2, 2)
        if j == 0:
            nisa.tensor_tensor(dst=a4, data1=pj, data2=vj, op=nl.multiply)
        else:
            t = nl.ndarray((B, NKV, 2, HD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=t, data1=pj, data2=vj, op=nl.multiply)
            nisa.tensor_tensor(dst=a4, data1=a4, data2=t, op=nl.add)


def _attention_pg(att_hbm, q_hbm, kc, vc, l, B, npos):
    """Attention with (row, kv-head) pairs on partitions: 8*B partitions in tiles of 128 (16 rows x 8 kv heads).
    q_hbm [B, 2048] fp32 scratch (heads 2g, 2g+1 of row b are q_hbm.reshape(8B, 256)[8b+g]); kc/vc [L, B, 8, S, 128]
    bf16 (positions 0..npos-1 valid, including this step's). Writes att_hbm [B, 2048] fp32 (same (b, g) layout)."""
    S = kc.shape[3]
    NP = B * NKV
    qv = q_hbm.reshape((NP, 2 * HD))
    av = att_hbm.reshape((NP, 2 * HD))
    kv_ = kc[l].reshape((NP, S, HD))
    vv_ = vc[l].reshape((NP, S, HD))
    for t0 in range(0, NP, P):
        n = min(P, NP - t0)
        qt = nl.ndarray((P, 2, HD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=qt[0:n].reshape((n, 2 * HD)), src=qv[t0:t0 + n, :])
        kt = nl.ndarray((P, npos, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        vt = nl.ndarray((P, npos, HD), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=kt[0:n], src=kv_[t0:t0 + n, 0:npos, :])
        nisa.dma_copy(dst=vt[0:n], src=vv_[t0:t0 + n, 0:npos, :])
        sc = nl.ndarray((P, 2, npos), dtype=nl.float32, buffer=nl.sbuf)
        for r in range(2):
            prod = nl.ndarray((P, npos, HD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=prod[0:n], data1=qt[0:n, r:r + 1, :].broadcast(1, npos), data2=kt[0:n], op=nl.multiply)
            nisa.tensor_reduce(dst=sc[0:n, r, :], data=prod[0:n], op=nl.add, axis=(2,))
        mx = nl.ndarray((P, 2, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx[0:n].reshape((n, 2)), data=sc[0:n], op=nl.maximum, axis=(2,))
        dj = nl.ndarray((P, 2, npos), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=dj[0:n], data1=sc[0:n], data2=mx[0:n].broadcast(2, npos), op=nl.subtract)
        ex = nl.ndarray((P, 2, npos), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=ex[0:n], op=nl.exp, data=dj[0:n], scale=HD ** -0.5)
        den = nl.ndarray((P, 2, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=den[0:n].reshape((n, 2)), data=ex[0:n], op=nl.add, axis=(2,))
        nisa.reciprocal(dst=den[0:n], data=den[0:n])
        pr = nl.ndarray((P, 2, npos), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=pr[0:n], data1=ex[0:n], data2=den[0:n].broadcast(2, npos), op=nl.multiply)
        at = nl.ndarray((P, 2, HD), dtype=nl.float32, buffer=nl.sbuf)
        vtt = vt.rearrange(("p", "j", "d"), ("p", "d", "j"))
        for r in range(2):
            pv = nl.ndarray((P, HD, npos), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pv[0:n], data1=pr[0:n, r:r + 1, :].broadcast(1, HD), data2=vtt[0:n], op=nl.multiply)
            nisa.tensor_reduce(dst=at[0:n, r, :], data=pv[0:n], op=nl.add, axis=(2,))
        nisa.dma_copy(dst=av[t0:t0 + n, :], src=at[0:n].reshape((n, 2 * HD)))


@nki.jit
def cp_step(x, kc, vc, wqkv, wo, wgu, wd, ln1, ln2, qn, kn, norm, cos, sin, pos, eps, ver=8, skip=0, attn=1, wdge=0):
    B = x.shape[0]
    S = kc.shape[3]
    npos = pos + 1
    q_s = nl.ndarray((B, QS), dtype=nl.float32, buffer=nl.shared_hbm)   # attention scratch (relayout via HBM)
    a_s = nl.ndarray((B, QS), dtype=nl.float32, buffer=nl.shared_hbm)
    y = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    xr = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
    xb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(dst=xb, src=x[0:B, 0:D])
    nisa.tensor_copy(dst=xr, src=xb)
    cs = nl.ndarray((B, HD), dtype=nl.float32, buffer=nl.sbuf)
    sn = nl.ndarray((B, HD), dtype=nl.float32, buffer=nl.sbuf)
    _bcast_rows(cs, cos[pos], B, HD)
    _bcast_rows(sn, sin[pos], B, HD)
    for l in range(L):
        # ---------------- attention
        w1 = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        _bcast_rows(w1, ln1[l], B, D)
        hn = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        _rms_rows(hn, xr, w1, B, D, eps)
        hT = nl.ndarray((P, D // P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _transpose_to_T(hT, hn, B, D // P)
        qkv = nl.ndarray((B, QS + 2 * KS), dtype=nl.float32, buffer=nl.sbuf)
        _proj(qkv, hT, wqkv, l, B, QS + 2 * KS)
        wq = nl.ndarray((B, HD), dtype=nl.float32, buffer=nl.sbuf)
        wk = nl.ndarray((B, HD), dtype=nl.float32, buffer=nl.sbuf)
        _bcast_rows(wq, qn[l], B, HD)
        _bcast_rows(wk, kn[l], B, HD)
        # per-head RMS + RoPE (rotate_half) on q (16 heads) and k (8 heads), all 24 heads per instruction
        NQK = NH + NKV
        qk = qkv[:, 0:QS + KS].reshape((B, NQK, HD))
        sq = nl.ndarray((B, NQK, HD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sq, data1=qk, data2=qk, op=nl.multiply)
        ms = nl.ndarray((B, NQK, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=ms.reshape((B, NQK)), data=sq, op=nl.add, axis=(2,))
        nisa.tensor_scalar(dst=ms, data=ms, op0=nl.multiply, operand0=1.0 / HD, op1=nl.add, operand1=eps)
        nisa.activation(dst=ms, op=nl.rsqrt, data=ms)
        t = nl.ndarray((B, NQK, HD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=t, data1=qk, data2=ms.broadcast(2, HD), op=nl.multiply)
        wqk = nl.ndarray((B, NQK, HD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=wqk[:, 0:NH, :], src=wq.reshape((B, 1, HD)).broadcast(1, NH))
        nisa.tensor_copy(dst=wqk[:, NH:NQK, :], src=wk.reshape((B, 1, HD)).broadcast(1, NKV))
        nisa.tensor_tensor(dst=t, data1=t, data2=wqk, op=nl.multiply)
        rot = nl.ndarray((B, NQK, HD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=rot[:, :, 0:HD // 2], data=t[:, :, HD // 2:HD], op0=nl.multiply, operand0=-1.0)
        nisa.tensor_copy(dst=rot[:, :, HD // 2:HD], src=t[:, :, 0:HD // 2])
        csb = cs.reshape((B, 1, HD)).broadcast(1, NQK)
        snb = sn.reshape((B, 1, HD)).broadcast(1, NQK)
        nisa.tensor_tensor(dst=t, data1=t, data2=csb, op=nl.multiply)
        nisa.tensor_tensor(dst=rot, data1=rot, data2=snb, op=nl.multiply)
        qr = nl.ndarray((B, QS), dtype=nl.float32, buffer=nl.sbuf)
        kr = nl.ndarray((B, KS), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=qr.reshape((B, NH, HD)), data1=t[:, 0:NH, :], data2=rot[:, 0:NH, :], op=nl.add)
        nisa.tensor_tensor(dst=kr.reshape((B, NKV, HD)), data1=t[:, NH:NQK, :], data2=rot[:, NH:NQK, :], op=nl.add)
        vb = nl.ndarray((B, KS), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=vb, src=qkv[:, QS + KS:QS + 2 * KS])
        # cache [L, B, 8, S, 128]: this step's k/v at [l, b, g, pos, :]
        nisa.dma_copy(dst=kc[l, 0:B, :, pos, :], src=kr.reshape((B, NKV, HD)))
        nisa.dma_copy(dst=vc[l, 0:B, :, pos, :], src=vb.reshape((B, NKV, HD)))
        # scores [B, NH, npos]: per head h, kv head h // 2
        kall = nl.ndarray((B, npos, KS), dtype=nl.bfloat16, buffer=nl.sbuf)
        vall = nl.ndarray((B, npos, KS), dtype=nl.bfloat16, buffer=nl.sbuf)
        if pos > 0 and skip != 6 and attn == 0:
            nisa.dma_copy(dst=kall[:, 0:pos, :].reshape((B, pos, NKV, HD)),
                          src=kc[l, 0:B, :, 0:pos, :].rearrange(("b", "g", "j", "d"), ("b", "j", "g", "d")))
            nisa.dma_copy(dst=vall[:, 0:pos, :].reshape((B, pos, NKV, HD)),
                          src=vc[l, 0:B, :, 0:pos, :].rearrange(("b", "g", "j", "d"), ("b", "j", "g", "d")))
        nisa.tensor_copy(dst=kall[:, pos, :], src=kr)
        nisa.tensor_copy(dst=vall[:, pos, :], src=vb)
        att = nl.ndarray((B, QS), dtype=nl.float32, buffer=nl.sbuf)
        if skip == 1:
            nisa.tensor_copy(dst=att, src=qr)
        elif attn == 1:
            nisa.dma_copy(dst=q_s[0:B, 0:QS], src=qr)
            _attention_pg(a_s, q_s, kc, vc, l, B, npos)
            nisa.dma_copy(dst=att, src=a_s[0:B, 0:QS])
        else:
            _attention(att, qr, kall, vall, B, npos, skip)
        attT = nl.ndarray((P, QS // P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _transpose_to_T(attT, att, B, QS // P)
        o = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        _proj(o, attT, wo, l, B, D)
        nisa.tensor_tensor(dst=xr, data1=xr, data2=o, op=nl.add)
        if skip != 2:
            # ---------------- MLP
            w2 = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
            _bcast_rows(w2, ln2[l], B, D)
            hn2 = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
            _rms_rows(hn2, xr, w2, B, D, eps)
            h2T = nl.ndarray((P, D // P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
            _transpose_to_T(h2T, hn2, B, D // P)
            gu = nl.ndarray((B, 2 * FF), dtype=nl.float32, buffer=nl.sbuf)
            _proj(gu, h2T, wgu, l, B, 2 * FF)
            sg = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
            nisa.activation(dst=sg, op=nl.silu, data=gu[:, 0:FF])
            nisa.tensor_tensor(dst=sg, data1=sg, data2=gu[:, FF:2 * FF], op=nl.multiply)
            sgT = nl.ndarray((P, FF // P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
            _transpose_to_T(sgT, sg, B, FF // P)
            dn = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
            _proj(dn, sgT, wd, l, B, D)
            nisa.tensor_tensor(dst=xr, data1=xr, data2=dn, op=nl.add)
    wn = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
    _bcast_rows(wn, norm, B, D)
    yo = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
    _rms_rows(yo, xr, wn, B, D, eps)
    yb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=yb, src=yo)
    nisa.dma_copy(dst=y[0:B, 0:D], src=yb)
    return y


def cp_params(cp):
    """Host-side packing of the code predictor's layer weights for cp_step (torch tensors on cp's device)."""
    import torch
    cpu = lambda t: t.detach().cpu()  # pack on host (device tensors may be non-contiguous views)
    st = lambda f: torch.stack([cpu(f(l)) for l in cp.layers]).contiguous()
    f32 = lambda t: cpu(t).float().contiguous()
    return dict(
        wqkv=st(lambda l: l.w_qkv.detach()), wo=st(lambda l: l.w_o.detach()),
        wgu=st(lambda l: l.w_gate_up.detach()), wd=st(lambda l: l.w_down.detach()),
        ln1=f32(st(lambda l: l.ln1.detach())), ln2=f32(st(lambda l: l.ln2.detach())),
        qn=f32(st(lambda l: l.qn.detach())), kn=f32(st(lambda l: l.kn.detach())),
        norm=f32(cp.norm.detach()), cos=f32(cp.cos.detach()), sin=f32(cp.sin.detach()))
