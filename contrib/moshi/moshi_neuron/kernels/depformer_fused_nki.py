"""NKI kernel: the whole Moshi depformer chain (all NS steps x 6 layers + per-step head, sampling, code embedding) in ONE
launch, for B rows. Same math as `depformer_step` x NS plus the torch glue in `MoshiLM.depformer_nki`:

    prev = text-embedding (caller)
    for s in 0..NS-1:
        x = din[s] + prev
        y, k_s, v_s = depformer_step(x, K/V of steps < s)          (6 layers, attention over positions 0..s)
        logits = y @ head[s]^T                                       [B, 2048] fp32
        code = multinomial top-k sample(logits / temp, k=TOPK, u[s])  (exactly lm.sample's multinomial branch)
        prev = emb[s][code]                                         (one-hot @ emb -> [B, 1024])

Everything stays in SBUF across steps (K/V of this frame [6, B, NS, 1024], codes); only weights stream from HBM.
The top-k uses the hardware 8-wide primitives: TOPK/8 rounds of max8 + nc_match_replace8 give the TOPK largest values
in descending order and their vocab indices (ties: first occurrence, as torch.topk on equal values may differ only on
exact ties). The inverse-CDF draw: j = #{c_i < u} over the softmax cumsum of the sorted values, code = idx[j].

Inputs (HBM): din [NS, B, 1024] bf16 (the per-step input projections), prev0 [B, 1024] bf16, u [B, NS] fp32,
    wqkvT/woT/wgT/wuT/wdT [Q, 6, ...] bf16, n1/n2 [6, 1024] fp32, headT [Q, 1024, 2048] bf16 (head weights [in, out]),
    emb [Q-1, 2048, 1024] bf16 (code embeddings for the next step)
Output: codes [B, NS] int32
"""
from __future__ import annotations

import nki
import nki.isa as nisa
import nki.language as nl

from .depformer_nki import D, DC, FC, FF, HD, L, NH, NMAX, P, QMAX, _proj, _proj_split, _rms_T, _transpose_to_T

CARD = 2048


def _layers(xr, kc_sb, vc_sb, s, wqkvT, woT, wgT, wuT, wdT, n1, n2, eps, B, proj, prg, wset):
    """6 layers of step s on the residual xr [B, D] fp32 (in place). kc_sb / vc_sb [B, L, QMAX, D] bf16 SBUF hold the
    K/V of steps < s; this step's k, v are written at position s."""
    for l in range(L):
        xT = nl.ndarray((P, DC, B), dtype=nl.float32, buffer=nl.sbuf)
        _transpose_to_T(xT, xr, B, DC)
        w1 = nl.ndarray((P, DC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w1, src=n1[l].reshape((DC, P)).rearrange(("c", "p"), ("p", "c")))
        hT = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _rms_T(hT, xT, w1, B, eps)
        qkv = nl.ndarray((B, 3 * D), dtype=nl.float32, buffer=nl.sbuf)
        proj(qkv, hT, wqkvT[wset], l, 3 * D, 0)
        nisa.tensor_copy(dst=kc_sb[:, l, s, :], src=qkv[:, D:2 * D])
        nisa.tensor_copy(dst=vc_sb[:, l, s, :], src=qkv[:, 2 * D:3 * D])
        npos = s + 1
        # scores [B, npos, NH]: q x K over all positions 0..s at once
        kf = nl.ndarray((B, QMAX, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=kf[:, 0:npos, :], src=kc_sb[:, l, 0:npos, :])
        prods = nl.ndarray((B, QMAX, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=prods[:, 0:npos, :], data1=kf[:, 0:npos, :],
                           data2=qkv[:, 0:D].reshape((B, 1, D)).broadcast(1, npos), op=nl.multiply)
        sc = nl.ndarray((B, QMAX, NH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=sc[:, 0:npos, :], data=prods[:, 0:npos, :].reshape((B, npos, NH, HD)), op=nl.add,
                           axis=(3,))
        # softmax over positions (per head); scale 1/8
        scT = nl.ndarray((B, NH, QMAX), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(npos):
            nisa.tensor_copy(dst=scT[:, :, j], src=sc[:, j, :])
        mx = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=mx, data=scT[:, :, 0:npos], op=nl.maximum, axis=(2,))
        ex = nl.ndarray((B, NH, QMAX), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(npos):
            dj = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=dj, data1=scT[:, :, j], data2=mx, op=nl.subtract)
            nisa.activation(dst=ex[:, :, j], op=nl.exp, data=dj, scale=HD ** -0.5)
        den = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=den, data=ex[:, :, 0:npos], op=nl.add, axis=(2,))
        nisa.reciprocal(dst=den, data=den)
        att = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=att, value=0.0)
        vf = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        for j in range(npos):
            pj = nl.ndarray((B, NH), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=pj, data1=ex[:, :, j], data2=den, op=nl.multiply)
            nisa.tensor_copy(dst=vf, src=vc_sb[:, l, j, :])
            tj = nl.ndarray((B, NH, HD), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=tj, data1=vf.reshape((B, NH, HD)), data2=pj.reshape((B, NH, 1)).broadcast(2, HD),
                               op=nl.multiply)
            nisa.tensor_tensor(dst=att, data1=att, data2=tj.reshape((B, D)), op=nl.add)
        attT = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _transpose_to_T(attT, att, B, DC)
        o = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        proj(o, attT, woT[wset], l, D, 1)
        nisa.tensor_tensor(dst=xr, data1=xr, data2=o, op=nl.add)
        xT2 = nl.ndarray((P, DC, B), dtype=nl.float32, buffer=nl.sbuf)
        _transpose_to_T(xT2, xr, B, DC)
        w2 = nl.ndarray((P, DC), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=w2, src=n2[l].reshape((DC, P)).rearrange(("c", "p"), ("p", "c")))
        h2T = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _rms_T(h2T, xT2, w2, B, eps)
        g = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
        u = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
        proj(g, h2T, wgT[wset], l, FF, 0)
        proj(u, h2T, wuT[wset], l, FF, 1)
        sg = nl.ndarray((B, FF), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=sg, op=nl.silu, data=g)
        nisa.tensor_tensor(dst=sg, data1=sg, data2=u, op=nl.multiply)
        sgT = nl.ndarray((P, FC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
        _transpose_to_T(sgT, sg, B, FC)
        dn = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        proj(dn, sgT, wdT[wset], l, D, 0)
        nisa.tensor_tensor(dst=xr, data1=xr, data2=dn, op=nl.add)


def _sample_topk(code_f, lg, u_col, B, temp, topk):
    """code_f [B, 1] fp32 <- multinomial top-k sample of lg [B, CARD] fp32 (scaled by 1/temp here), u_col [B, 1]."""
    l = nl.ndarray((B, CARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=l, data=lg, op0=nl.multiply, operand0=1.0 / temp)
    # exact top-k for any k: ceil(k/8) rounds of the 8-wide primitives give the ceil(k/8)*8 largest in descending
    # order; the last round's surplus (k % 8 != 0) is dropped by zeroing its probabilities below
    nr = (topk + 7) // 8
    vals = nl.ndarray((B, nr * 8), dtype=nl.float32, buffer=nl.sbuf)
    idx = nl.ndarray((B, nr * 8), dtype=nl.uint32, buffer=nl.sbuf)
    work = nl.ndarray((B, CARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=work, src=l)
    for r in range(nr):
        nisa.max8(dst=vals[:, r * 8:(r + 1) * 8], src=work)
        nisa.nc_match_replace8(dst=work, data=work, vals=vals[:, r * 8:(r + 1) * 8], imm=-3.0e38,
                               dst_idx=idx[:, r * 8:(r + 1) * 8])
    # softmax over the kept sorted values -> cumulative -> count(c < u)
    n = topk
    ex = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    neg = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=neg, data=vals[:, 0:1], op0=nl.multiply, operand0=-1.0)
    nisa.activation(dst=ex, op=nl.exp, data=vals[:, 0:n], bias=neg)
    # c_i = cumsum(ex)_i / total: normalize like torch (softmax first, then cumsum) so u compares against the same
    # rounded CDF values the torch path builds
    cs0 = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=cs0, src=ex, op0=nl.multiply, op1=nl.add, imm0=1.0)
    rt = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=rt, data=cs0[:, n - 1:n])
    pn = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=pn, data=ex, op0=nl.multiply, operand0=rt)
    cs = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar_cumulative(dst=cs, src=pn, op0=nl.multiply, op1=nl.add, imm0=1.0)
    lt = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lt, data=cs, op0=nl.less, operand0=u_col)
    j = nl.ndarray((B, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=j, data=lt, op=nl.add, axis=(1,))
    nisa.tensor_scalar(dst=j, data=j, op0=nl.minimum, operand0=float(n - 1))
    # code = idx[j]: one-hot over the n sorted slots, dot with idx (as fp32; vocab < 2^24 is exact)
    idxf = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idxf, src=idx[:, 0:n])
    pos = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pos, [[1, n]], 0)
    oh = nl.ndarray((B, n), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh, data=pos, op0=nl.equal, operand0=j)
    nisa.tensor_tensor(dst=oh, data1=oh, data2=idxf, op=nl.multiply)
    nisa.tensor_reduce(dst=code_f, data=oh, op=nl.add, axis=(1,))


def _proj_rows(out_sb, hT, w_hbm, layer, B, n_out, n_in):
    """_proj with the weight's input dim limited to its first n_in rows (n_in multiple of 128)."""
    kin = n_in // P
    wv = w_hbm[layer][0:n_in, :].reshape((kin, P, n_out)).rearrange(("c", "p", "n"), ("p", "c", "n"))
    nblk = (n_out + NMAX - 1) // NMAX
    wts = []
    for bi in range(nblk):
        n0 = bi * NMAX
        nn = min(NMAX, n_out - n0)
        wt = nl.ndarray((P, kin, NMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt[:, :, 0:nn], src=wv[:, :, n0:n0 + nn])
        wts.append((wt, n0, nn))
    for wt, n0, nn in wts:
        ps = nl.ndarray((B, NMAX), dtype=nl.float32, buffer=nl.psum)
        for c in range(kin):
            nisa.nc_matmul(dst=ps[:, 0:nn], stationary=hT[:, c, :], moving=wt[:, c, 0:nn])
        nisa.tensor_copy(dst=out_sb[:, n0:n0 + nn], src=ps[:, 0:nn])


def _head_logits(lg, yT, headT, s, B):
    """lg [B, CARD] fp32 = y @ head[s]^T with yT [128, DC, B] bf16 and headT [Q, D, CARD] ([in, out])."""
    _proj(lg, yT, headT, s, B, CARD)


def _embed(prev, code_f, emb, s, B):
    """prev [B, D] fp32 = emb[s][code] via a one-hot [B, CARD] matmul (one-hot bf16 is exact)."""
    pos = nl.ndarray((B, CARD), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(pos, [[1, CARD]], 0)
    oh = nl.ndarray((B, CARD), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=oh, data=pos, op0=nl.equal, operand0=code_f)
    ohT = nl.ndarray((P, CARD // P, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    for c in range(CARD // P):
        tp = nl.ndarray((P, B), dtype=nl.bfloat16, buffer=nl.psum)
        nisa.nc_transpose(dst=tp, data=oh[:, c * P:(c + 1) * P])
        nisa.tensor_copy(dst=ohT[:, c, :], src=tp)
    # emb[s] may carry extra rows past CARD (moshi tables are [2049, 1024]: the extra row is the initial token,
    # never a sampled code) -> contract over the first CARD rows only
    _proj_rows(prev, ohT, emb, s, B, D, CARD)


@nki.jit
def depformer_all(din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb, ns, q, temp, topk, eps, ver=1,
                  dbg_logits=0, force=None):
    """All `ns` depformer steps (weight set s % q) + head + top-k sampling + code embedding. Returns codes [B, ns]
    int32. Runs on one core per LNC2 pair: the grid-2 launch does the work on program 0 only (program 1 idles), since
    the step chain is serial and the pair split costs a cross-core handshake per projection (Task 062)."""
    B = din.shape[1]
    codes = nl.ndarray((B, ns), dtype=nl.int32, buffer=nl.shared_hbm)
    lgo = nl.ndarray((ns, B, CARD), dtype=nl.float32, buffer=nl.shared_hbm) if dbg_logits else None
    if dbg_logits:
        # zero the whole buffer on every program so the non-working core's copy (if the runtime returns it) is defined
        zb = nl.ndarray((B, CARD), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=zb, value=0.0)
    prg = nl.program_id(0) if nl.num_programs(0) > 1 else 0
    if prg == 0:
        kc_sb = nl.ndarray((B, L, QMAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        vc_sb = nl.ndarray((B, L, QMAX, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        u_sb = nl.ndarray((B, ns), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=u_sb, src=uu)
        prev = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
        pb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=pb, src=prev0)
        nisa.tensor_copy(dst=prev, src=pb)
        codes_f = nl.ndarray((B, ns), dtype=nl.float32, buffer=nl.sbuf)
        proj = lambda o_, h_, w_, l_, n_, pp: _proj(o_, h_, w_, l_, B, n_)
        for s in range(ns):
            ws = s % q
            xr = nl.ndarray((B, D), dtype=nl.float32, buffer=nl.sbuf)
            db = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=db, src=din[ws])
            nisa.tensor_copy(dst=xr, src=db)
            # x = bf16(din + prev), as the torch path does (.to(bf16)) before the step
            nisa.tensor_tensor(dst=xr, data1=xr, data2=prev, op=nl.add)
            xb = nl.ndarray((B, D), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.tensor_copy(dst=xb, src=xr)
            nisa.tensor_copy(dst=xr, src=xb)
            _layers(xr, kc_sb, vc_sb, s, wqkvT, woT, wgT, wuT, wdT, n1, n2, eps, B, proj, 0, ws)
            yT = nl.ndarray((P, DC, B), dtype=nl.bfloat16, buffer=nl.sbuf)
            _transpose_to_T(yT, xr, B, DC)
            lg = nl.ndarray((B, CARD), dtype=nl.float32, buffer=nl.sbuf)
            _head_logits(lg, yT, headT, ws, B)
            if dbg_logits:
                nisa.dma_copy(dst=lgo[s], src=lg)
            if temp > 0:
                _sample_topk(codes_f[:, s:s + 1], lg, u_sb[:, s:s + 1], B, temp, topk)
            else:
                # greedy (validation): argmax via max8 + find_index8
                m8 = nl.ndarray((B, 8), dtype=nl.float32, buffer=nl.sbuf)
                nisa.max8(dst=m8, src=lg)
                i8 = nl.ndarray((B, 8), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.nc_find_index8(dst=i8, data=lg, vals=m8)
                nisa.tensor_copy(dst=codes_f[:, s:s + 1], src=i8[:, 0:1])
            nxt = codes_f[:, s:s + 1]
            if force is not None:
                # teacher forcing (validation only): feed the given code into the next step instead of the sample
                if s == 0:
                    fb = nl.ndarray((B, ns), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.dma_copy(dst=fb, src=force)
                    ff = nl.ndarray((B, ns), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=ff, src=fb)
                nxt = ff[:, s:s + 1]
            if s + 1 < ns:
                _embed(prev, nxt, emb, min(ws, q - 2), B)
        ci = nl.ndarray((B, ns), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ci, src=codes_f)
        nisa.dma_copy(dst=codes, src=ci)
    if dbg_logits:
        return codes, lgo
    return codes
