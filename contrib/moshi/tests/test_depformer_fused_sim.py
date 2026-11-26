"""Simulator test: kernels/depformer_fused_nki.depformer_all vs a numpy reference of the same chain (random weights).
Checks the sampled codes match exactly and reports per-step logit agreement. usage: B=4 NS=8 python ..."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nki  # noqa: E402

from moshi_neuron.kernels.depformer_fused_nki import CARD, depformer_all  # noqa: E402
from moshi_neuron.kernels.depformer_nki import D, FF, HD, L, NH  # noqa: E402


def bfr(x):
    import ml_dtypes
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


def ref(din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb, ns, q, temp, topk, eps, force=None, dbg=None):
    f = lambda a: np.asarray(a, np.float32)
    B = din.shape[1]
    prev = f(prev0)
    K = np.zeros((L, B, ns, D), np.float32); V = np.zeros((L, B, ns, D), np.float32)
    codes = np.zeros((B, ns), np.int64)
    for s in range(ns):
        ws = s % q
        x = bfr(f(din[ws]) + prev)
        for l in range(L):
            h = bfr(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * n1[l])
            qkv = h @ f(wqkvT[ws, l])
            qq, kk, vv = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
            K[l, :, s], V[l, :, s] = bfr(kk), bfr(vv)
            qh = qq.reshape(B, NH, HD); Kh = K[l, :, :s + 1].reshape(B, s + 1, NH, HD)
            Vh = V[l, :, :s + 1].reshape(B, s + 1, NH, HD)
            sc = np.einsum("bhd,bjhd->bhj", qh, Kh) * HD ** -0.5
            p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
            o = np.einsum("bhj,bjhd->bhd", p, Vh).reshape(B, D)
            x = x + bfr(o) @ f(woT[ws, l])
            h = bfr(x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * n2[l])
            g = h @ f(wgT[ws, l]); u = h @ f(wuT[ws, l])
            x = x + bfr((g / (1 + np.exp(-g))) * u) @ f(wdT[ws, l])
        lg = bfr(x) @ f(headT[ws])
        l_ = lg / temp
        idx = np.argsort(-l_, axis=-1, kind="stable")[:, :topk]
        vals = np.take_along_axis(l_, idx, -1)
        pr = np.exp(vals - vals[:, :1]); c = np.cumsum(pr, -1) / pr.sum(-1, keepdims=True)
        j = np.minimum((c < uu[:, s:s + 1]).sum(-1), topk - 1)
        code = idx[np.arange(B), j]
        if dbg is not None:
            dbg.append(dict(lg=l_, idx=idx, c=c, j=j, code=code.copy()))
        if force is not None:
            code = force[:, s]
        codes[:, s] = code
        if s + 1 < ns:
            prev = f(emb[min(ws, q - 2)])[code]
    return codes


def main():
    rng = np.random.default_rng(0)
    B, ns, q = int(os.environ.get("B", "4")), int(os.environ.get("NS", "3")), 8
    temp, topk, eps = 0.8, int(os.environ.get('TOPK', '250')), 1e-8
    import ml_dtypes
    bf = lambda x: x.astype(ml_dtypes.bfloat16)
    r = lambda *sh, sc=0.02: bf(rng.standard_normal(sh).astype(np.float32) * sc)
    din = r(q, B, D, sc=1.0); prev0 = r(B, D, sc=1.0)
    uu = rng.uniform(0.0, 1.0, (B, ns)).astype(np.float32)
    wqkvT, woT = r(q, L, D, 3 * D), r(q, L, D, D)
    wgT, wuT, wdT = r(q, L, D, FF), r(q, L, D, FF), r(q, L, FF, D)
    n1 = (1 + 0.1 * rng.standard_normal((L, D))).astype(np.float32)
    n2 = (1 + 0.1 * rng.standard_normal((L, D))).astype(np.float32)
    headT = r(q, D, CARD, sc=0.05); emb = r(q - 1, CARD + 1, D, sc=1.0)   # real tables have 2049 rows
    out, klg = nki.simulate(depformer_all[2])(din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb, ns, q,
                                              temp, topk, eps, 1, 1)
    klg = np.asarray(klg, np.float32)
    rc = ref(din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb, ns, q, temp, topk, eps)
    out = np.asarray(out).astype(np.int64)
    dbg = []
    rf = ref(din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb, ns, q, temp, topk, eps, force=out, dbg=dbg)
    for s in range(ns):
        d = dbg[s]
        for b in range(B):
            if d["code"][b] != out[b, s]:
                jj = int(d["j"][b]); kc = int(np.where(d["idx"][b] == out[b, s])[0][0]) if out[b, s] in d["idx"][b] else -1
                if os.environ.get("VERBOSE"): print(f"  step {s} row {b}: ref j={jj} code={d['code'][b]} | kernel code={out[b, s]} at sorted pos {kc}; "
                      f"u={uu[b, s]:.6f} c[j-1..j+1]={d['c'][b, max(jj-1,0):jj+2]} c[k]={d['c'][b, kc] if kc >= 0 else None}")
    # a mismatch is a near-tie when u lies within TOL of the CDF boundary between the two codes (logit rounding)
    TOL = float(os.environ.get("TOL", "2e-3"))
    hard = 0
    for s in range(ns):
        d = dbg[s]
        for b in range(B):
            if d["code"][b] != out[b, s]:
                kc = int(np.where(d["idx"][b] == out[b, s])[0][0]) if out[b, s] in d["idx"][b] else -1
                lo = d["c"][b, kc - 1] if kc > 0 else 0.0
                hi = d["c"][b, kc] if kc >= 0 else -1
                if not (kc >= 0 and lo - TOL <= uu[b, s] <= hi + TOL):
                    hard += 1
    m = (out == rc).mean()
    # how close are the two candidates' logits (the swap is a near-tie in logit space when |dl| is tiny)
    dls = []
    for s in range(ns):
        d = dbg[s]
        for b in range(B):
            if d["code"][b] != out[b, s]:
                dls.append(abs(float(d["lg"][b, d["code"][b]] - d["lg"][b, out[b, s]])))
    if dls:
        print(f"  mismatch |logit_ref(code_ref) - logit_ref(code_kernel)| (x temp): max {max(dls)*0.8:.4f} median {sorted(dls)[len(dls)//2]*0.8:.4f}")
    rel = [float(np.linalg.norm(klg[s] - dbg[s]["lg"] * temp) / np.linalg.norm(dbg[s]["lg"] * temp)) for s in range(ns)]
    same = 0
    for s in range(ns):
        l_ = klg[s] / temp
        idx = np.argsort(-l_, axis=-1, kind="stable")[:, :topk]
        vals = np.take_along_axis(l_, idx, -1)
        pr = np.exp(vals - vals[:, :1]); c = np.cumsum(pr / pr.sum(-1, keepdims=True), -1)
        j = np.minimum((c < uu[:, s:s + 1]).sum(-1), topk - 1)
        same += int((idx[np.arange(B), j] == out[:, s]).sum())
    print(f"  logits rel err per step (teacher-forced on the kernel's codes): {[round(x, 4) for x in rel]}")
    print(f"  sampler: kernel codes == numpy multinomial top-k on the kernel's own logits: {same}/{out.size}")
    print(f"B={B} ns={ns}: codes match {m * 100:.1f} % ({int((out == rc).sum())}/{out.size}); "
          f"mismatches not explained by a CDF near-tie (tol {TOL}): {hard}")
    print(" kernel", out.tolist()); print(" ref   ", rc.tolist())


if __name__ == "__main__":
    main()
