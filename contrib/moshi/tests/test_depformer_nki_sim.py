"""Simulator test for kernels/depformer_nki.depformer_step vs a numpy reference (random weights, small B)."""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import nki  # noqa: E402

from moshi_neuron.kernels.depformer_nki import D, FF, HD, L, NH, depformer_step  # noqa: E402


def bf(x):
    import ml_dtypes
    return x.astype(ml_dtypes.bfloat16)


def ref(x, kc, vc, wqkvT, woT, wgT, wuT, wdT, n1, n2, s, eps):
    f = lambda a: a.astype(np.float32)
    x = f(x)
    B = x.shape[0]
    kn = np.zeros((L, B, D), np.float32); vn = np.zeros((L, B, D), np.float32)
    for l in range(L):
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * n1[l]
        h = f(bf(h))
        qkv = h @ f(wqkvT[l])
        q, k, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
        kn[l], vn[l] = k, v
        K = np.concatenate([f(kc[l, :, :s]), f(bf(k))[:, None]], 1)   # [B, s+1, D]
        V = np.concatenate([f(vc[l, :, :s]), f(bf(v))[:, None]], 1)
        qh = q.reshape(B, NH, HD); Kh = K.reshape(B, s + 1, NH, HD); Vh = V.reshape(B, s + 1, NH, HD)
        sc = np.einsum("bhd,bjhd->bhj", qh, Kh) * HD ** -0.5
        p = np.exp(sc - sc.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
        o = np.einsum("bhj,bjhd->bhd", p, Vh).reshape(B, D)
        x = x + f(bf(o)) @ f(woT[l])
        h = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * n2[l]
        h = f(bf(h))
        g = h @ f(wgT[l]); u = h @ f(wuT[l])
        m = (g / (1 + np.exp(-g))) * u
        x = x + f(bf(m)) @ f(wdT[l])
    return x, kn, vn


def main():
    rng = np.random.default_rng(0)
    B, s, eps = int(os.environ.get("B", "4")), int(os.environ.get("S", "3")), 1e-8
    r = lambda *sh, sc=0.02: bf(rng.standard_normal(sh).astype(np.float32) * sc)
    x = r(B, D, sc=1.0)
    kc, vc = r(L, B, 8, D, sc=1.0), r(L, B, 8, D, sc=1.0)
    wqkvT, woT = r(L, D, 3 * D), r(L, D, D)
    wgT, wuT, wdT = r(L, D, FF), r(L, D, FF), r(L, FF, D)
    n1 = (1 + 0.1 * rng.standard_normal((L, D))).astype(np.float32)
    n2 = (1 + 0.1 * rng.standard_normal((L, D))).astype(np.float32)
    y, kn, vn = nki.simulate(depformer_step)(x, kc, vc, wqkvT, woT, wgT, wuT, wdT, n1, n2, s, eps)
    yr, knr, vnr = ref(x, kc, vc, wqkvT, woT, wgT, wuT, wdT, n1, n2, s, eps)
    y = y.astype(np.float32)
    rel = lambda a, b: float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    print(f"B={B} s={s}: y rel {rel(y, yr):.4f}  k rel {rel(kn.astype(np.float32), knr):.4f}  "
          f"v rel {rel(vn.astype(np.float32), vnr):.4f}  cos {float((y * yr).sum() / np.linalg.norm(y) / np.linalg.norm(yr)):.5f}")


if __name__ == "__main__":
    main()
