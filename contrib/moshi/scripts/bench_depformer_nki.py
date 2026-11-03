"""Hardware: NKI depformer_step (6 layers of one step) vs the torch path, per step, B rows; correctness vs torch."""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    s = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    neuron_runtime_env(); dev = torch_device(); cf = compile_fn()
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from moshi_neuron.kernels.depformer_nki import D, FF, L, depformer_step
    g = torch.Generator().manual_seed(0)
    r = lambda *sh, sc=0.02: (torch.randn(*sh, generator=g) * sc).to(torch.bfloat16)
    x = r(B, D, sc=1.0)
    kc, vc = r(L, B, 8, D, sc=1.0), r(L, B, 8, D, sc=1.0)
    wqkvT, woT, wgT, wuT, wdT = r(L, D, 3 * D), r(L, D, D), r(L, D, FF), r(L, D, FF), r(L, FF, D)
    n1 = 1 + 0.1 * torch.randn(L, D, generator=g); n2 = 1 + 0.1 * torch.randn(L, D, generator=g)
    args = [t.to(dev) for t in (x, kc, vc, wqkvT, woT, wgT, wuT, wdT, n1, n2)]
    grid = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    k = wrap_nki(depformer_step)

    class NK(nn.Module):
        def forward(self, *a):
            return k[grid](*a, s, 1e-8, int(os.environ.get('DEP_VER', '2')))

    class TorchStep(nn.Module):
        def forward(self, x, kc, vc, wqkvT, woT, wgT, wuT, wdT, n1, n2):
            xr = x.float()
            ks, vs = [], []
            for l in range(L):
                h = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-8) * n1[l]).to(torch.bfloat16)
                qkv = h @ wqkvT[l]
                q, kk, v = qkv[:, :D], qkv[:, D:2 * D], qkv[:, 2 * D:]
                ks.append(kk); vs.append(v)
                K = torch.cat([kc[l, :, :s], kk[:, None]], 1).view(B, s + 1, 16, 64)
                V = torch.cat([vc[l, :, :s], v[:, None]], 1).view(B, s + 1, 16, 64)
                sc_ = torch.einsum("bhd,bjhd->bhj", q.view(B, 16, 64).float(), K.float()) * 0.125
                p = torch.softmax(sc_, -1)
                o = torch.einsum("bhj,bjhd->bhd", p, V.float()).reshape(B, D).to(torch.bfloat16)
                xr = xr + (o @ woT[l]).float()
                h = (xr * torch.rsqrt(xr.pow(2).mean(-1, keepdim=True) + 1e-8) * n2[l]).to(torch.bfloat16)
                xr = xr + ((F.silu(h @ wgT[l]) * (h @ wuT[l])) @ wdT[l]).float()
            return xr.to(torch.bfloat16), torch.stack(ks), torch.stack(vs)

    fn_k, fn_t = cf(NK()), cf(TorchStep())
    with torch.inference_mode():
        res = {}
        for name, fn in (("nki", fn_k), ("torch", fn_t)):
            for _ in range(3):
                out = fn(*args); out[0].cpu()
            t0 = time.perf_counter()
            for _ in range(50):
                out = fn(*args)
            out[0].cpu()
            res[name] = ((time.perf_counter() - t0) / 50 * 1e3, [o.cpu().float() for o in out])
    yk, yt = res["nki"][1][0], res["torch"][1][0]
    rel = float((yk - yt).norm() / yt.norm())
    print(f"B={B} s={s}: NKI step {res['nki'][0]:.3f} ms vs torch step {res['torch'][0]:.3f} ms (x8 steps per frame); "
          f"y rel diff {rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
