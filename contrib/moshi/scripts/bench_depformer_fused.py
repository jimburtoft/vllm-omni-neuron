"""Hardware: fused all-steps depformer (depformer_all) vs the per-step NKI kernel + torch glue, B rows, NS steps.
Times both (wall, 50 calls) and compares codes; the fused kernel's sampler uses the same multinomial top-k with the
same uniforms. usage: python bench_depformer_fused.py [B=18] [NS=8]"""
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 18
    NS = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    neuron_runtime_env(); dev = torch_device(); cf = compile_fn()
    from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
    from moshi_neuron.kernels.depformer_nki import D, FF, L, depformer_step
    from moshi_neuron.kernels.depformer_fused_nki import CARD, depformer_all
    from moshi_neuron.lm import sample
    Q, TEMP, TOPK = 8, 0.8, int(os.environ.get("TOPK", "248"))
    g = torch.Generator().manual_seed(0)
    r = lambda *sh, sc=0.02: (torch.randn(*sh, generator=g) * sc).to(torch.bfloat16)
    din = r(Q, B, D, sc=1.0); prev0 = r(B, D, sc=1.0)
    uu = torch.rand(B, NS, generator=g)
    wqkvT, woT = r(Q, L, D, 3 * D), r(Q, L, D, D)
    wgT, wuT, wdT = r(Q, L, D, FF), r(Q, L, D, FF), r(Q, L, FF, D)
    n1 = 1 + 0.1 * torch.randn(L, D, generator=g); n2 = 1 + 0.1 * torch.randn(L, D, generator=g)
    headT = r(Q, D, CARD, sc=0.05); emb = r(Q - 1, CARD + 1, D, sc=1.0)
    grid = int(os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2"))
    kstep, kall = wrap_nki(depformer_step), wrap_nki(depformer_all)
    tens = [t.to(dev) for t in (din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb)]

    class Fused(nn.Module):
        def forward(self, din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb):
            return kall[grid](din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb, NS, Q, TEMP, TOPK, 1e-8,
                              int(os.environ.get("FUSED_VER", "1")))

    class PerStep(nn.Module):
        def forward(self, din, prev0, uu, wqkvT, woT, wgT, wuT, wdT, n1, n2, headT, emb):
            kc = torch.zeros(L, B, NS, D, dtype=torch.bfloat16, device=din.device)
            vc = torch.zeros(L, B, NS, D, dtype=torch.bfloat16, device=din.device)
            prev = prev0
            codes = []
            for s_ in range(NS):
                s = s_ % Q
                x = (din[s].float() + prev.float()).to(torch.bfloat16).contiguous()
                y, kn, vn = kstep[grid](x, kc, vc, wqkvT[s], woT[s], wgT[s], wuT[s], wdT[s], n1, n2, s_, 1e-8, 4, 0)
                if s_ + 1 < NS:
                    kc = torch.cat([kc[:, :, :s_], kn.unsqueeze(2), kc[:, :, s_ + 1:]], 2)
                    vc = torch.cat([vc[:, :, :s_], vn.unsqueeze(2), vc[:, :, s_ + 1:]], 2)
                lg = (y @ headT[s]).float()
                c = sample(lg, uu[:, s_:s_ + 1], TEMP, TOPK)
                codes.append(c)
                if s_ + 1 < NS:
                    prev = F.embedding(c, emb[min(s, Q - 2)])
            return torch.stack(codes, 1)

    res = {}
    with torch.inference_mode():
        for name, m in (("fused", Fused()), ("perstep", PerStep())):
            fn = cf(m)
            for _ in range(3):
                out = fn(*tens); out.cpu()
            t0 = time.perf_counter()
            for _ in range(50):
                out = fn(*tens)
            out.cpu()
            res[name] = ((time.perf_counter() - t0) / 50 * 1e3, out.cpu().long())
    a, b = res["fused"][1], res["perstep"][1]
    print(f"B={B} NS={NS} topk={TOPK}: fused {res['fused'][0]:.3f} ms vs per-step {res['perstep'][0]:.3f} ms; "
          f"codes equal {float((a == b).float().mean()) * 100:.1f} % (step 0: {float((a[:, 0] == b[:, 0]).float().mean()) * 100:.1f} %)",
          flush=True)


if __name__ == "__main__":
    main()
