"""Microbenchmark: Helium decode attention over a C-slot KV cache, B rows, H=32 heads, D=128, one layer x L.

Variants:
  mm      current: K/V stored [B,H,C,D], scores = q @ K^T (contraction over D, the free dim of K)
  kt      K stored transposed [B,H,D,C]: scores = q @ K_T (contraction over D on the partition dim)
  vec     elementwise: scores = (q * K).sum(-1) on VectorE, out = (p * V).sum over C
  nki     nkilib attention_tkg via vllm_neuron NF.attention_decode (block KV, in-kernel KV write)
Each variant also writes the new k/v (index_copy_ into the flat cache, or the kernel's in-kernel write).

  python attn_bench.py --B 3 --C 3000 --L 4 --what mm,kt,vec
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402

NEG = -1e30


class Attn(nn.Module):
    def __init__(self, form, B, H, D, C, L, dt=torch.bfloat16):
        super().__init__()
        self.form, self.B, self.H, self.D, self.C, self.L = form, B, H, D, C, L
        for i in range(L):
            if form == "kt":
                self.register_buffer(f"k{i}", torch.randn(B, H, D, C, dtype=dt) * 0.1, persistent=False)
            else:
                self.register_buffer(f"k{i}", torch.randn(B * H * C, D, dtype=dt) * 0.1, persistent=False)
            self.register_buffer(f"v{i}", torch.randn(B * H * C, D, dtype=dt) * 0.1, persistent=False)
        self.register_buffer("hoff", torch.arange(H), persistent=False)
        self.register_buffer("cidx", torch.arange(C), persistent=False)

    def forward(self, q, k, v, off):
        # q/k/v [B,H,1,D]; off [B] long (write slot)
        B, H, D, C = self.B, self.H, self.D, self.C
        mask = torch.where(self.cidx.view(1, C) <= off.view(B, 1), 0.0, NEG).float().view(B, 1, 1, C)
        widx = ((torch.arange(B, device=q.device).view(B, 1) * H + self.hoff.view(1, H)) * C + off.view(B, 1)).reshape(-1)
        acc = torch.zeros_like(q)
        for i in range(self.L):
            kb, vb = getattr(self, f"k{i}"), getattr(self, f"v{i}")
            vb.index_copy_(0, widx, v.reshape(-1, D))
            V = vb.view(B, H, C, D)
            if self.form == "kt":
                # write column `off` of K^T: one-hot blend restricted to the written column via index_copy on a
                # [B*H*D, C] view is not expressible per-row, so use scatter_ along the last dim (B*H*D elements)
                kb.scatter_(3, off.view(B, 1, 1, 1).expand(B, H, D, 1), k.transpose(-1, -2))
                sc = (q @ kb).float() * D ** -0.5 + mask
            else:
                kb.index_copy_(0, widx, k.reshape(-1, D))
                K = kb.view(B, H, C, D)
                if self.form == "vec":
                    sc = (q * K).float().sum(-1, keepdim=True).transpose(-1, -2) * D ** -0.5 + mask
                else:
                    sc = (q @ K.transpose(-1, -2)).float() * D ** -0.5 + mask
            p = torch.softmax(sc, -1).to(V.dtype)
            if self.form == "vec":
                o = (p.transpose(-1, -2) * V).sum(2, keepdim=True)
            else:
                o = p @ V
            acc = acc + o
        return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=3)
    ap.add_argument("--C", type=int, default=3000)
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--what", default="mm,kt,vec")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    neuron_runtime_env()
    dev, cf = torch_device(), compile_fn()
    B, H, D, C = a.B, 32, 128, a.C
    q = (torch.randn(B, H, 1, D) * 0.1).to(torch.bfloat16).to(dev)
    k, v = q.clone(), q.clone()
    off = torch.full((B,), C // 2, dtype=torch.long).to(dev)
    ref = None
    with torch.inference_mode():
        for w in a.what.split(","):
            torch.manual_seed(0)
            m = Attn(w, B, H, D, C, a.L).to(dev)
            fn = cf(m)
            try:
                for _ in range(3):
                    out = fn(q, k, v, off)
                out.cpu()
                t0 = time.perf_counter()
                for _ in range(a.n):
                    out = fn(q, k, v, off)
                o = out.cpu()
                ms = (time.perf_counter() - t0) / a.n * 1e3
                print(f"{w:5s} B={B} C={C} L={a.L}: {ms:7.3f} ms total, {ms / a.L:6.3f} ms/layer", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"{w}: FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)


if __name__ == "__main__":
    main()
