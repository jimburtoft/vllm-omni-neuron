"""The rotate-half RoPE on permuted Q/K rows gives the same attention scores as interleaved RoPE (CPU, fp32)."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.mimi import apply_rope_interleaved, rope_tables  # noqa: E402


def test():
    torch.manual_seed(0)
    B, H, T, D, S = 2, 4, 1, 128, 7
    q, k = torch.randn(B, H, T, D), torch.randn(B, H, S, D)
    pq = torch.tensor([[5], [123]])
    pk = torch.arange(S).view(1, S).expand(B, S) + torch.tensor([[0], [100]])
    cq, sq = rope_tables(pq, D, 10000.0)
    ck, sk = rope_tables(pk, D, 10000.0)
    ref = apply_rope_interleaved(q, cq, sq) @ apply_rope_interleaved(k, ck, sk).transpose(-1, -2)
    perm = torch.cat([torch.arange(0, D, 2), torch.arange(1, D, 2)])
    qp, kp = q[..., perm], k[..., perm]

    def half(x, c, s):
        c2, s2 = torch.cat([c, c], -1), torch.cat([-s, s], -1)
        return x * c2 + torch.cat([x[..., D // 2:], x[..., :D // 2]], -1) * s2

    out = half(qp, cq, sq) @ half(kp, ck, sk).transpose(-1, -2)
    err = (out - ref).abs().max().item()
    print("max abs err", err)
    assert err < 1e-4


if __name__ == "__main__":
    test()
