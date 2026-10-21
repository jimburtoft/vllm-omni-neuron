"""Per-row sampling: greedy rows are exact, top-k is respected, the sampled distribution matches softmax(l/T) on top-k."""
import torch

from qwen3_tts_neuron.fused import TOPK_MAX, sample_rows


def gumbel(*shape, g):
    return -torch.log(-torch.log(torch.rand(*shape, generator=g).clamp_(1e-10, 1 - 1e-7)))


def test_greedy_rows_exact():
    g = torch.Generator().manual_seed(0)
    l = torch.randn(6, 3072, generator=g)
    out = sample_rows(l, gumbel(6, 3072, g=g), torch.zeros(6), torch.zeros(6, dtype=torch.long))
    assert torch.equal(out, l.argmax(-1))


def test_topk_and_distribution():
    g = torch.Generator().manual_seed(1)
    V, N, T, K = 64, 20000, 0.9, 5
    l = torch.randn(1, V, generator=g).expand(N, V).contiguous()
    out = sample_rows(l, gumbel(N, V, g=g), torch.full((N,), T), torch.full((N,), K, dtype=torch.long))
    top = set(torch.topk(l[0], K).indices.tolist())
    assert set(out.tolist()) <= top
    p = torch.softmax(torch.topk(l[0] / T, K).values, -1)
    idx = torch.topk(l[0], K).indices
    emp = torch.stack([(out == i).float().mean() for i in idx])
    assert (emp - p).abs().max() < 0.02


def test_mixed_batch_and_3d():
    g = torch.Generator().manual_seed(2)
    l = torch.randn(4, 1, 2048, generator=g)
    temp = torch.tensor([0.0, 0.9, 0.0, 1.2]); topk = torch.tensor([0, 50, 7, 0])
    out = sample_rows(l, gumbel(4, 1, 2048, g=g), temp, topk)
    assert out.shape == (4, 1)
    assert out[0, 0] == l[0, 0].argmax() and out[2, 0] == l[2, 0].argmax()
    assert out[1, 0] in torch.topk(l[1, 0], 50).indices
    assert TOPK_MAX >= 50
