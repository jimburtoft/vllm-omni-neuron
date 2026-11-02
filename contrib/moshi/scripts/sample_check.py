"""Is on-device top-k + Gumbel sampling correct at batch B? Compare the sampled-code distribution on device vs a
CPU reference with identical logits and noise, for B rows. (Isolates the sampler from the model.)"""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device
from moshi_neuron.lm import sample
import torch.nn as nn


class S(nn.Module):
    def __init__(self, temp, k):
        super().__init__(); self.t, self.k = temp, k
    def forward(self, l, g):
        return sample(l, g, self.t, self.k)


def main():
    neuron_runtime_env(); dev = torch_device(); cf = compile_fn()
    torch.manual_seed(0)
    for B, V, k, T in ((1, 2048, 250, 0.8), (5, 2048, 250, 0.8), (8, 2048, 250, 0.8), (5, 32000, 25, 0.7), (5, 8 * 2048, 250, 0.8)):
        shape = (B, V) if V != 8 * 2048 else (B, 8, 2048)
        l = torch.randn(*shape) * 3
        g = -torch.log(-torch.log(torch.rand(*shape).clamp(1e-10, 1 - 1e-7)))
        ref = sample(l, g, T, k)
        fn = cf(S(T, k).to(dev))
        got = fn(l.to(dev), g.to(dev)).cpu()
        print(f"shape {tuple(shape)} k={k}: match {float((got == ref).float().mean()):.4f}", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
