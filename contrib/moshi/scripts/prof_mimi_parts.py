"""Mimi encoder breakdown at B rows: SEANet convs only / + transformer / + downsample / + RVQ (full)."""
import argparse, os, sys, time
import torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths
from moshi_neuron.mimi import MimiEncoder, elu
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device


class Part(nn.Module):
    def __init__(self, enc, upto):
        super().__init__(); self.e, self.upto = enc, upto
    def forward(self, pcm, reset, active):
        e = self.e
        x = e.seanet[0](pcm, reset, active)
        for j in range(1, len(e.seanet), 2):
            x = e.seanet[j](x, reset, active); x = e.seanet[j + 1](elu(x), reset, active)
        x = e.last(elu(x), reset, active)
        if self.upto == "seanet":
            return x
        x = e.transformer(x, reset, active)
        if self.upto == "transformer":
            return x
        return e.down(x, reset, active)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=7); ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args(); neuron_runtime_env(); dev, cf = torch_device(), compile_fn()
    _, mp, _ = checkpoint_paths()
    enc = MimiEncoder(mp, a.B).to(dev).eval()
    x = (torch.randn(a.B, 1, 1920) * 0.1).to(dev); r = torch.zeros(a.B, dtype=torch.bool, device=dev); act = ~r
    with torch.inference_mode():
        for name, m in (("seanet", Part(enc, "seanet")), ("transformer", Part(enc, "transformer")),
                        ("down", Part(enc, "down")), ("full(+rvq)", enc)):
            fn = cf(m)
            for _ in range(3): fn(x, r, act).cpu()
            t0 = time.perf_counter()
            for _ in range(a.n): fn(x, r, act).cpu()
            print(f"B={a.B} enc upto {name}: {(time.perf_counter() - t0) / a.n * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
