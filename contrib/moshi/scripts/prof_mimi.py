"""Mimi encoder/decoder per-frame timing vs B (fp32 and bf16), on one core.

  python prof_mimi.py --B 1,3,5 --dtype float32
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths  # noqa: E402
from moshi_neuron.mimi import MimiStreaming  # noqa: E402
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", default="1,3,5")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--n", type=int, default=50)
    a = ap.parse_args()
    neuron_runtime_env()
    dev, cf = torch_device(), compile_fn()
    _, mp, _ = checkpoint_paths()
    for B in [int(x) for x in a.B.split(",")]:
        m = MimiStreaming(mp, B, dev, dtype=getattr(torch, a.dtype), compile_fn=cf)
        x = torch.randn(B, 1920) * 0.1
        c = torch.randint(0, 2048, (B, 8))
        with torch.inference_mode():
            for _ in range(3):
                m.encode(x).cpu(); m.decode(c).cpu()
            t0 = time.perf_counter()
            for _ in range(a.n):
                m.encode(x).cpu()
            te = (time.perf_counter() - t0) / a.n * 1e3
            t0 = time.perf_counter()
            for _ in range(a.n):
                m.decode(c).cpu()
            td = (time.perf_counter() - t0) / a.n * 1e3
        print(f"B={B} {a.dtype}: encode {te:.2f} ms, decode {td:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
