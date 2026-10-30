"""Neuron (trn2) parity + timing for moshi_neuron against the CPU goldens.

  python test_neuron.py --what mimi            # Mimi encode/decode graphs (fp32 and bf16)
  python test_neuron.py --what lm --frames 40  # Helium + depformer frame graph, teacher-forced (bf16)
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths  # noqa: E402
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402
from test_cpu_equivalence import test_lm, test_mimi  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default=os.path.expanduser("~/goldens"))
    ap.add_argument("--what", default="mimi")
    ap.add_argument("--mimi-dtype", default="float32")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--clip", default="utt1")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--golden-dtype", default="float32")
    args = ap.parse_args()
    neuron_runtime_env()
    dev = torch_device()
    lm_path, mimi_path, _ = checkpoint_paths()
    cf = compile_fn()
    if "mimi" in args.what:
        test_mimi(args.goldens, mimi_path, getattr(torch, args.mimi_dtype), device=dev, compile_fn=cf)
    if "lm" in args.what:
        test_lm(args.goldens, lm_path, getattr(torch, args.dtype), args.clip, args.frames, device=dev,
                compile_fn=cf, golden_dtype=args.golden_dtype)


if __name__ == "__main__":
    with torch.inference_mode():
        main()
