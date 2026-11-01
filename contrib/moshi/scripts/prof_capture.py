"""Compile + run the Moshi stage-0 frame graph (B rows) and the Mimi graphs, so their NEFFs land in the compile cache,
then print the NEFF paths created by this run (newest first) for neuron-explorer capture.

  python prof_capture.py --B 5 --mark /tmp/mark
"""
import argparse
import glob
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths  # noqa: E402
from moshi_neuron.lm import MoshiFrame, MoshiLM  # noqa: E402
from moshi_neuron.mimi import MimiStreaming  # noqa: E402
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=5)
    ap.add_argument("--what", default="lm,mimi")
    a = ap.parse_args()
    neuron_runtime_env()
    dev, cf = torch_device(), compile_fn()
    lm_path, mimi_path, _ = checkpoint_paths()
    B = a.B
    t_start = time.time()
    with torch.inference_mode():
        if "mimi" in a.what:
            m = MimiStreaming(mimi_path, B, dev, compile_fn=cf)
            x = torch.randn(B, 1920) * 0.1
            c = torch.randint(0, 2048, (B, 8))
            for _ in range(5):
                m.encode(x).cpu(); m.decode(c).cpu()
        if "lm" in a.what:
            lm = MoshiLM(lm_path, B).to(dev).eval()
            fn = cf(MoshiFrame(lm, ctx=lm.C).to(dev))
            tok = torch.full((B, 17), 5, dtype=torch.long).to(dev)
            r = torch.zeros(B, dtype=torch.bool).to(dev); act = ~r
            args = (tok, r, act, torch.zeros(B, 32000).to(dev), torch.zeros(B, 8, 2048).to(dev),
                    torch.zeros(B, dtype=torch.long).to(dev), torch.zeros(B, 8, dtype=torch.long).to(dev), r)
            for _ in range(5):
                fn(*args).cpu()
            t0 = time.perf_counter()
            for _ in range(20):
                fn(*args).cpu()
            print(f"frame B={B}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms wall", flush=True)
    roots = [os.path.expanduser("~/.cache"), "/var/tmp", "/tmp"]
    neffs = []
    for r in roots:
        neffs += [p for p in glob.glob(os.path.join(r, "**", "*.neff"), recursive=True) if os.path.getmtime(p) >= t_start - 5]
    for p in sorted(neffs, key=os.path.getsize, reverse=True):
        print("NEFF", os.path.getsize(p), p)


if __name__ == "__main__":
    main()
