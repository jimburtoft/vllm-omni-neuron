"""Profile the Moshi LM frame on one NeuronCore: where do the 44 ms (B=1) go?

Variants (each compiled separately, timed over N iterations after warmup):
  full        MoshiFrame (temporal + text sample + depformer + audio sample)
  temporal    temporal step + text head only
  depformer   depformer (8 steps) + sampling only (input: tout [B,1,4096], text [B])
  temporal_nowrite   temporal without the KV index_copy_ (measures the KV write cost)
Context buckets: ctx in {3000 (full ring), 1024, 256} for full / temporal.

  python prof_lm.py --B 1 --what full,temporal,depformer --ctx 3000,256
"""
import argparse
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths  # noqa: E402
from moshi_neuron.lm import MoshiFrame, MoshiLM, sample  # noqa: E402
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


class Temporal(nn.Module):
    def __init__(self, lm, ctx, write=True):
        super().__init__()
        self.lm, self.ctx, self.write = lm, ctx, write

    def forward(self, tokens, reset, active):
        if not self.write:
            self.lm._skip_write = True
        tout, tl = self.lm.temporal(tokens, reset, active, self.ctx)
        return tout, torch.argmax(tl, -1)


class Depformer(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.lm = lm

    def forward(self, tout, text, ga):
        codes, _ = self.lm.depformer(tout, text, ga, 0.8, 250)
        return codes


def bench(fn, args, n, label):
    for _ in range(3):
        out = fn(*args)
    (out[0] if isinstance(out, tuple) else out).cpu()
    t0 = time.perf_counter()
    for _ in range(n):
        out = fn(*args)
        (out[0] if isinstance(out, tuple) else out).cpu()
    ms = (time.perf_counter() - t0) / n * 1e3
    print(f"{label:40s} {ms:8.2f} ms", flush=True)
    return ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--what", default="full,temporal,depformer")
    ap.add_argument("--ctx", default="3000,256")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--abl", default="", help="comma list: simplemask,norope,noattn,nomlp (temporal only)")
    args = ap.parse_args()
    neuron_runtime_env()
    dev = torch_device()
    cf = compile_fn()
    lm_path, _, _ = checkpoint_paths()
    B = args.B
    lm = MoshiLM(lm_path, B).to(dev).eval()
    lm._abl = args.abl
    tok = torch.full((B, 17), 5, dtype=torch.long).to(dev)
    r = torch.zeros(B, dtype=torch.bool).to(dev)
    a = torch.ones(B, dtype=torch.bool).to(dev)
    gt = torch.zeros(B, 32000).to(dev)
    ga = torch.zeros(B, 8, 2048).to(dev)
    ft = torch.zeros(B, dtype=torch.long).to(dev)
    fa = torch.zeros(B, 8, dtype=torch.long).to(dev)
    uf = torch.zeros(B, dtype=torch.bool).to(dev)
    res = {}
    with torch.inference_mode():
        for w in args.what.split(","):
            for c in ([int(x) for x in args.ctx.split(",")] if w in ("full", "temporal", "temporal_nowrite") else [0]):
                if w == "full":
                    fn = cf(MoshiFrame(lm, ctx=c).to(dev))
                    res[(w, c)] = bench(fn, (tok, r, a, gt, ga, ft, fa, uf), args.n, f"B={B} full ctx={c}")
                elif w in ("temporal", "temporal_nowrite"):
                    fn = cf(Temporal(lm, c, write=(w == "temporal")).to(dev))
                    res[(w, c)] = bench(fn, (tok, r, a), args.n, f"B={B} {w} ctx={c} abl={args.abl}")
                    lm._skip_write = False
                elif w == "depformer":
                    fn = cf(Depformer(lm).to(dev))
                    tout = torch.zeros(B, 1, 4096, dtype=torch.bfloat16).to(dev)
                    res[(w, c)] = bench(fn, (tout, ft, ga), args.n, f"B={B} depformer")
    print(res)


if __name__ == "__main__":
    main()
