"""TP1 LM-frame graph at B rows with the window full (offsets = 1600 > context) for neuron-explorer capture.
Prints wall time per step and the NEFF paths produced/used by this run. Same per-rank attention shape as TP4 at 4B rows.
  python prof_lm_tp1.py --B 10 --mark ~/mark_tp1"""
import argparse, glob, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.lm import MoshiFrame, MoshiLM
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=10)
    ap.add_argument("--abl", default="")
    a = ap.parse_args()
    neuron_runtime_env()
    dev, cf = torch_device(), compile_fn()
    B = a.B
    t_start = time.time()
    lm = MoshiLM(os.environ["MOSHI_WEIGHTS"], B).to(dev).eval()
    lm._abl = a.abl
    ctx = getattr(lm, "SCTX", lm.C)
    fn = cf(MoshiFrame(lm, ctx=ctx).to(dev))
    with torch.inference_mode():
        lm.offset.copy_(torch.full_like(lm.offset, 1600))
        tok = torch.full((B, 17), 5, dtype=torch.long).to(dev)
        r = torch.zeros(B, dtype=torch.bool).to(dev); act = ~r
        args = [tok, r, act, torch.zeros(B, 32000).to(dev), torch.zeros(B, 8, 2048).to(dev),
                torch.zeros(B, dtype=torch.long).to(dev), torch.zeros(B, 8, dtype=torch.long).to(dev), r]
        if os.environ.get("MOSHI_EMB_INPUTS") == "1":
            args += [torch.zeros(B, 4096, dtype=torch.bfloat16).to(dev), r]
        for _ in range(5):
            fn(*args).cpu() if not isinstance(fn(*args), tuple) else None
        t0 = time.perf_counter()
        for _ in range(20):
            o = fn(*args)
            (o[0] if isinstance(o, tuple) else o).cpu()
        print(f"LM frame TP1 B={B} abl={a.abl!r}: {(time.perf_counter() - t0) / 20 * 1e3:.2f} ms wall", flush=True)
    neffs = []
    for root in [os.path.expanduser("~/.cache/neuron_libtorch"), "/var/tmp"]:
        neffs += [p for p in glob.glob(os.path.join(root, "**", "*.neff"), recursive=True) if os.path.getatime(p) >= t_start - 5]
    for p in sorted(neffs, key=os.path.getsize, reverse=True)[:4]:
        print("NEFF", os.path.getsize(p), p, flush=True)


if __name__ == "__main__":
    main()
