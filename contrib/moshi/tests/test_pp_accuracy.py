"""PersonaPlex accuracy vs the NVIDIA reference (CPU fp32 goldens from scripts/make_pp_goldens.py), TP1 on Neuron.

Replays the reference's system prompt (voice prompt + silences + text prompt: the exact transformer input embeddings
it consumed, via emb_in), then every user frame teacher-forced on the reference's transformer input tokens, and
reports per-frame argmax agreement of the text logits and the 8 agent depformer logits, plus logit rel. error.
usage: python tests/test_pp_accuracy.py --golden ~/pp_goldens/pp_utt2_float32_greedy.pt
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-9))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--frames", type=int, default=10000)
    ap.add_argument("--no-prompt", action="store_true", help="skip the system prompt (control)")
    args = ap.parse_args()
    from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device
    from moshi_neuron.lm import MoshiLM, MoshiFrame
    from moshi_neuron.config import checkpoint_paths
    neuron_runtime_env()
    dev = torch_device()
    g = torch.load(args.golden)
    lm_path = os.environ["MOSHI_WEIGHTS"]
    lm = MoshiLM(lm_path, 1, dtype=torch.bfloat16).to(dev).eval()
    ctx = getattr(lm, "SCTX", lm.C)
    fn = compile_fn()(MoshiFrame(lm, ctx=ctx, temp_text=0.0, temp=0.0, debug=True).to(dev))
    z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
    one = torch.ones(1, dtype=torch.bool, device=dev)
    P = 0 if args.no_prompt else g["prompt_emb"].shape[0]
    T = min(args.frames, g["inputs"].shape[0])
    stats = dict(text_top1=[], dep_top1=[], text_rel=[], dep_rel=[])
    dummy_tok = torch.zeros(1, 17, dtype=torch.long, device=dev)
    with torch.inference_mode():
        for p in range(P):
            r = fn(dummy_tok, torch.tensor([p == 0], device=dev), one, z(1, 32000), z(1, 8, 2048),
               z(1, dtype=torch.long), z(1, 8, dtype=torch.long), one,
               g["prompt_emb"][p].view(1, -1).to(dev), one)
            r[0].cpu()  # sync: avoid "Execution Queue Full"
        ft = g["text_logits"].argmax(-1)
        fa = g["dep_logits"].argmax(-1)
        for t in range(T):
            inp = g["inputs"][t].view(1, -1).to(dev)
            out, tl, al, _ = fn(inp, torch.tensor([P == 0 and t == 0], device=dev), one, z(1, 32000), z(1, 8, 2048),
                                ft[t].view(1).to(dev), fa[t].view(1, 8).to(dev), one,
                                z(1, 4096, dtype=torch.bfloat16), torch.zeros(1, dtype=torch.bool, device=dev))
            tl, al = tl.cpu()[0], al.cpu()[0]
            stats["text_top1"].append(int(tl.argmax() == ft[t]))
            stats["dep_top1"].append(float((al.argmax(-1) == fa[t]).float().mean()))
            stats["text_rel"].append(rel(tl, g["text_logits"][t]))
            stats["dep_rel"].append(rel(al, g["dep_logits"][t]))
            if t < 4 or t % 60 == 0:
                print(f"  t={t} text_top1 {np.mean(stats['text_top1']):.3f} dep_top1 {np.mean(stats['dep_top1']):.3f} "
                      f"text_rel {stats['text_rel'][-1]:.3f}", flush=True)
    s = {k: float(np.mean(v)) for k, v in stats.items()}
    s.update(text_rel_max=float(np.max(stats["text_rel"])), frames=T, prompt_frames=P)
    print("PP_ACC", s, flush=True)


if __name__ == "__main__":
    main()
