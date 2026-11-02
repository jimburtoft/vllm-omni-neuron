"""Sampled-mode trajectory check vs the kyutai CPU reference is not possible (different RNG), so instead: run the
Neuron frame graph sampled at B rows with *identical* inputs and identical noise across rows -- every row must then
produce the same token stream. Row divergence = a batch-dependent numerical defect."""
import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths
from moshi_neuron.lm import LMGenHost, MoshiFrame, MoshiLM
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device


def main():
    B, N = int(sys.argv[1]), int(sys.argv[2])
    neuron_runtime_env(); dev = torch_device(); cf = compile_fn()
    lm_path, _, _ = checkpoint_paths()
    lm = MoshiLM(lm_path, B).to(dev).eval()
    fn = cf(MoshiFrame(lm, ctx=lm.C, debug=True).to(dev))
    g = torch.load(os.path.expanduser("~/goldens/duplex_utt2_float32_greedy.pt"))
    host = LMGenHost(B)
    gen = torch.Generator().manual_seed(3)
    z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
    div, first = 0, None
    with torch.inference_mode():
        for t in range(N):
            inp = host.inputs(g["user_codes"][t].view(1, -1).expand(B, -1).contiguous())
            u1 = torch.rand(1, 32000, generator=gen).clamp(1e-10, 1 - 1e-7); u2 = torch.rand(1, 8, 2048, generator=gen).clamp(1e-10, 1 - 1e-7)
            gt = (-torch.log(-torch.log(u1))).expand(B, -1).contiguous().to(dev)
            ga = (-torch.log(-torch.log(u2))).expand(B, -1, -1).contiguous().to(dev)
            out, tl, al, _ = fn(inp.to(dev), torch.full((B,), t == 0, device=dev), torch.ones(B, dtype=torch.bool, device=dev),
                                gt, ga, z(B, dtype=torch.long), z(B, 8, dtype=torch.long), z(B, dtype=torch.bool))
            out = out.cpu(); host.commit(out)
            d = int((out != out[0:1]).any(1).sum())
            if d and first is None:
                first = t
                tl = tl.cpu(); al = al.cpu()
                print(f"t={t}: rows differ {d}; text logit max|diff| row0 vs others {float((tl - tl[0:1]).abs().max()):.4f}, "
                      f"dep {float((al - al[0:1]).abs().max()):.4f}", flush=True)
            div += d
    print(f"B={B}: total row-divergent steps {div} (first at {first})", flush=True)


if __name__ == "__main__":
    main()
