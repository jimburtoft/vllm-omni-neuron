"""Long-session check (> 3000 frames): run the LM frame graph teacher-free (greedy) for N frames on silence + golden
user audio looped, and compare the fused-kernel block ring against the torch flat ring step by step.

Both paths implement moshi's RingKVCache window (last 2999 frames). Up to frame 3000 they both match the golden; past
it the reference itself is the only spec, so we check the two independent implementations against each other
(teacher-forced on the same tokens: argmax agreement of text and depformer logits per frame).
  python long_ring_test.py --frames 3600
"""
import argparse, os, sys, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths
from moshi_neuron.lm import LMGenHost, MoshiFrame, MoshiLM
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=3600)
    ap.add_argument("--goldens", default=os.path.expanduser("~/goldens"))
    ap.add_argument("--mode", default=os.environ.get("MOSHI_KERNELS", "1"))
    ap.add_argument("--out", default="ring_trace.pt")
    ap.add_argument("--force", default="", help="trace .pt from another run: force its tokens, record agreement")
    a = ap.parse_args()
    neuron_runtime_env(); dev = torch_device(); cf = compile_fn()
    lm_path, _, _ = checkpoint_paths()
    lm = MoshiLM(lm_path, 1).to(dev).eval()
    fn = cf(MoshiFrame(lm, ctx=lm.C, temp_text=0.0, temp=0.0, debug=True).to(dev))
    g = torch.load(os.path.join(a.goldens, "duplex_utt2_float32_greedy.pt"))
    uc = g["user_codes"]
    host = LMGenHost(1)
    z = lambda *s: torch.zeros(*s, device=dev)
    texts, audios = [], []
    forced = torch.load(a.force) if a.force else None
    agree_t, agree_a = [], []
    with torch.inference_mode():
        t0 = time.time()
        for t in range(a.frames):
            inp = host.inputs(uc[t % uc.shape[0]].view(1, -1))
            if forced is not None:
                ftx = forced["text"][t].view(1).to(dev); fau = forced["audio"][t].view(1, 8).to(dev)
                uf = torch.ones(1, dtype=torch.bool, device=dev)
            else:
                ftx = torch.zeros(1, dtype=torch.long, device=dev); fau = torch.zeros(1, 8, dtype=torch.long, device=dev)
                uf = torch.zeros(1, dtype=torch.bool, device=dev)
            out, tl, al, _ = fn(inp.to(dev), torch.tensor([t == 0], device=dev), torch.ones(1, dtype=torch.bool, device=dev),
                                z(1, 32000), z(1, 8, 2048), ftx, fau, uf)
            out = out.cpu()
            if forced is not None:
                agree_t.append(int(tl.cpu()[0].argmax()) == int(forced["text"][t]))
                agree_a.append(float((al.cpu()[0].argmax(-1) == forced["audio"][t]).float().mean()))
            host.commit(out)
            texts.append(int(out[0, 0])); audios.append(out[0, 1:].clone())
            if t % 500 == 0:
                print(f"t={t} {(time.time() - t0) / (t + 1) * 1e3:.1f} ms/frame", flush=True)
    torch.save({"text": torch.tensor(texts), "audio": torch.stack(audios)}, a.out)
    if forced is not None:
        import numpy as np
        for lo, hi in ((0, 1000), (1000, 2000), (2000, 2999), (2999, 3100), (3100, a.frames)):
            if hi <= len(agree_t):
                print(f"frames {lo}-{hi}: text top1 agree {np.mean(agree_t[lo:hi]):.4f}  audio {np.mean(agree_a[lo:hi]):.4f}")
    print("saved", a.out)


if __name__ == "__main__":
    main()
