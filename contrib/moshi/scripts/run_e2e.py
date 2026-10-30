"""End-to-end duplex run on Neuron (or CPU) over the golden clips.

  python run_e2e.py --goldens ~/goldens --clips utt1,utt2 --greedy        # parity vs fp32 greedy golden
  python run_e2e.py --goldens ~/goldens --clips utt1 --B 4                # 4 sessions in lockstep (sampled)

Writes out/<clip>_<mode>_b<row>.wav and .txt, and prints per-step timing (encode / LM / decode).
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import FRAME, SAMPLE_RATE  # noqa: E402
from moshi_neuron.engine import MoshiEngine, run_file  # noqa: E402
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default=os.path.expanduser("~/goldens"))
    ap.add_argument("--clips", default="utt1")
    ap.add_argument("--B", type=int, default=1)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    ap.add_argument("--repeat", type=int, default=1, help="run the clip set N times (timing warm)")
    args = ap.parse_args()
    neuron_runtime_env()
    dev = torch_device()
    eng = MoshiEngine(B=args.B, device=dev, compile_fn=compile_fn(), greedy=args.greedy, seed=args.seed)
    os.makedirs(args.out, exist_ok=True)
    import soundfile as sf
    summary = {}
    for rep in range(args.repeat):
        for clip in args.clips.split(","):
            wav = torch.from_numpy(np.load(os.path.join(args.goldens, f"clip_{clip}.npy")))
            eng.reset(range(args.B))
            res, tim = run_file(eng, [wav] * args.B)
            mode = "greedy" if args.greedy else "sampled"
            for b, (pcm, text) in enumerate(res):
                sf.write(os.path.join(args.out, f"{clip}_{mode}_b{b}.wav"), pcm.numpy(), SAMPLE_RATE)
                with open(os.path.join(args.out, f"{clip}_{mode}_b{b}.txt"), "w") as f:
                    f.write(text + "\n")
            st = tim[5:] if tim.shape[0] > 10 else tim
            s = dict(frames=int(tim.shape[0]), step_ms_p50=float(st[:, 0].median() * 1e3),
                     step_ms_p99=float(st[:, 0].quantile(0.99) * 1e3), step_ms_max=float(st[:, 0].max() * 1e3),
                     enc_ms=float(st[:, 1].median() * 1e3), lm_ms=float(st[:, 2].median() * 1e3),
                     dec_ms=float(st[:, 3].median() * 1e3), text_b0=res[0][1])
            g = os.path.join(args.goldens, f"duplex_{clip}_float32_greedy.pt")
            if args.greedy and os.path.exists(g):
                gold = torch.load(g)
                gt = "".join(eng.piece(t) for t in gold["tokens"][:, 0] if int(t) >= 0)
                s["golden_text"] = gt
                s["text_exact"] = gt == res[0][1]
            print(clip, json.dumps(s, indent=1), flush=True)
            summary[f"{clip}_rep{rep}"] = s
    with open(os.path.join(args.out, f"summary_B{args.B}_{'greedy' if args.greedy else 'sampled'}.json"), "w") as f:
        json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
