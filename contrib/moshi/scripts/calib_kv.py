"""Calibrate per-layer FP8 KV scales: amax of post-RoPE K and of V per Helium layer over the golden clips.

Teacher-forced on the torch LM path (MOSHI_KERNELS=0) on Neuron; writes kv_amax.json {"k": [32], "v": [32]}.
  MOSHI_KERNELS=0 python calib_kv.py --goldens ~/goldens --out kv_amax.json
"""
import argparse, json, os, sys
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths
from moshi_neuron.lm import LMGenHost, MoshiFrame, MoshiLM
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device
import torch.nn as nn


class Calib(nn.Module):
    def __init__(self, lm):
        super().__init__(); self.lm = lm
    def forward(self, tokens, reset, active):
        self.lm._calib = []
        self.lm.temporal(tokens, reset, active, self.lm.C)
        out = torch.stack(self.lm._calib)  # [32,2]
        self.lm._calib = None
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default=os.path.expanduser("~/goldens"))
    ap.add_argument("--clips", default="utt1,utt2,silence12")
    ap.add_argument("--out", default="kv_amax.json")
    a = ap.parse_args()
    neuron_runtime_env(); dev = torch_device()
    lm_path, _, _ = checkpoint_paths()
    lm = MoshiLM(lm_path, 1).to(dev).eval()
    fn = compile_fn()(Calib(lm))
    amax = torch.zeros(32, 2)
    with torch.inference_mode():
        for clip in a.clips.split(","):
            g = torch.load(os.path.join(a.goldens, f"duplex_{clip}_float32_greedy.pt"))
            host = LMGenHost(1)
            ft, fa = g["text_logits"].argmax(-1), g["dep_logits"].argmax(-1)
            for t in range(g["user_codes"].shape[0]):
                inp = host.inputs(g["user_codes"][t].view(1, -1))
                m = fn(inp.to(dev), torch.tensor([t == 0], device=dev), torch.ones(1, dtype=torch.bool, device=dev)).cpu()
                amax = torch.maximum(amax, m)
                host.commit(torch.cat([ft[t].view(1, 1), fa[t].view(1, 8)], 1))
            print(clip, "k amax max", float(amax[:, 0].max()), "v amax max", float(amax[:, 1].max()), flush=True)
    json.dump({"k": amax[:, 0].tolist(), "v": amax[:, 1].tolist()}, open(a.out, "w"))
    print(json.dumps({"k": [round(x, 2) for x in amax[:, 0].tolist()], "v": [round(x, 2) for x in amax[:, 1].tolist()]}))


if __name__ == "__main__":
    main()
