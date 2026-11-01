"""Offline: how much does FP8 weight quantization of Helium MLPs perturb a layer's output? (CPU, a few layers.)

Feeds real hidden states (random-normal scaled like activations is not good enough -> we use the embedding sum of real
golden tokens passed through layer 0..L-1 in bf16 on CPU) and compares MLP outputs with fp8 per-row / per-group /
gate-up-only against the bf16 weights. Reports relative error per layer.
  python quant_err.py --layers 0,8,16,31
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import checkpoint_paths  # noqa: E402
from moshi_neuron.lm import _fp8_group, _fp8_rowwise  # noqa: E402


def mlp(h, wg, wu, wd):
    return F.linear(F.silu(F.linear(h, wg)) * F.linear(h, wu), wd)


def deq_row(w):
    q, s = _fp8_rowwise(w, 240.0)
    return (q.float() * s[:, None]).to(w.dtype)


def deq_grp(w, g):
    q, s = _fp8_group(w, 240.0, g)
    o, i = w.shape
    return (q.float().view(o, i // g, g) * s[..., None]).view(o, i).to(w.dtype)


def deq_i8(w, g=0):
    o, i = w.shape
    wg = w.view(o, i // g, g) if g else w.view(o, 1, i)
    sc = wg.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (torch.round(wg / sc).clamp(-127, 127) * sc).view(o, i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,8,16,31")
    ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    torch.manual_seed(0)
    lm_path, _, _ = checkpoint_paths()
    f = safe_open(lm_path, framework="pt")
    g = torch.load(os.path.expanduser("~/goldens/duplex_utt2_float32_greedy.pt"))
    # proxy hidden state: the golden transformer output (post out_norm) has realistic scale/direction
    h0 = g["transformer_out"][5:5 + a.n].float()
    h0 = h0 / h0.pow(2).mean(-1, keepdim=True).sqrt()  # rms-normalized, like the MLP input after norm2 (alpha ~1)
    for li in [int(x) for x in a.layers.split(",")]:
        p = f"transformer.layers.{li}."
        win = f.get_tensor(p + "gating.linear_in.weight").float()
        wd = f.get_tensor(p + "gating.linear_out.weight").float()
        alpha = f.get_tensor(p + "norm2.alpha").float().view(-1)
        I = win.shape[0] // 2
        wg, wu = win[:I], win[I:]
        h = (h0 * alpha).to(torch.bfloat16).float()
        ref = mlp(h, wg, wu, wd)
        rel = lambda y: float((y - ref).norm() / ref.norm())
        res = {"row(all)": rel(mlp(h, deq_row(wg), deq_row(wu), deq_row(wd))),
               "row(gate/up)": rel(mlp(h, deq_row(wg), deq_row(wu), wd)),
               "g128(all)": rel(mlp(h, deq_grp(wg, 128), deq_grp(wu, 128), deq_grp(wd, 128))),
               "g128(gate/up)": rel(mlp(h, deq_grp(wg, 128), deq_grp(wu, 128), wd)),
               "g32(all)": rel(mlp(h, deq_grp(wg, 32), deq_grp(wu, 32), deq_grp(wd, 32))),
               "int8 row(all)": rel(mlp(h, deq_i8(wg), deq_i8(wu), deq_i8(wd))),
               "int8 g128(all)": rel(mlp(h, deq_i8(wg, 128), deq_i8(wu, 128), deq_i8(wd, 128))),
               "bf16-round(all)": rel(mlp(h, wg.bfloat16().float(), wu.bfloat16().float(), wd.bfloat16().float()))}
        print(f"layer {li}: " + "  ".join(f"{k} {v:.4f}" for k, v in res.items()), flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main()
