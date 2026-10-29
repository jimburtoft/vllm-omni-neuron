#!/usr/bin/env python3
"""A/B of the full Code2Wav crop window (stock graph vs NKI residual units) on device, plus accuracy vs CPU fp32.
usage: QTTS_C2W_NKI_BLOCKS=2,3 c2w_nki_ab.py <model_dir> [B]   (run once with and once without the env var)"""
import os, sys, time
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
import qwen3_tts_neuron.code2wav as cw
B = int(sys.argv[2]) if len(sys.argv) > 2 else 1
dev = torch.device("neuron:0")
m = cw.Code2Wav(os.path.join(sys.argv[1], "speech_tokenizer"), dtype=torch.bfloat16).to(dev)
if cw.NKI_BLOCKS:
    m.prepare_nki()
fn = compile_fn()(cw._Crop(m, 25))
torch.manual_seed(0)
codes = torch.randint(0, 2048, (B, 16, 50))
with torch.inference_mode():
    for _ in range(3): y = fn(codes.to(dev)).cpu()
    ts = []
    for _ in range(30):
        t0 = time.perf_counter(); y = fn(codes.to(dev)).cpu(); ts.append(time.perf_counter() - t0)
tag = "nki" + "".join(map(str, sorted(m.nki_blocks))) if m.nki_blocks else "stock"
torch.save(y.float(), f"/home/ubuntu/c2w_ab_{tag}_B{B}.pt")
print(f"[ab] {tag} B={B}: {1000 * sorted(ts)[15]:.2f} ms (p50 of 30)", flush=True)
