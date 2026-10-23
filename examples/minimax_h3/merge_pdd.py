"""Merge the PDD 8-step LoRA into MiniMax-H3's `transformer/` and extract its per-interval heads.

Usage: python examples/minimax_h3/merge_pdd.py <MiniMax-H3 dir> <MiniMax-H3-FL2VA-Acc-8Step.safetensors> <out_dir>
(the LoRA is `alibaba-pai/MiniMax-H3-Acc-LoRAs`). Then run with --model-path <out_dir>, --steps 9 and
MINIMAX_H3_PDD_HEADS=<out_dir>/pdd_heads.safetensors.
Writes <out_dir>/transformer/ (same shard layout, LoRA merged in, W += (alpha/rank) * up @ down,
computed in fp32 and cast back to each tensor's dtype) and <out_dir>/pdd_heads.safetensors
(proj_out / audio_proj_out per-interval heads + the PDD config as metadata). Symlinks every
other component of the base model so <out_dir> is a drop-in model path.
"""
import json, os, sys, shutil, time
import torch
from safetensors import safe_open
from safetensors.torch import save_file

base, lora_path, out = sys.argv[1:4]
torch.set_num_threads(os.cpu_count())
os.makedirs(os.path.join(out, "transformer"), exist_ok=True)
for name in os.listdir(base):
    if name != "transformer" and not os.path.exists(os.path.join(out, name)):
        os.symlink(os.path.abspath(os.path.join(base, name)), os.path.join(out, name))

lora = safe_open(lora_path, "pt")
meta = lora.metadata()
scale = float(meta["lora_alpha"]) / int(meta["lora_rank"])
lora_keys = set(lora.keys())
heads = {k: lora.get_tensor(k) for k in ("proj_out.weight", "proj_out.bias", "audio_proj_out.weight", "audio_proj_out.bias")}
save_file(heads, os.path.join(out, "pdd_heads.safetensors"), metadata=meta)

src = os.path.join(base, "transformer")
index = json.load(open(os.path.join(src, "diffusion_pytorch_model.safetensors.index.json")))
for f in os.listdir(src):
    if not f.endswith(".safetensors"):
        shutil.copy(os.path.join(src, f), os.path.join(out, "transformer", f))
merged = 0
for shard in sorted(set(index["weight_map"].values())):
    t0 = time.time()
    tensors = {}
    with safe_open(os.path.join(src, shard), "pt") as fh:
        shard_meta = fh.metadata()
        for key in fh.keys():
            w = fh.get_tensor(key)
            prefix = key[: -len(".weight")] if key.endswith(".weight") else None
            if prefix and f"{prefix}.lora_down" in lora_keys:
                down = lora.get_tensor(f"{prefix}.lora_down").float()
                up = lora.get_tensor(f"{prefix}.lora_up").float()
                w = (w.float() + scale * (up @ down)).to(w.dtype)
                merged += 1
            tensors[key] = w.contiguous()
    save_file(tensors, os.path.join(out, "transformer", shard), metadata=shard_meta)
    print(f"{shard}: {len(tensors)} tensors, {time.time() - t0:.0f}s, merged so far {merged}", flush=True)
expected = sum(1 for k in lora_keys if k.endswith(".lora_down"))
print(f"merged {merged}/{expected} LoRA pairs, scale {scale}")
assert merged == expected, "some LoRA targets were not found in the base checkpoint"
