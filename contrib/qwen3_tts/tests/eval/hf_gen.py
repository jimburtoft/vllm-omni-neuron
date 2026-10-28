#!/usr/bin/env python3
"""Reference generations with the official qwen_tts package (CPU, fp32).

usage: hf_gen.py <model_dir> <cases.jsonl> <out_dir> <greedy|sampled> [seed] [batch]
Writes <out_dir>/<tag>.wav and <out_dir>/gen.jsonl (tag, frames, sec, wall).
"""
import json, os, sys, time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

model_dir, cases_f, out, mode = sys.argv[1:5]
seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
bs = int(sys.argv[6]) if len(sys.argv) > 6 else 4
os.makedirs(out, exist_ok=True)
cases = [json.loads(l) for l in open(cases_f) if l.strip()]
done = {json.loads(l)["tag"] for l in open(f"{out}/gen.jsonl")} if os.path.exists(f"{out}/gen.jsonl") else set()
cases = [c for c in cases if c["tag"] not in done]
m = Qwen3TTSModel.from_pretrained(model_dir, device_map="cpu", dtype=torch.float32, attn_implementation="eager")
kw = dict(do_sample=False, subtalker_dosample=False) if mode == "greedy" else {}
MAX_NEW = int(os.environ.get("QTTS_EVAL_MAX_NEW", "600"))
for i in range(0, len(cases), bs):
    b = cases[i:i + bs]
    torch.manual_seed(seed + i)
    t0 = time.time()
    wavs, sr = m.generate_voice_design(text=[c["text"] for c in b], language=[c["language"] for c in b],
                                       instruct=[c["instruct"] for c in b], max_new_tokens=MAX_NEW, **kw)
    wall = time.time() - t0
    with open(f"{out}/gen.jsonl", "a") as f:
        for c, w in zip(b, wavs):
            sf.write(f"{out}/{c['tag']}.wav", w, sr)
            f.write(json.dumps(dict(tag=c["tag"], sec=round(len(w) / sr, 3), frames=round(len(w) / sr * 12.5),
                                    wall=round(wall / len(b), 2))) + "\n")
    print(f"{i + len(b)}/{len(cases)} {wall:.1f}s", flush=True)
