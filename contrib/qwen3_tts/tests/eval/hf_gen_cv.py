#!/usr/bin/env python3
"""Reference CustomVoice generations (qwen_tts, CPU fp32, sampled defaults). usage: hf_gen_cv.py <model> <cases> <out>"""
import json, os, sys
import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel
m = Qwen3TTSModel.from_pretrained(sys.argv[1], device_map="cpu", dtype=torch.float32, attn_implementation="eager")
cases = [json.loads(l) for l in open(sys.argv[2]) if l.strip()]
os.makedirs(sys.argv[3], exist_ok=True)
for i, c in enumerate(cases):
    torch.manual_seed(1000 + i)
    w, sr = m.generate_custom_voice(text=c["text"], speaker=c["speaker"], language=c["language"],
                                    non_streaming_mode=c["non_streaming_mode"], max_new_tokens=600)
    sf.write(f"{sys.argv[3]}/{c['tag']}.wav", w[0], sr)
    print(i, c["tag"], round(len(w[0]) / sr, 2), flush=True)
