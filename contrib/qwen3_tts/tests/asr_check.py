#!/usr/bin/env python3
"""Whisper ASR of wav files vs expected text. usage: asr_check.py <dir> <expected.json {stem: text}> [lang]"""
import glob, json, os, re, sys

import librosa
import soundfile as sf
import torch
from transformers import pipeline

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
asr = pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", "openai/whisper-small"), device="cpu")


def norm(s, zh):
    s = re.sub(r"[^\w\u4e00-\u9fff ]+", "", s.lower())
    return list(s.replace(" ", "")) if zh else s.split()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


exp = json.load(open(sys.argv[2]))
for f in sorted(glob.glob(sys.argv[1] + "/*.wav")):
    stem = os.path.basename(f)[:-4]
    if stem not in exp:
        continue
    text, lang = exp[stem] if isinstance(exp[stem], list) else (exp[stem], "english")
    zh = lang == "chinese"
    w, sr = sf.read(f)
    hyp = asr({"raw": librosa.resample(w.astype("float32"), orig_sr=sr, target_sr=16000), "sampling_rate": 16000},
              return_timestamps=True, generate_kwargs={"language": lang, "task": "transcribe"})["text"]
    r, h = norm(text, zh), norm(hyp, zh)
    print(json.dumps(dict(file=stem, sec=round(len(w) / sr, 2), ref_words=len(r), hyp_words=len(h),
                          err=round(ed(r, h) / max(1, len(r)), 4), hyp=hyp[:300]), ensure_ascii=False), flush=True)
