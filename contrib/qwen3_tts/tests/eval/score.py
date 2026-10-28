#!/usr/bin/env python3
"""ASR (WER/CER) + duration stats for one or more generation dirs against a cases file.

usage: score.py <cases.jsonl> <out.jsonl> <dir> [<dir> ...]
ASR model: $ASR_MODEL (default openai/whisper-large-v3). CER for zh/ja/ko, WER otherwise.
"""
import glob, json, os, re, sys, unicodedata

import librosa
import soundfile as sf
import torch
from transformers import pipeline

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
cases = {c["tag"]: c for c in (json.loads(l) for l in open(sys.argv[1]) if l.strip())}
out = open(sys.argv[2], "a")
asr = pipeline("automatic-speech-recognition", model=os.environ.get("ASR_MODEL", "openai/whisper-large-v3"),
               device="cpu", torch_dtype=torch.float32)
CJK = {"Chinese", "Japanese", "Korean"}
NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven",
       "8": "eight", "9": "nine", "10": "ten", "15": "fifteen", "18": "eighteen"}


def norm(s, lang):
    s = unicodedata.normalize("NFKC", s).lower()
    s = re.sub(r"[^\w\s']", " ", s).replace("'", "")
    if lang == "English":
        s = " ".join(NUM.get(w, w) for w in s.split())
    if lang in CJK:
        return list(s.replace(" ", ""))
    return s.split()


def ed(a, b):
    d = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        p, d[0] = d[0], i
        for j in range(1, len(b) + 1):
            p, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, p + (a[i - 1] != b[j - 1]))
    return d[len(b)]


for d in sys.argv[3:]:
    for f in sorted(glob.glob(f"{d}/*.wav")):
        tag = os.path.basename(f)[:-4]
        tag = re.sub(r"_\d+$", "", tag) if tag not in cases else tag
        c = cases.get(tag)
        if c is None:
            continue
        w, sr = sf.read(f, dtype="float32")
        if w.ndim > 1:
            w = w.mean(-1)
        hyp = asr({"raw": librosa.resample(w, orig_sr=sr, target_sr=16000), "sampling_rate": 16000},
                  return_timestamps=True, generate_kwargs={"language": c["language"].lower(), "task": "transcribe"})["text"]
        r, h = norm(c["text"], c["language"]), norm(hyp, c["language"])
        out.write(json.dumps(dict(dir=d, file=os.path.basename(f)[:-4], tag=tag, lang=c["language"], sec=round(len(w) / sr, 3),
                                  chars=len(c["text"]), n=len(r), err=ed(r, h), hyp=hyp[:400]), ensure_ascii=False) + "\n")
        out.flush()
    print("scored", d, flush=True)
