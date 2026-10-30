"""Whisper ASR of Moshi output speech vs its inner-monologue text (quality check). CPU.

  python asr_check.py out/*.wav     (expects a sibling .txt with the inner-monologue text)
"""
import re
import sys

import whisper


def norm(s):
    return re.sub(r"[^a-z' ]", " ", s.lower()).split()


def wer(ref, hyp):
    r, h = norm(ref), norm(hyp)
    d = list(range(len(h) + 1))
    for i in range(1, len(r) + 1):
        prev, d[0] = d[0], i
        for j in range(1, len(h) + 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (r[i - 1] != h[j - 1]))
            prev, d[j] = d[j], cur
    return d[len(h)] / max(1, len(r))


m = whisper.load_model("small.en")
for p in sys.argv[1:]:
    txt = open(p[:-4] + ".txt").read().strip()
    hyp = m.transcribe(p, language="en")["text"].strip()
    print(f"{p}\n  monologue: {txt}\n  asr:       {hyp}\n  wer(monologue->asr): {wer(txt, hyp):.2f}")
