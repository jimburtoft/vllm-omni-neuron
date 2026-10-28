#!/usr/bin/env python3
"""Dump HF reference prompts + greedy codes for CustomVoice / Base (xvec, ICL) and streaming text (refvenv).

usage: hf_task_dump.py <customvoice_dir> <base_dir> <ref_wav> <ref_text> <out.pt>
Captures talker.generate's inputs (inputs_embeds, trailing_text_hidden, tts_pad_embed) and the greedy codes.
"""
import sys

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

cv_dir, base_dir, ref_wav, ref_text, out = sys.argv[1:6]
torch.manual_seed(0)
G = dict(do_sample=False, subtalker_dosample=False, max_new_tokens=60)
TEXT = "The library closes at nine, so please return your books before then."
res = {}


def run(model, tag, fn):
    cap = {}
    talker = model.model.talker
    orig = talker.generate

    def grab(*a, **kw):
        cap.update(emb=kw["inputs_embeds"].clone(), trailing=kw["trailing_text_hidden"].clone(),
                   pad=kw["tts_pad_embed"].clone())
        return orig(*a, **kw)
    talker.generate = grab
    codes = {}
    m = model.model
    origm = m.speech_tokenizer.decode

    def dec(items):
        codes["c"] = [it["audio_codes"].clone() for it in items]
        return origm(items)
    m.speech_tokenizer.decode = dec
    try:
        fn()
    finally:
        talker.generate = orig
        m.speech_tokenizer.decode = origm
    res[tag] = dict(emb=cap["emb"][0], trailing=cap["trailing"][0], pad=cap["pad"].reshape(-1), codes=codes["c"][0])
    print(tag, tuple(cap["emb"].shape), tuple(cap["trailing"].shape), tuple(codes["c"][0].shape), flush=True)


cv = Qwen3TTSModel.from_pretrained(cv_dir, device_map="cpu", dtype=torch.float32, attn_implementation="eager")
run(cv, "cv_ryan_ns", lambda: cv.generate_custom_voice(text=TEXT, speaker="ryan", language="English", **G))
run(cv, "cv_vivian_instr_ns", lambda: cv.generate_custom_voice(text=TEXT, speaker="vivian", language="English",
                                                             instruct="Speak slowly and softly.", **G))
run(cv, "cv_ryan_stream", lambda: cv.generate_custom_voice(text=TEXT, speaker="ryan", language="English",
                                                         non_streaming_mode=False, **G))
del cv
base = Qwen3TTSModel.from_pretrained(base_dir, device_map="cpu", dtype=torch.float32, attn_implementation="eager")
w, sr = sf.read(ref_wav, dtype="float32")
res["ref_audio"] = (torch.from_numpy(w), sr)
run(base, "base_xvec", lambda: base.generate_voice_clone(text=TEXT, language="English", ref_audio=(w, sr),
                                                        x_vector_only_mode=True, **G))
run(base, "base_icl", lambda: base.generate_voice_clone(text=TEXT, language="English", ref_audio=(w, sr),
                                                       ref_text=ref_text, **G))
run(base, "base_icl_ns", lambda: base.generate_voice_clone(text=TEXT, language="English", ref_audio=(w, sr),
                                                          ref_text=ref_text, non_streaming_mode=True, **G))
res["text"], res["ref_text"] = TEXT, ref_text
torch.save(res, out)
