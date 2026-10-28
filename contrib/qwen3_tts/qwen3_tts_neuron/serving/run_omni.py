#!/usr/bin/env python3
"""Drive Qwen3-TTS through vLLM-Omni (Omni offline API) with the Neuron workers.

usage: run_omni.py <deploy_yaml> <out_dir> [n_repeat]
Cases follow the checkpoint type: VoiceDesign -> en_calm/en_excited/zh_soft; CustomVoice -> preset speakers;
Base -> voice clone of QTTS_REF_AUDIO (+ QTTS_REF_TEXT for ICL) in x-vector and ICL modes. Placeholder prompt length is computed with the same
builder the shipped example uses; our runner ignores the placeholder ids and builds embeds itself.
"""
import os, sys, time, json
import torch
import soundfile as sf

from qwen3_tts_neuron.utils import model_path
MODEL = model_path(os.environ.get("QTTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"))
CASES = [
    dict(tag="en_calm", text="She said she would be here by noon, and she always keeps her word.",
         language="English", instruct="Calm, warm adult female voice, steady pace, gentle and reassuring."),
    dict(tag="en_excited", text="No way! Did you really win the whole tournament on your first try?",
         language="English", instruct="Energetic young male voice, bright timbre, fast and excited."),
    dict(tag="zh_soft", text="哥哥，你回来啦，人家等了你好久好久了。",
         language="Chinese", instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。"),
]


REF_TEXT = os.environ.get("QTTS_REF_TEXT", "")
TEXT2 = "The library closes at nine, so please return your books before then."
MODEL_TYPE = json.load(open(f"{MODEL}/config.json")).get("tts_model_type", "voice_design")
if MODEL_TYPE == "custom_voice":
    CASES = [dict(tag="cv_ryan", text=TEXT2, language="English", speaker="ryan"),
             dict(tag="cv_vivian", text=TEXT2, language="English", speaker="vivian", instruct="Speak slowly and softly."),
             dict(tag="cv_ryan_stream", text=TEXT2, language="English", speaker="ryan", non_streaming_mode=False),
             dict(tag="cv_dylan_zh", text="哥哥，你回来啦，人家等了你好久好久了。", language="Chinese", speaker="dylan")]
elif MODEL_TYPE == "base":
    CASES = [dict(tag="base_xvec", text=TEXT2, language="English", x_vector_only_mode=True),
             dict(tag="base_icl", text=TEXT2, language="English", ref_text=REF_TEXT),
             dict(tag="base_icl_ns", text=TEXT2, language="English", ref_text=REF_TEXT, non_streaming_mode=True)]


if os.environ.get("QTTS_CASES"):  # jsonl of {tag, text, language, instruct, ...}
    CASES = [json.loads(l) for l in open(os.environ["QTTS_CASES"]) if l.strip()]


def to_info(c):
    task = {"custom_voice": "CustomVoice", "base": "Base"}.get(MODEL_TYPE, "VoiceDesign")
    info = {"task_type": [task], "text": [c["text"]], "language": [c["language"]], "max_new_tokens": [int(os.environ.get("QTTS_EVAL_MAX_NEW", "2048"))],
            "non_streaming_mode": [c.get("non_streaming_mode", task != "Base")]}
    for k in ("instruct", "speaker", "ref_text", "x_vector_only_mode"):
        if c.get(k) is not None:
            info[k] = [c[k]]
    if task == "Base":
        info["ref_audio"] = [os.environ["QTTS_REF_AUDIO"]]
    return info, task


def placeholder_len(info, task="VoiceDesign"):
    from transformers import AutoTokenizer
    from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
    cfg = json.load(open(f"{MODEL}/config.json"))["talker_config"]
    tok = AutoTokenizer.from_pretrained(MODEL)
    return Qwen3TTSPromptEmbedsBuilder.estimate_prompt_len_from_additional_information(
        info, task_type=task, tokenize_prompt=lambda s: tok(s, padding=False)["input_ids"],
        codec_language_id=cfg.get("codec_language_id"), spk_is_dialect=cfg.get("spk_is_dialect"),
        estimate_ref_code_len=_ref_code_len)


def _ref_code_len(ref_audio):
    """Codec frames (12.5 Hz) of a reference clip path, for the placeholder length only."""
    import math
    ra = ref_audio[0] if isinstance(ref_audio, list) else ref_audio
    info = sf.info(ra)
    return math.ceil(info.frames / info.samplerate * 12.5)


def main():
    from vllm_omni import Omni
    deploy, out = sys.argv[1], sys.argv[2]
    rep = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    os.makedirs(out, exist_ok=True)
    inputs, tags = [], []
    for r in range(rep):
        for c in CASES:
            info, task = to_info(c)
            inputs.append({"prompt_token_ids": [0] * placeholder_len(info, task), "additional_information": info})
            tags.append(f"{c['tag']}_{r}")
    t0 = time.time()
    omni = Omni(model=MODEL, deploy_config=deploy, init_timeout=int(os.environ.get("QTTS_INIT_TIMEOUT", "2400")), stage_init_timeout=int(os.environ.get("QTTS_INIT_TIMEOUT", "2400")))
    print(f"[omni] engine up in {time.time()-t0:.1f}s", flush=True)
    # warm pass
    for _ in omni.generate(inputs[:1]):
        pass
    t1 = time.time()
    audio_s = 0.0
    n = 0
    for so in omni.generate(inputs):
        if not getattr(so, "final_output", True) or not hasattr(so.outputs[0], "multimodal_output"):
            continue
        mm = so.outputs[0].multimodal_output
        a = mm["audio"]
        a = torch.cat(a, dim=-1) if isinstance(a, list) else a
        a = a.float().cpu().numpy().flatten()
        idx = int(str(so.request_id).split("_")[0]) if str(so.request_id).split("_")[0].isdigit() else n
        tag = tags[idx] if idx < len(tags) else f"req{n}"
        sf.write(f"{out}/{tag}.wav", a, 24000)
        audio_s += len(a) / 24000
        n += 1
        print(f"[omni] {so.request_id} -> {tag} {len(a)/24000:.2f}s", flush=True)
    wall = time.time() - t1
    print("[omni-result]", json.dumps(dict(reqs=n, audio_s=round(audio_s, 2), wall_s=round(wall, 3),
                                          rtfx=round(audio_s / wall, 2))), flush=True)


if __name__ == "__main__":
    main()
