#!/usr/bin/env python3
"""Edge cases through the OpenAI server: long text, very long prompt, empty text, bad language, odd characters.
Records status code, duration, and whether the server survives. usage: server_limits_check.py <base> [model]"""
import asyncio, io, json, sys

import httpx
import soundfile as sf

LONG = ("The history of the lighthouse goes back thousands of years. Early beacons were simple fires on hilltops, "
        "later replaced by towers with oil lamps and mirrors. In the nineteenth century, the Fresnel lens transformed "
        "how far a light could be seen, and keepers lived lonely lives tending the flame through storms. ")
CASES = {
    "medium_3_sentences": LONG,
    "long_x3": LONG * 3,
    "very_long_prompt_x8": LONG * 8,
    "too_long_x30": LONG * 30,
    "empty": "",
    "punct_only": "...!?",
    "emoji_mixed": "Hello 👋 world — ça va? 你好！",
    "numbers": "Call 555-0123 at 3:45 PM on 12/25/2026, it costs $1,299.99.",
}


async def one(cl, base, model, name, text, lang="English"):
    p = dict(model=model, task_type="VoiceDesign", response_format="wav", input=text, language=lang,
             instructions="Neutral adult narrator voice, clear and steady.", extra_params=dict(temperature=0.0, top_k=-1,
                                                                                               subtalker_dosample=False))
    try:
        r = await cl.post(f"{base}/v1/audio/speech", json=p)
    except Exception as e:
        return dict(case=name, error=type(e).__name__)
    if r.status_code != 200:
        return dict(case=name, status=r.status_code, body=r.text[:200])
    w, sr = sf.read(io.BytesIO(r.content))
    import os
    os.makedirs("limits_out", exist_ok=True)
    sf.write(f"limits_out/{name}.wav", w, sr)
    return dict(case=name, status=200, sec=round(len(w) / sr, 2), chars=len(text))


async def main():
    base = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    async with httpx.AsyncClient(timeout=1800) as cl:
        for name, text in CASES.items():
            print("[limits]", json.dumps(await one(cl, base, model, name, text), ensure_ascii=False), flush=True)
        print("[limits]", json.dumps(await one(cl, base, model, "bad_language", "Hello there.", lang="Klingon")), flush=True)
        print("[limits]", json.dumps(await one(cl, base, model, "after_all_ok", "Still alive?")), flush=True)


asyncio.run(main())
