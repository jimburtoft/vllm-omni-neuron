#!/usr/bin/env python3
"""Per-request sampling through the OpenAI server, with concurrent (batched) requests.
usage: server_sampling_check.py <api_base> [model]"""
import asyncio, hashlib, io, json, sys

import httpx
import soundfile as sf

C0 = dict(input="She said she would be here by noon, and she always keeps her word.", language="English",
          instructions="Calm, warm adult female voice, steady pace, gentle and reassuring.")
C1 = dict(input="No way! Did you really win the whole tournament on your first try?", language="English",
          instructions="Energetic young male voice, bright timbre, fast and excited.")


async def call(cl, base, model, case, **kw):
    p = dict(model=model, task_type="VoiceDesign", response_format="wav", **case, **kw)
    r = await cl.post(f"{base}/v1/audio/speech", json=p)
    assert r.status_code == 200, r.text[:300]
    w, sr = sf.read(io.BytesIO(r.content), dtype="int16")
    return hashlib.md5(w.tobytes()).hexdigest()[:10], round(len(w) / sr, 2)


async def main():
    base = sys.argv[1]; model = sys.argv[2] if len(sys.argv) > 2 else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    greedy = dict(extra_params=dict(temperature=0.0, top_k=-1, subtalker_dosample=False))
    samp = lambda seed: dict(seed=seed, extra_params=dict(temperature=0.9, top_k=50, subtalker_dosample=True,
                                                          subtalker_temperature=0.9, subtalker_top_k=50))
    async with httpx.AsyncClient(timeout=900) as cl:
        g_alone = await call(cl, base, model, C0, **greedy)
        s7_alone = await call(cl, base, model, C1, **samp(7))
        batch = await asyncio.gather(
            call(cl, base, model, C0, **greedy), call(cl, base, model, C1, **samp(7)), call(cl, base, model, C1, **samp(7)),
            call(cl, base, model, C1, **samp(8)), call(cl, base, model, C0, max_new_tokens=20, **greedy),
            call(cl, base, model, C1, **samp(9)), call(cl, base, model, C0, **greedy))
    r = dict(greedy_alone_eq_in_batch=g_alone[0] == batch[0][0] == batch[6][0],
             seed7_alone_eq_in_batch=s7_alone[0] == batch[1][0] == batch[2][0],
             seed8_differs=batch[3][0] != batch[1][0], sampled_differs_from_greedy=s7_alone[0] != g_alone[0],
             max_new_tokens_20_s=batch[4][1], durations=[b[1] for b in batch])
    print("[server-sampling]", json.dumps(r), flush=True)


asyncio.run(main())
