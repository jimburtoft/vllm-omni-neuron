#!/usr/bin/env python3
"""Online client: POST /v1/audio/speech with concurrency; saves wavs, reports latency + RTFx.
usage: speech_client.py <api_base> <out_dir> <concurrency> <n_requests> [model]
QTTS_TASK=VoiceDesign (default) | CustomVoice (QTTS_VOICE, default ryan) | Base (QTTS_REF_AUDIO wav path,
QTTS_REF_TEXT for ICL, else x-vector only). QTTS_STREAM=1 streams PCM and reports time to first audio."""
import io, sys, os, time, json, asyncio
import httpx
import soundfile as sf

CASES = [
    dict(tag="en_calm", input="She said she would be here by noon, and she always keeps her word.",
         language="English", instructions="Calm, warm adult female voice, steady pace, gentle and reassuring."),
    dict(tag="en_excited", input="No way! Did you really win the whole tournament on your first try?",
         language="English", instructions="Energetic young male voice, bright timbre, fast and excited."),
    dict(tag="zh_soft", input="哥哥，你回来啦，人家等了你好久好久了。",
         language="Chinese", instructions="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。"),
]


STREAM = os.environ.get("QTTS_STREAM", "0") == "1"
TASK = os.environ.get("QTTS_TASK", "VoiceDesign")
_REF = None


def payload_for(model, c):
    global _REF
    p = dict(model=model, input=c["input"], language=c["language"], task_type=TASK)
    if TASK == "VoiceDesign":
        p["instructions"] = c["instructions"]
    elif TASK == "CustomVoice":
        p["voice"] = os.environ.get("QTTS_VOICE", "ryan")
    elif TASK == "Base":
        if _REF is None:
            import base64
            _REF = "data:audio/wav;base64," + base64.b64encode(open(os.environ["QTTS_REF_AUDIO"], "rb").read()).decode()
        p["ref_audio"] = _REF
        if os.environ.get("QTTS_REF_TEXT"):
            p["ref_text"] = os.environ["QTTS_REF_TEXT"]
        else:
            p["x_vector_only_mode"] = True
    return p


async def one_stream(client, url, model, i, out):
    import numpy as np
    c = CASES[i % 3]
    payload = dict(payload_for(model, c), response_format="pcm", stream=True, stream_format="audio")
    t0 = time.perf_counter()
    ttfa, buf = None, bytearray()
    async with client.stream("POST", url, json=payload) as r:
        if r.status_code != 200:
            body = await r.aread()
            return dict(i=i, ok=False, status=r.status_code, err=body[:300].decode(errors="ignore"), lat=time.perf_counter() - t0)
        async for chunk in r.aiter_bytes():
            if chunk and ttfa is None:
                ttfa = time.perf_counter() - t0
            buf.extend(chunk)
    dt = time.perf_counter() - t0
    w = np.frombuffer(bytes(buf[: len(buf) // 2 * 2]), dtype=np.int16).astype("float32") / 32768.0
    sf.write(f"{out}/{c['tag']}_{i}.wav", w, 24000)
    return dict(i=i, ok=True, lat=dt, ttfa=ttfa, audio_s=len(w) / 24000)


async def one(client, url, model, i, out):
    if STREAM:
        return await one_stream(client, url, model, i, out)
    c = CASES[i % 3]
    payload = dict(payload_for(model, c), response_format="wav")
    t0 = time.perf_counter()
    r = await client.post(url, json=payload)
    dt = time.perf_counter() - t0
    if r.status_code != 200:
        return dict(i=i, ok=False, status=r.status_code, err=r.text[:300], lat=dt)
    w, sr = sf.read(io.BytesIO(r.content))
    sf.write(f"{out}/{c['tag']}_{i}.wav", w, sr)
    return dict(i=i, ok=True, lat=dt, audio_s=len(w) / sr)


async def main():
    base, out, conc, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    model = sys.argv[5] if len(sys.argv) > 5 else "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    os.makedirs(out, exist_ok=True)
    url = f"{base}/v1/audio/speech"
    sem = asyncio.Semaphore(conc)
    # httpx defaults to max_connections=100, which silently caps concurrency above 100
    limits = httpx.Limits(max_connections=conc + 8, max_keepalive_connections=conc + 8)
    async with httpx.AsyncClient(timeout=600.0, limits=limits) as client:
        async def run(i):
            async with sem:
                return await one(client, url, model, i, out)
        r0 = await one(client, url, model, 0, out)  # warm
        print("[client] warm:", json.dumps(r0)[:300], flush=True)
        t0 = time.perf_counter()
        rs = await asyncio.gather(*[run(i) for i in range(n)])
        wall = time.perf_counter() - t0
    ok = [r for r in rs if r["ok"]]
    lat = sorted(r["lat"] for r in ok)
    audio = sum(r["audio_s"] for r in ok)
    print("[client-result]", json.dumps(dict(conc=conc, reqs=n, ok=len(ok), audio_s=round(audio, 1), wall_s=round(wall, 2),
          rtfx=round(audio / wall, 2), lat_p50=round(lat[len(lat) // 2], 3) if lat else None,
          lat_p90=round(lat[int(0.9 * len(lat))], 3) if lat else None,
          ttfa_p50=round(sorted(r["ttfa"] for r in ok if r.get("ttfa"))[len(ok) // 2], 3) if ok and ok[0].get("ttfa") else None,
          errors=[r for r in rs if not r["ok"]][:2])), flush=True)


asyncio.run(main())
