"""Duplex client for Moshi on vLLM-Omni: streams wav files to `/v1/realtime?duplex=1` in real time (80 ms chunks)
over N concurrent sessions, records the returned audio + text, and reports latency / real-time statistics.

  python -m moshi_neuron.serving.duplex_client --url ws://127.0.0.1:8000/v1/realtime?duplex=1 \
      --wav ~/goldens/clip_utt1.npy --sessions 1 --out out_duplex
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time

import numpy as np


def load_wav(p):
    if p.endswith(".npy"):
        return np.load(p).astype(np.float32)
    import soundfile as sf
    x, sr = sf.read(p, dtype="float32")
    if x.ndim > 1:
        x = x.mean(1)
    if sr != 24000:
        import librosa
        x = librosa.resample(x, orig_sr=sr, target_sr=24000)
    return x


async def one_session(i, args, pcm, model):
    from vllm_omni.clients.duplex import AudioFormat, DuplexClient, SessionConfig
    cfg = SessionConfig(input_audio=AudioFormat("pcm_f32le", 24000), output_audio=AudioFormat("pcm16", 24000))
    audio, text, arrivals = [], [], []
    t_start = None
    async with DuplexClient(args.url, model=model, config=cfg, handshake_timeout_s=600) as cl:
        async def reader():
            async for ev in cl.events():
                if ev.type in ("response.output_audio.delta", "response.audio.delta"):
                    a = ev.audio
                    if a:
                        audio.append(np.frombuffer(a, dtype="<i2").astype(np.float32) / 32768.0)
                        arrivals.append((time.perf_counter() - t_start, len(a) // 2))
                    t = ev.raw.get("text")
                    if isinstance(t, str) and t:
                        text.append(t)
                elif ev.type in ("response.output_text.delta", "response.text.delta",
                                 "response.output_audio_transcript.delta", "response.audio_transcript.delta"):
                    t = ev.raw.get("delta") or ev.raw.get("text")
                    if isinstance(t, str) and t:
                        text.append(t)
        rt = asyncio.create_task(reader())
        t_start = time.perf_counter()
        await cl.stream_pcm(pcm.astype("<f4").tobytes(), chunk_ms=80, realtime=True)
        await asyncio.sleep(args.tail_s)
        rt.cancel()
    wav = np.concatenate(audio) if audio else np.zeros(0, np.float32)
    dur_in = len(pcm) / 24000
    # lag of each audio chunk vs the moment the server could have produced it at real-time pace
    got = np.cumsum([n for _, n in arrivals]) / 24000 if arrivals else np.zeros(0)
    lag = [t - g for (t, _), g in zip(arrivals, got)]
    return dict(session=i, text="".join(text), out_s=len(wav) / 24000, in_s=dur_in, chunks=len(arrivals),
                first_audio_s=arrivals[0][0] if arrivals else None,
                lag_p50_ms=float(np.median(lag) * 1e3) if lag else None,
                lag_p99_ms=float(np.quantile(lag, 0.99) * 1e3) if lag else None,
                lag_max_ms=float(np.max(lag) * 1e3) if lag else None), wav


async def main_async(args):
    pcm = load_wav(os.path.expanduser(args.wav))
    os.makedirs(args.out, exist_ok=True)
    res = await asyncio.gather(*[one_session(i, args, pcm, args.model) for i in range(args.sessions)])
    import soundfile as sf
    summ = []
    for r, wav in res:
        sf.write(os.path.join(args.out, f"session{r['session']}.wav"), wav, 24000)
        with open(os.path.join(args.out, f"session{r['session']}.txt"), "w") as f:
            f.write(r["text"] + "\n")
        summ.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summ, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8000/v1/realtime?duplex=1")
    ap.add_argument("--model", default="kyutai/moshiko-pytorch-bf16")
    ap.add_argument("--wav", required=True)
    ap.add_argument("--sessions", type=int, default=1)
    ap.add_argument("--tail-s", type=float, default=2.0)
    ap.add_argument("--out", default="out_duplex")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
