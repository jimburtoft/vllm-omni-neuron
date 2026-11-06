"""MOSHI_MARKS_FIX equivalence: stock vs patched realtime projection over a synthetic N-frame response.
Same event stream into a fresh NativeRealtimeSessionProtocol each time; compares every projected payload list and
the final conversation item (after response.done); times the audio-delta projection."""
import copy
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def events(N):
    ev = [{"type": "session.created", "session": {"id": "s"}},
          {"type": "response.created", "response_id": "r1", "session_id": "s", "epoch": 0}]
    chars = 0
    for f in range(N):
        txt = "ab " if f % 3 == 0 else ""
        chars += len(txt)
        if f % 50 == 25:
            ev.append({"type": "playback.acknowledged", "session_id": "s", "played_ms": 80 * f})
        ev.append({"type": "response.output_audio.delta", "response_id": "r1", "session_id": "s", "epoch": 0,
                   "text": txt, "audio": "AAAA", "format": "pcm_s16le", "sample_rate_hz": 24000,
                   "audio_duration_ms": 80 * (f + 1), "end_of_turn": False, "model_speak": True,
                   "audio_text_marks": [{"text_chars": chars, "audio_end_ms": 80 * (f + 1)}],
                   "playback": {"sent_ms": 80 * (f + 1)}})
    ev.append({"type": "response.done", "response_id": "r1", "session_id": "s", "epoch": 0, "status": "completed"})
    return ev


def run(N, fixed):
    from vllm_omni.entrypoints.duplex.realtime_session import NativeRealtimeSessionProtocol as Proto
    p = Proto({})
    outs, t_delta = [], 0.0
    for e in events(N):
        t0 = time.perf_counter()
        o = p._from_duplex_event(copy.deepcopy(e))
        dt = time.perf_counter() - t0
        if e["type"] == "response.output_audio.delta":
            t_delta += dt
        outs.append(json.dumps(o, sort_keys=True, default=str))
    flush = getattr(p, "_moshi_flush", None)
    if flush:
        flush()
    items = json.dumps(p._conversation_items, sort_keys=True, default=str)
    return outs, items, t_delta / N * 1e3


if __name__ == "__main__":
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(4)
    N = int(os.environ.get("N", "750"))
    a_out, a_items, a_t = run(N, False)
    from moshi_neuron.serving.phases import install_realtime_marks_fix
    install_realtime_marks_fix()
    b_out, b_items, b_t = run(N, True)
    diff = sum(x != y for x, y in zip(a_out, b_out))
    print(f"N={N}: payload lists differ {diff}/{len(a_out)}; final items equal {a_items == b_items}; "
          f"audio-delta projection stock {a_t:.3f} ms/frame, fixed {b_t:.3f} ms/frame")


def test_client_hook():
    """the client-event flush hook must skip input_audio_buffer.* and run for item events"""
    import asyncio
    from vllm_omni.entrypoints.duplex.realtime_session import NativeRealtimeSessionProtocol as Proto
    p = Proto({})
    calls = []
    p._moshi_flush = lambda: calls.append(1)

    async def go():
        for t in ("input_audio_buffer.append", "conversation.item.retrieve"):
            try:
                await p._to_duplex_event({"type": t, "audio": "", "item_id": "x"})
            except Exception:  # noqa: BLE001
                pass
    asyncio.run(go())
    return calls


if __name__ == "__main__" and os.environ.get("HOOK") == "1":
    print("flush calls for [append, retrieve]:", len(test_client_hook()))
