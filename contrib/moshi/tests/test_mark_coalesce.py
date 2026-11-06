"""MOSHI_MARK_COALESCE equivalence: _text_chars_for_audio_ms on the stock vs coalesced mark list gives the same chars
at every audio position (1 ms grid) for a synthetic 750-frame response with words every ~4 frames."""
import os
import random
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(4)
    from vllm_omni.entrypoints.duplex import protocol as pr
    from vllm_omni.entrypoints.duplex.realtime_input import RealtimeInputTranslator as RI

    def drive(sess, N, seed=0):
        rnd = random.Random(seed)
        chars = 0
        for f in range(N):
            if rnd.random() < 0.3:
                chars += rnd.randint(1, 6)
            d = 80 * (f + 1)
            sess.mark_audio_sent(d, text_chars=chars, audio_text_marks=[{"text_chars": chars, "audio_end_ms": d}])
        return chars

    def new():
        s = pr.DuplexSession.__new__(pr.DuplexSession)
        pr.DuplexSession.__init__(s, session_id="s") if False else None
        return s
    import inspect
    sig = inspect.signature(pr.DuplexSession.__init__)
    kw = {p: "s" for p in sig.parameters if p not in ("self",) and sig.parameters[p].default is inspect._empty}
    a = pr.DuplexSession(**kw)
    N = 750
    T = drive(a, N)
    ref = list(a._response.assistant_audio_text_marks)
    from moshi_neuron.serving.phases import install_mark_coalesce
    install_mark_coalesce()
    b = pr.DuplexSession(**kw)
    drive(b, N)
    got = list(b._response.assistant_audio_text_marks)
    diff = 0
    for ms in range(0, 80 * N + 200, 7):
        x = a._text_chars_for_audio_ms(ms, T, marks=ref)
        y = b._text_chars_for_audio_ms(ms, T, marks=got)
        x2 = RI._text_chars_for_audio_ms_from_marks(ms, T, [m.__dict__ for m in ref], final_ms=80 * N)
        y2 = RI._text_chars_for_audio_ms_from_marks(ms, T, [m.__dict__ for m in got], final_ms=80 * N)
        diff += (x != y) + (x2 != y2)
    print(f"marks stock {len(ref)} -> coalesced {len(got)}; positions differing: {diff}")
