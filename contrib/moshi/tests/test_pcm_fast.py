"""MOSHI_PCM_FAST equivalence: numpy PCM16 output == soundfile output (base64) for real-looking Mimi frames."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

if __name__ == "__main__":
    import numpy as np
    import time
    import io
    import base64
    import soundfile as sf
    from moshi_neuron.serving import audio_encoding_fast as af

    def ref(a, sr, fmt, sp):
        b = io.BytesIO()
        sf.write(b, a, sr, format={"pcm": "RAW", "wav": "WAV"}[fmt], **({"subtype": "PCM_16"} if fmt == "pcm" else {}))
        return base64.b64encode(b.getvalue()).decode()
    af._FALLBACK["fn"] = ref
    for fmt in ("pcm", "wav"):
        mode = af._pcm16_mode(fmt, 24000)
        x = np.load(os.path.expanduser("~/goldens/clip_utt2.npy")).astype(np.float32)
        diff, n = 0, 0
        t_ref = t_new = 0.0
        for k in range(0, len(x) - 1920, 1920):
            fr = x[k:k + 1920] * 1.3
            t0 = time.perf_counter(); r = ref(fr, 24000, fmt, 1.0); t1 = time.perf_counter()
            o = af._encode_pcm16(fr, 24000, fmt, mode) if mode else r; t2 = time.perf_counter()
            diff += (r != o); n += 1; t_ref += t1 - t0; t_new += t2 - t1
        print(f"{fmt}: mode={mode} frames {n} differing {diff}; soundfile {t_ref / n * 1e6:.0f} us, numpy {t_new / n * 1e6:.0f} us")
