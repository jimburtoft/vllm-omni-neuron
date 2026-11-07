"""Backport of vllm-omni PR #8405 (open, from #8329) numpy PCM16 / WAV writer for the 0.29 duplex data plane
(MOSHI_PCM_FAST=1). Stock 0.29 encodes every 80 ms audio frame of every session through soundfile (libsndfile into
a BytesIO); upstream measured that this alone took an A800 at 80 PersonaPlex sessions from 0/80 to 80/80 real time.
The numpy writer is used only after a one-time calibration probe proves it reproduces soundfile's exact bytes for
the (format, sample rate); otherwise soundfile stays in use. Code below is upstream's, adapted to 0.29's
`NativeRuntimeBridgeMixin._encode_native_data_plane_audio` and `_chat_service.create_audio`."""
from __future__ import annotations

import base64
import struct
from collections.abc import Callable

import numpy as np

from vllm.logger import init_logger

logger = init_logger(__name__)

_FALLBACK = {"fn": None}


def _encode_with_soundfile(audio_tensor, sample_rate_hz, response_format, speed):
    return _FALLBACK["fn"](audio_tensor, sample_rate_hz, response_format, speed)


_FLOAT32_BELOW_2_31 = float(np.nextafter(np.float32(2.0**31), np.float32(0.0)))


def _pcm16_int32_high(samples: np.ndarray) -> np.ndarray:
    """libsndfile 1.2: round ``x * 2**31`` half to even into an int32, keep its high 16 bits."""
    # Scaling by a power of two is exact, so rint rounds the value libsndfile rounds.
    scaled = samples * np.float32(2.0**31)
    np.clip(scaled, -(2.0**31), _FLOAT32_BELOW_2_31, out=scaled)
    return (np.rint(scaled, out=scaled).astype(np.int32) >> 16).astype("<i2")


def _pcm16_rint(scale: float) -> Callable[[np.ndarray], np.ndarray]:
    """Older libsndfile builds: clip ``x * scale`` to int16 and round it half to even (``lrintf``)."""

    def quantize(samples: np.ndarray) -> np.ndarray:
        scaled = samples * np.float32(scale)
        np.clip(scaled, -32768.0, 32767.0, out=scaled)
        return np.rint(scaled, out=scaled).astype("<i2")

    return quantize


def _pcm16_floor_f64(samples: np.ndarray) -> np.ndarray:
    """``AudioMixin``'s own RAW writer: ``floor(x * 32768)`` in float64, clipped to int16."""
    return np.clip(np.floor(samples.astype(np.float64) * 32768.0), -32768, 32767).astype("<i2")


# Float-to-PCM_16 quantizers that libsndfile builds and AudioMixin's RAW writer
# have used; calibration picks the one that reproduces the fallback's bytes.


_PCM16_QUANTIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "int32_high": _pcm16_int32_high,
    "rint_32768": _pcm16_rint(32768.0),
    "rint_32767": _pcm16_rint(32767.0),
    "floor_f64": _pcm16_floor_f64,
}
# (format, sample rate) -> the calibrated quantizer's name, or None.


_pcm16_mode_by_key: dict[tuple[str, int], str | None] = {}


def _encode_pcm16(audio: np.ndarray, sample_rate_hz: int, fmt: str, mode: str) -> str | None:
    """Base64 of soundfile's PCM_16 bytes for *audio* under quantizer *mode*; ``None`` if a sample is not finite."""
    if not np.isfinite(audio.sum()):
        return None
    data = _PCM16_QUANTIZERS[mode](audio.astype(np.float32, copy=False)).tobytes()
    if fmt == "wav":
        data = _wav_pcm16_mono_header(sample_rate_hz, len(data)) + data
    return base64.b64encode(data).decode("utf-8")


def _wav_pcm16_mono_header(sample_rate_hz: int, data_bytes: int) -> bytes:
    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_bytes,
        b"WAVE",
        b"fmt ",
        16,
        1,  # WAVE_FORMAT_PCM
        1,  # mono
        sample_rate_hz,
        sample_rate_hz * 2,
        2,
        16,
        b"data",
        data_bytes,
    )


def _pcm16_mode(fmt: str, sample_rate_hz: int) -> str | None:
    key = (fmt, sample_rate_hz)
    if key not in _pcm16_mode_by_key:
        _pcm16_mode_by_key[key] = _calibrate_pcm16(fmt, sample_rate_hz)
    return _pcm16_mode_by_key[key]


def _calibration_probe() -> np.ndarray:
    """Samples around every edge of the conversion: clipping, rounding ties, tiny and random values."""
    edges = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 1.5, -1.5, 2.0, -2.0, 1e5, -1e5, 1e-6, -1e-6, 1e-40, -1e-40]
    edges += [np.nextafter(np.float32(1.0), np.float32(2.0)), np.nextafter(np.float32(-1.0), np.float32(-2.0))]
    edges += [np.nextafter(np.float32(1.0), np.float32(0.0)), np.nextafter(np.float32(-1.0), np.float32(0.0))]
    # Where x * 2**31 rounds to 0 or to +-1 in the int32, and normal-but-tiny
    # values a decay tail can leave.
    half = np.float32(2.0**-32)
    edges += [np.nextafter(half, np.float32(0.0)), half, np.nextafter(half, np.float32(1.0)), -half]
    edges += [np.nextafter(-half, np.float32(-1.0)), 1e-37, -1e-37, 1e-20, -1e-20]
    for scale in (32767.0, 32768.0):
        for steps in (0.5, 1.5, 2.5, 100.5, 32766.5, 32767.5):
            edges += [steps / scale, -steps / scale]
    # One float32 ulp outside whole int16 steps: rounding x * 2**31 into the
    # int32 carries these to the step, a plain floor(x * 32768) does not.
    for steps in (1, 2, 17, 63, 64, 82, 127):
        step = np.float32(steps / 32768.0)
        edges += [np.nextafter(step, np.float32(0.0)), np.nextafter(-step, np.float32(-1.0))]
    rng = np.random.default_rng(0)
    return np.concatenate(
        [
            np.asarray(edges, dtype=np.float32),
            (rng.standard_normal(1500) * 0.3).astype(np.float32),
            rng.uniform(-1.2, 1.2, 1500).astype(np.float32),
        ]
    )


def _calibrate_pcm16(fmt: str, sample_rate_hz: int) -> str | None:
    """The quantizer that makes ``_encode_pcm16`` equal soundfile's output for *fmt* at this rate, if any."""
    probe = _calibration_probe()
    # Three lengths, so the WAV header's size fields are checked too.
    probes = (probe, probe[:1001], probe[:1])
    try:
        expected = [_encode_with_soundfile(samples, sample_rate_hz, fmt, 1.0) for samples in probes]
    except Exception:
        logger.info(
            "Duplex %s audio at %s Hz: soundfile rejected the probe; encoding stays on soundfile", fmt, sample_rate_hz
        )
        return None
    for mode in _PCM16_QUANTIZERS:
        encoded = [_encode_pcm16(samples, sample_rate_hz, fmt, mode) for samples in probes]
        if encoded == expected:
            return mode
    logger.info(
        "Duplex %s audio at %s Hz: numpy PCM_16 differs from soundfile; encoding stays on soundfile",
        fmt,
        sample_rate_hz,
    )
    return None


def install():
    from vllm_omni.entrypoints.duplex import runtime_bridge as rb
    M = rb.NativeRuntimeBridgeMixin
    if getattr(M, "_moshi_pcm", False):
        return False
    M._moshi_pcm = True
    orig = M._encode_native_data_plane_audio

    def enc(self, audio_data, sample_rate_hz, response_format, speed):
        if audio_data is None:
            return None
        try:
            import torch
            if isinstance(audio_data, torch.Tensor):
                a = audio_data.detach().cpu().float().numpy()
            else:
                a = np.asarray(audio_data, dtype=np.float32)
            if a.ndim > 1:
                a = a.reshape(-1)
            if _FALLBACK["fn"] is None:
                def fb(t, sr, fmt, sp, _self=self):
                    return orig(_self, t, sr, fmt, sp)
                _FALLBACK["fn"] = fb
            sp = float(speed) if isinstance(speed, int | float) and speed > 0 else 1.0
            if sp == 1.0 and a.size > 0 and isinstance(sample_rate_hz, int):
                fmt = str(response_format).lower()
                mode = _pcm16_mode(fmt, sample_rate_hz) if fmt in ("pcm", "wav") else None
                if mode is not None:
                    out = _encode_pcm16(a, sample_rate_hz, fmt, mode)
                    if out is not None:
                        return out
            return orig(self, a, sample_rate_hz, response_format, speed)
        except Exception:  # noqa: BLE001
            return orig(self, audio_data, sample_rate_hz, response_format, speed)

    M._encode_native_data_plane_audio = enc
    return True


def install_shm_probe():
    """#8405 part 2: SHM connector poll misses answered by os.access instead of a raising shm_open
    (MOSHI_SHM_PROBE=1)."""
    import os
    from vllm_omni.distributed.omni_connectors.connectors import shm_connector as sc
    C = sc.SharedMemoryConnector
    if getattr(C, "_moshi_probe", False) or not os.path.isdir("/dev/shm"):
        return False
    C._moshi_probe = True
    orig = C._get_by_key

    def get(self, get_key):
        if not os.access(f"/dev/shm/{get_key}", os.F_OK):
            return None
        return orig(self, get_key)

    C._get_by_key = get
    return True
