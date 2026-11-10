"""Patches applied to the installed vllm-omni 0.29.0rc1 so the PersonaPlex duplex pipeline serves kyutai Moshi.

0.29's PersonaPlex serving adapter always builds the PersonaPlex voice + persona prefill, and Moshi checkpoints ship
no voice bundle, so every session open fails (`bundled voice prompt 'NATF2.pt' was not found`). Upstream
vllm-omni PR #9319 fixes this on main by treating an 8-step-depformer checkpoint as Moshi: no prefill, the session
starts from the initial-token frame, and voice / instructions are refused. We apply the same rule to the 0.29
adapter. `apply()` must run in every process that hosts the adapter (the API server, see serve_launcher).
"""
from __future__ import annotations

import functools
import os

import torch


@functools.lru_cache(maxsize=8)
def depformer_steps(model_path: str) -> int | None:
    from safetensors import safe_open
    from .common import model_dir
    d = model_dir(model_path)
    if not d or not os.path.isfile(os.path.join(d, "model.safetensors")):
        return None
    with safe_open(os.path.join(d, "model.safetensors"), framework="pt") as f:
        steps = {k.split(".")[1] for k in f.keys() if k.startswith("linears.")}
    return len(steps) or None


def _patch_serving_adapter():
    from vllm_omni.model_executor.models.personaplex.duplex import serving_adapter as sa
    cls = sa.PersonaPlexServingRuntimeAdapter
    if getattr(cls, "_moshi_backport", False):
        return False
    orig_prepare = cls.prepare_runtime_config.__func__
    orig_update = cls.runtime_config_for_update.__func__

    def _reject(config):
        v, ins = getattr(config, "voice", None), getattr(config, "instructions", None)
        if isinstance(v, str) and v.strip():
            raise sa.ServingRuntimeConfigError("Moshi has no voice prompts; leave voice unset")
        if isinstance(ins, str) and ins.strip():
            raise sa.ServingRuntimeConfigError("Moshi takes no instructions; leave instructions unset")

    async def prepare_runtime_config(c, config, *, model_config):
        model_path = getattr(model_config, "model", None)
        if isinstance(model_path, str) and depformer_steps(model_path) == 8:
            c.validate_client_extra_body(getattr(config, "extra_body", None))
            _reject(config)
            return {"personaplex_model_path": model_path, "personaplex_voice_prompt": None,
                    "personaplex_persona": None, "personaplex_prefill_slots": 0}
        return await orig_prepare(c, config, model_config=model_config)

    def runtime_config_for_update(c, config, current):
        if "personaplex_voice_prompt" in current and current["personaplex_voice_prompt"] is None:
            c.validate_client_extra_body(getattr(config, "extra_body", None))
            _reject(config)
            return dict(current)
        return orig_update(c, config, current)

    cls.prepare_runtime_config = classmethod(prepare_runtime_config)
    cls.runtime_config_for_update = classmethod(runtime_config_for_update)
    cls._moshi_backport = True
    return True


def _patch_text_channel():
    """Carry Moshi's text token to stage 1: the stage-0 row is [8 audio codes, text]. The stock async-chunk processor
    de-delays rows 0..7 and drops the rest; we append the text tokens of the emitted frames (text has delay 0, so
    output frame t takes the text of raw step t) as a ninth codebook-major row."""
    from vllm_omni.model_executor.stage_input_processors import personaplex as proc
    if getattr(proc.talker2code2wav_async_chunk, "_moshi_text", False):
        return False
    orig_dd = proc._agent_codes_to_codebook_major

    def dd_with_text(audio):
        flat = orig_dd(audio)
        if audio.ndim == 2 and audio.shape[1] == 9 and flat.numel():
            a = audio.to(torch.long)
            valid = (a[:, :8] >= 0).all(dim=1)
            txt = a[valid][:-1, 8]
            flat = torch.cat([flat, txt.reshape(-1)])
        return flat

    proc._agent_codes_to_codebook_major = dd_with_text
    proc.talker2code2wav_async_chunk._moshi_text = True
    return True


def _patch_data_plane_text():
    """Let the PersonaPlex data plane read text from `meta.llm_output_text_utf8` (uint8 bytes), the client-MM meta key
    stage 1 uses for Moshi's cumulative inner-monologue text."""
    from vllm_omni.model_executor.models.personaplex.duplex import data_plane as dp
    if getattr(dp._text_value, "_moshi_text", False):
        return False
    orig = dp._text_value

    def _decode(v):
        if isinstance(v, (list, tuple)) and v:
            v = v[-1]
        if isinstance(v, torch.Tensor):
            v = v.reshape(-1).to(torch.uint8).tolist()
        if isinstance(v, (list, bytes, bytearray)):
            try:
                return bytes(v).decode("utf-8", errors="ignore")
            except Exception:  # noqa: BLE001
                return ""
        return v if isinstance(v, str) else ""

    def text_value(multimodal, completion):
        t = orig(multimodal, completion)
        if t:
            return t
        meta = multimodal.get("meta") if isinstance(multimodal, dict) else None
        for v in (multimodal.get("meta.llm_output_text_utf8"),
                  meta.get("llm_output_text_utf8") if isinstance(meta, dict) else None):
            d = _decode(v) if v is not None else ""
            if d:
                return d
        return ""

    text_value._moshi_text = True
    dp._text_value = text_value
    # MOSHI_TEXT_DELTA=1: stage 1 sends "\x00" + only the new text; rebuild the cumulative text per request here
    # (cheap string append) so the stock _text_delta / cursor logic is unchanged.
    orig_proj = dp.PersonaPlexDataPlaneSession._project_output
    cur = {"acc": None, "rid": None}

    def _project_output(self, output, *, context):
        o, _ = dp._unwrap_output(output)
        cur["acc"] = self.__dict__.setdefault("_moshi_text_acc", {})
        cur["rid"] = getattr(o, "request_id", None)
        try:
            return orig_proj(self, output, context=context)
        finally:
            cur["acc"] = None

    def text_value2(multimodal, completion):
        t = text_value(multimodal, completion)
        if t.startswith("\x00") and cur["acc"] is not None:
            acc = cur["acc"]
            acc[cur["rid"]] = acc.get(cur["rid"], "") + t[1:]
            return acc[cur["rid"]]
        if t.startswith("\x00"):
            return t[1:]
        if t == "" and cur["acc"] is not None:
            return cur["acc"].get(cur["rid"], "")   # no new text this frame: cumulative unchanged
        return t
    text_value2._moshi_text = True
    dp._text_value = text_value2
    dp.PersonaPlexDataPlaneSession._project_output = _project_output
    return True


def _patch_pipeline_processor():
    """Point the PersonaPlex pipeline's stage-0 async-chunk processor at `moshi_neuron.serving.processors` so the
    stage-0 engine-core process imports our module (and its backports) itself -- needed when stage-0 workers are
    separate TP processes. Must run in the process that builds the stage configs (API server / launcher)."""
    import dataclasses
    from vllm_omni.model_executor.models.personaplex import pipeline as pl
    p = pl.PERSONAPLEX_PIPELINE
    s0 = p.stages[0]
    target = "moshi_neuron.serving.processors.moshi_async_chunk"
    if s0.async_chunk_process_next_stage_input_func == target:
        return False
    new0 = dataclasses.replace(s0, async_chunk_process_next_stage_input_func=target)
    pl.PERSONAPLEX_PIPELINE = dataclasses.replace(p, stages=(new0,) + tuple(p.stages[1:]))
    try:  # the registry may already hold the old object
        from vllm_omni.config import pipeline_registry as reg
        for name in dir(reg):
            obj = getattr(reg, name)
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if v is p:
                        obj[k] = pl.PERSONAPLEX_PIPELINE
    except Exception:  # noqa: BLE001
        pass
    return True


def apply():
    applied = []
    if os.environ.get("MOSHI_OQ_PROBE"):
        try:
            from .phases import install_output_queue_probe
            if install_output_queue_probe():
                applied.append("async_omni_engine:oq-probe")
        except Exception as e:  # noqa: BLE001
            applied.append(f"oq probe failed: {e!r}")
    if os.environ.get("MOSHI_DRAIN_BATCH", "0") == "1":
        try:
            from .phases import install_drain_batch
            if install_drain_batch():
                applied.append("duplex_request_client:drain-batch")
        except Exception as e:  # noqa: BLE001
            applied.append(f"drain batch failed: {e!r}")
    if os.environ.get("MOSHI_PCM_FAST", "0") == "1":
        try:
            from .audio_encoding_fast import install as _pcm
            if _pcm():
                applied.append("runtime_bridge:numpy-pcm16 (#8405)")
        except Exception as e:  # noqa: BLE001
            applied.append(f"pcm fast failed: {e!r}")
    if os.environ.get("MOSHI_SHM_PROBE", "0") == "1":
        try:
            from .audio_encoding_fast import install_shm_probe as _shp
            if _shp():
                applied.append("shm_connector:os.access-probe (#8405)")
        except Exception as e:  # noqa: BLE001
            applied.append(f"shm probe failed: {e!r}")
    if os.environ.get("MOSHI_API_TORCH_THREADS"):
        # #8703 part 2: cap torch intra-op threads in the API process (small host tensor ops otherwise keep a large
        # OpenMP pool spinning next to the stage processes). Only in the API-server process.
        try:
            if "serve_launcher" in open("/proc/self/cmdline", "rb").read().decode(errors="ignore"):
                import torch
                n = int(os.environ["MOSHI_API_TORCH_THREADS"])
                if 0 < n < torch.get_num_threads():
                    torch.set_num_threads(n)
                    applied.append(f"api torch threads={n} (#8703)")
        except Exception as e:  # noqa: BLE001
            applied.append(f"api torch threads failed: {e!r}")
    if os.environ.get("MOSHI_MARK_COALESCE", "0") == "1":
        try:
            from .phases import install_mark_coalesce
            if install_mark_coalesce():
                applied.append("duplex_protocol:mark-coalesce")
        except Exception as e:  # noqa: BLE001
            applied.append(f"mark coalesce failed: {e!r}")
    if os.environ.get("MOSHI_STAGE_EVENTS_CAP"):
        try:
            from .phases import install_stage_events_cap
            if install_stage_events_cap():
                applied.append("metrics:stage-events-cap")
        except Exception as e:  # noqa: BLE001
            applied.append(f"stage events cap failed: {e!r}")
    if os.environ.get("MOSHI_SNAP_FIX", "0") == "1":
        try:
            from .phases import install_snapshot_fix
            if install_snapshot_fix():
                applied.append("chunk_transfer_adapter:light-snapshot")
        except Exception as e:  # noqa: BLE001
            applied.append(f"snapshot fix failed: {e!r}")
    if os.environ.get("MOSHI_WS_NO_DEFLATE", "0") == "1":
        # permessage-deflate on every 80 ms audio delta was ~13% of the API event-loop thread at 28 sessions;
        # PCM barely compresses and the bandwidth is small (~64 KB/s per session base64), so do not negotiate it.
        try:
            import uvicorn.config as _uc
            _orig_init = _uc.Config.__init__

            def _init(self, *a, **k):
                k["ws_per_message_deflate"] = False
                _orig_init(self, *a, **k)
            if not getattr(_uc.Config, "_moshi_nd", False):
                _uc.Config.__init__ = _init
                _uc.Config._moshi_nd = True
                applied.append("uvicorn:ws-no-deflate")
        except Exception as e:  # noqa: BLE001
            applied.append(f"ws no-deflate failed: {e!r}")
    if os.environ.get("MOSHI_MARKS_FIX", "0") == "1":
        try:
            from .phases import install_realtime_marks_fix
            if install_realtime_marks_fix():
                applied.append("realtime_output:marks-fix")
        except Exception as e:  # noqa: BLE001
            applied.append(f"marks fix failed: {e!r}")
    if os.environ.get("MOSHI_FAST_APPEND", "0") == "1":
        try:
            from .phases import install_fast_append
            if install_fast_append():
                applied.append("async_omni_engine:fast-append")
        except Exception as e:  # noqa: BLE001
            applied.append(f"fast append failed: {e!r}")
    if os.environ.get("MOSHI_CONS_FIX", "0") == "1":
        try:
            from .phases import install_consolidate_fix
            if install_consolidate_fix():
                applied.append("mm_outputs:amortized-audio-append")
        except Exception as e:  # noqa: BLE001
            applied.append(f"consolidate fix failed: {e!r}")
    if os.environ.get("MOSHI_CONS_LOG") == "1":
        try:
            from .phases import install_consolidate_log
            if install_consolidate_log():
                applied.append("mm_outputs:consolidate-log")
        except Exception as e:  # noqa: BLE001
            applied.append(f"consolidate log failed: {e!r}")
    if os.environ.get("MOSHI_PHASES"):
        try:
            from .phases import install as _ph
            if _ph():
                applied.append("engine_core:phase-timing")
        except Exception as e:  # noqa: BLE001
            applied.append(f"phase timing failed: {e!r}")
    if os.environ.get("MOSHI_FRAMESYNC", "0") == "1":
        try:
            from .framesync import install
            if install():
                applied.append("engine_core:frame-sync-gate")
        except Exception as e:  # noqa: BLE001
            applied.append(f"frame-sync gate failed: {e!r}")
    try:
        if _patch_pipeline_processor():
            applied.append("personaplex.pipeline:moshi-processor")
    except Exception as e:  # noqa: BLE001
        applied.append(f"pipeline processor backport failed: {e!r}")
    try:
        if _patch_data_plane_text():
            applied.append("personaplex.data_plane:text-from-meta")
    except Exception as e:  # noqa: BLE001
        applied.append(f"data plane text backport failed: {e!r}")
    try:
        if _patch_text_channel():
            applied.append("personaplex.processor:text-row")
    except Exception as e:  # noqa: BLE001
        applied.append(f"text channel backport failed: {e!r}")
    try:
        if _patch_serving_adapter():
            applied.append("personaplex.serving_adapter:moshi-no-prefill")
    except Exception as e:  # never break serving because of a backport
        applied.append(f"serving_adapter backport failed: {e!r}")
    return applied
