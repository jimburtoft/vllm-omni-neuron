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
