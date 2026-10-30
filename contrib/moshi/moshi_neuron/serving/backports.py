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


def apply():
    applied = []
    try:
        if _patch_serving_adapter():
            applied.append("personaplex.serving_adapter:moshi-no-prefill")
    except Exception as e:  # never break serving because of a backport
        applied.append(f"serving_adapter backport failed: {e!r}")
    return applied
