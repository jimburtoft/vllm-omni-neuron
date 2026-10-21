"""Backports applied at import time to the installed vllm-omni (0.29.0rc1) when it lacks a fix.

1. `talker2code2wav_async_chunk` re-sends the final chunk when a request ends exactly on a chunk boundary
   (finished flush with `length % chunk == 0` -> `context_length = chunk`), so the vocoder decodes ~2 s twice.
   Upstream vllm-omni main fixes this by tracking frames already emitted per request
   (`_qwen3_tts_emitted_frames`). We wrap the installed processor with the same logic.
"""
from __future__ import annotations

import inspect

import torch


def _patch_async_chunk_dup():
    from vllm_omni.model_executor.stage_input_processors import qwen3_tts as proc
    if "_qwen3_tts_emitted_frames" in inspect.getsource(proc.talker2code2wav_async_chunk):
        return False  # installed version already has the fix
    if getattr(proc.talker2code2wav_async_chunk, "_qtts_backport", False):
        return False
    orig = proc.talker2code2wav_async_chunk
    from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct

    def wrapped(transfer_manager, multimodal_output, request, is_finished=False):
        rid = request.external_req_id
        emitted = getattr(transfer_manager, "_qtts_emitted_frames", None)
        if emitted is None:
            emitted = transfer_manager._qtts_emitted_frames = {}
        out = orig(transfer_manager, multimodal_output, request, is_finished=is_finished)
        length = len(transfer_manager.code_prompt_token_ids.get(rid, []))
        if out is None or out.codes is None or not isinstance(out.codes.audio, torch.Tensor):
            return out
        n = out.codes.audio.numel()
        prev = emitted.get(rid, 0)
        if n and prev >= length:  # nothing new since the last emit: send an empty finish marker instead
            finished = bool(out.meta is not None and out.meta.finished is not None and bool(out.meta.finished))
            if not finished:
                return None
            return OmniPayloadStruct(codes=CodesStruct(audio=torch.empty(0, dtype=torch.long)),
                                     meta=MetaStruct(request_id=rid, left_context_size=0,
                                                     finished=torch.tensor(True, dtype=torch.bool)))
        if n:
            emitted[rid] = length
        if out.meta is not None and out.meta.finished is not None and bool(out.meta.finished):
            emitted.pop(rid, None)
        return out

    import functools
    functools.update_wrapper(wrapped, orig)
    wrapped.__signature__ = inspect.signature(orig)  # the chunk adapter introspects the processor signature
    wrapped._qtts_backport = True
    proc.talker2code2wav_async_chunk = wrapped
    return True


def apply():
    applied = []
    try:
        if _patch_async_chunk_dup():
            applied.append("talker2code2wav_async_chunk:dedup-final-chunk")
    except Exception:  # never break serving because of a backport
        pass
    return applied
