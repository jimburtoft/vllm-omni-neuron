"""Stage-0 -> stage-1 processors for Moshi, referenced by name from the deploy YAML (`engine_extras`), so they are
imported -- with the Moshi backports applied -- in the stage-0 engine-core process that runs the chunk adapter. With
TP>1 the workers are separate processes and never execute in that process, so patching from the worker is not enough.

* `moshi_async_chunk`: stock PersonaPlex de-delay processor + the text row (stage 1 Mimi-decodes).
* MOSHI_S0_DECODE=1: stage 0 already decoded the frame (Mimi decoder sharded across its TP ranks); the payload carries
  this frame's PCM (codes.ref, fp32 [1920]) plus the text token (codes.audio); stage 1 only forwards.
"""
import os

import torch

from .backports import apply as _apply

_apply()

from vllm_omni.data_entry_keys import CodesStruct, MetaStruct, OmniPayloadStruct  # noqa: E402
from vllm_omni.model_executor.stage_input_processors import personaplex as _pp  # noqa: E402


def pcm_passthrough_async_chunk(transfer_manager, multimodal_output, request, is_finished: bool = False):
    finished = bool(is_finished and not getattr(request, "resumable", False))
    pcm, txt = None, None
    for src in (getattr(request, "additional_information", None), multimodal_output):
        if not isinstance(src, dict):
            continue
        c = src.get("codes")
        r = (c.get("ref") if isinstance(c, dict) else None) if c is not None else src.get("codes.ref")
        a = (c.get("audio") if isinstance(c, dict) else None) if c is not None else src.get("codes.audio")
        if pcm is None and isinstance(r, torch.Tensor):
            pcm = r
        if txt is None and isinstance(a, torch.Tensor) and a.numel():
            txt = a.reshape(-1)[-1:].to(torch.long)
    if not isinstance(pcm, torch.Tensor) or pcm.numel() == 0:
        if finished:
            return OmniPayloadStruct(codes=CodesStruct(ref=torch.empty(0), audio=torch.tensor([-1])),
                                     meta=MetaStruct(finished=torch.tensor(True)))
        return OmniPayloadStruct(meta=MetaStruct(finished=torch.tensor(False), is_segment_finished=torch.tensor(False)))
    return OmniPayloadStruct(codes=CodesStruct(ref=pcm.reshape(-1).float(),
                                               audio=txt if txt is not None else torch.tensor([-1])),
                             meta=MetaStruct(finished=torch.tensor(finished)))


def moshi_async_chunk(transfer_manager, multimodal_output, request, is_finished: bool = False):
    if os.environ.get("MOSHI_S0_DECODE", "0") == "1":
        return pcm_passthrough_async_chunk(transfer_manager, multimodal_output, request, is_finished)
    return _pp.talker2code2wav_async_chunk(transfer_manager, multimodal_output, request, is_finished)
