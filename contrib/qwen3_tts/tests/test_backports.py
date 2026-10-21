"""The async-chunk processor must not re-send the final chunk when a request ends on a chunk boundary."""
from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm_omni")


def _tm():
    return SimpleNamespace(code_prompt_token_ids=defaultdict(list), put_req_chunk=defaultdict(int),
                           ramp_chunk_count=defaultdict(int), request_payload={}, scheduler_max_num_seqs=8,
                           connector=SimpleNamespace(config={"extra": {"codec_chunk_frames": 25,
                                                                        "initial_codec_chunk_frames": 1}}))


def _drive(fn, n_frames):
    tm, sent = _tm(), []
    req = SimpleNamespace(external_req_id="r0", additional_information=None, is_finished=lambda: False)
    for i in range(n_frames + 1):
        fin = i == n_frames
        mm = {"codes": {"audio": torch.full((1, 16), 5 + i, dtype=torch.long)}} if not fin else None
        if fin:
            req.is_finished = lambda: True
        out = fn(tm, mm, req, is_finished=fin)
        if out is not None and out.codes is not None and out.codes.audio.numel():
            sent.append(out.codes.audio.numel() // 16)
            tm.put_req_chunk["r0"] += 1
    return sent


def test_no_duplicate_final_chunk():
    from qwen3_tts_neuron.serving.backports import apply
    apply()
    from vllm_omni.model_executor.stage_input_processors import qwen3_tts as proc
    for n in (26, 51, 30, 1):  # 26 and 51 end exactly on a 25-frame boundary after the 1-frame first chunk
        sent = _drive(proc.talker2code2wav_async_chunk, n)
        assert sum(sent) == n, (n, sent)
