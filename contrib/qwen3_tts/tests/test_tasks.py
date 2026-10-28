"""CustomVoice / Base (x-vector, ICL) / streaming-text prompts and greedy codes vs the HF reference (fp32 CPU).

Needs a dump from `tests/hf_task_dump.py` (run in a venv with the official `qwen_tts` package):
    QTTS_TASK_DUMP=/path/dump.pt QTTS_CV_MODEL=<CustomVoice dir> QTTS_BASE_MODEL=<Base dir> pytest tests/test_tasks.py
"""
import os

import pytest
import torch

pytest.importorskip("vllm_omni")
DUMP = os.environ.get("QTTS_TASK_DUMP")
pytestmark = pytest.mark.skipif(not (DUMP and os.path.exists(DUMP)), reason="QTTS_TASK_DUMP not set")


def _builder(path):
    from qwen3_tts_neuron.engine import TTSEngine
    from qwen3_tts_neuron.serving.prompts import PromptBuilder
    eng = TTSEngine(path, device="cpu", max_ctx=512, dtype=torch.float32, compile=False, prefill_buckets=(64, 128, 256))
    return eng, PromptBuilder(eng, path)


@pytest.fixture(scope="module")
def dump():
    return torch.load(DUMP, weights_only=False)


@pytest.fixture(scope="module")
def cv():
    return _builder(os.environ["QTTS_CV_MODEL"])


@pytest.fixture(scope="module")
def base():
    return _builder(os.environ["QTTS_BASE_MODEL"])


def _check(eng, pb, ref, info, ref_frames=0):
    emb, trailing, pad, ref_code = pb.build(info)
    assert emb.shape[1:] == ref["emb"].shape, (emb.shape, ref["emb"].shape)
    assert (emb[0] - ref["emb"]).abs().max() < 2e-3
    assert (pad.reshape(-1) - ref["pad"]).abs().max() < 1e-4
    assert trailing.shape == ref["trailing"].shape, (trailing.shape, ref["trailing"].shape)
    assert (trailing - ref["trailing"]).abs().max() < 2e-3
    tr = None if (trailing.shape[0] == 1 and torch.allclose(trailing[0], pad.reshape(-1))) else trailing
    codes = eng.generate(emb, pad, max_new=60, trailing=tr)
    if isinstance(ref_code, torch.Tensor):  # HF decodes cat(ref_code, codes)
        ref_frames = ref_code.shape[0]
    want = ref["codes"][ref_frames:]
    n = min(len(codes), len(want))
    match = (codes[:n] == want[:n]).all(-1).float().mean().item()
    capped = len(want) >= 59  # HF hit max_new_tokens=60
    assert match == 1.0 and (capped and len(codes) >= len(want) or len(codes) == len(want)), (len(codes), len(want), match)
    return ref_code


def _info(d, **kw):
    out = {"text": [d["text"]], "language": ["English"]}
    out.update({k: [v] for k, v in kw.items()})
    return out


@pytest.mark.parametrize("tag,kw", [
    ("cv_ryan_ns", dict(speaker="ryan")),
    ("cv_vivian_instr_ns", dict(speaker="vivian", instruct="Speak slowly and softly.")),
    ("cv_ryan_stream", dict(speaker="ryan", non_streaming_mode=False)),
])
def test_custom_voice(dump, cv, tag, kw):
    eng, pb = cv
    _check(eng, pb, dump[tag], _info(dump, task_type="CustomVoice", **kw))


def _ref_audio(dump):
    w, sr = dump["ref_audio"]
    return [w.numpy().tolist(), sr]


@pytest.mark.parametrize("tag,kw", [
    ("base_xvec", dict(x_vector_only_mode=True)),
    ("base_icl", dict()),
    ("base_icl_ns", dict(non_streaming_mode=True)),
])
def test_base(dump, base, tag, kw):
    eng, pb = base
    icl = not kw.get("x_vector_only_mode")
    info = _info(dump, task_type="Base", **kw)
    info["ref_audio"] = [_ref_audio(dump)]
    if icl:
        info["ref_text"] = [dump["ref_text"]]
    ref_code = _check(eng, pb, dump[tag], info, ref_frames=0)
    assert (ref_code is not None) == icl
