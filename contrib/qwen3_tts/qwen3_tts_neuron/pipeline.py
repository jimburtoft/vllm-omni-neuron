"""High-level text-in / audio-out API for Qwen3-TTS on Neuron (no vLLM needed).

    from qwen3_tts_neuron.pipeline import Qwen3TTSNeuron
    tts = Qwen3TTSNeuron("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", batch_size=8)
    wavs, sr = tts.generate_voice_design(["Hello there."], instruct=["Calm female voice."], language=["English"])

Graphs: fused frame (talker decode + code predictor + on-device sampling) for `batch_size` streams with continuous
batching, slot prefill per bucket, Code2Wav windows. `device="cpu"` runs the same modules eagerly (fp32) for tests.
"""
from __future__ import annotations

import os

import torch

from .tokenizer import TextTokenizer
from .utils import DEFAULT_MODEL, assistant_text, instruct_text, model_path


def _compile_backend(device: str):
    """`torch.compile` backend for the active venv: torch-neuronx ("neuron") or libtorch-neuronx-lite (native)."""
    if device == "cpu":
        return None
    try:
        import torch_neuronx  # noqa: F401
        return "neuron"
    except ImportError:
        from libtorch_neuronx_lite.compile.native_backend import register
        return register()


class Qwen3TTSNeuron:
    def __init__(self, model: str = DEFAULT_MODEL, device: str | None = None, batch_size: int = 8,
                 slot_len: int = 512, prefill_buckets=(64, 128, 256), dtype=None, sampling: dict | None = None,
                 seed: int = 0, vocoder: bool = True):
        from .code2wav import Code2WavRunner
        from .engine import TTSEngine
        from .fused import FusedEngine

        self.path = model_path(model)
        if device is None:
            device = "neuron" if _compile_backend("neuron") == "neuron" else "neuron:0"
        self.device = device
        cpu = device == "cpu"
        dtype = dtype or (torch.float32 if cpu else torch.bfloat16)
        backend = _compile_backend(device)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 256)
        orig = torch.compile
        if backend not in (None, "neuron"):  # route the modules' torch.compile(backend="neuron") calls to lite
            torch.compile = lambda m, backend_=None, dynamic=None, **k: orig(m, backend=backend, dynamic=False)
        try:
            self.eng = TTSEngine(self.path, device=device, max_ctx=slot_len, dtype=dtype, compile=False,
                                 prefill_buckets=prefill_buckets)
            self.fe = FusedEngine(self.eng, batch_size, S=slot_len, compile=not cpu, sampling=sampling, seed=seed)
            self.c2w = Code2WavRunner(os.path.join(self.path, "speech_tokenizer"), device=device, dtype=dtype,
                                      compile=not cpu) if vocoder else None
        finally:
            torch.compile = orig
        self.tok = TextTokenizer(self.path)
        self.sample_rate = 24000

    # ------------------------------------------------------------------ prompts
    def prompt(self, text: str, instruct: str | None = None, language: str | None = "Auto"):
        ids = torch.tensor([self.tok.encode(assistant_text(text))])
        iids = torch.tensor([self.tok.encode(instruct_text(instruct))]) if instruct and instruct.strip() else None
        with torch.no_grad():
            return self.eng.build_prompt(ids, iids, language or "Auto")

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_codes(self, texts, instruct=None, language=None, max_new: int = 400):
        n = len(texts)
        instruct = instruct or [None] * n
        language = language or ["Auto"] * n
        reqs = [self.prompt(t, i, l) for t, i, l in zip(texts, instruct, language)]
        out, wall, steps = self.fe.run(reqs, max_new=max_new)
        return [o["codes"] for o in out]

    @torch.no_grad()
    def generate_voice_design(self, texts, instruct=None, language=None, max_new: int = 400):
        assert self.c2w is not None, "constructed with vocoder=False"
        codes = self.generate_codes(texts, instruct, language, max_new)
        return [self.c2w.decode(c) if len(c) else torch.zeros(0) for c in codes], self.sample_rate
