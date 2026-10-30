"""Checkpoint location and fixed Moshi / Mimi hyper-parameters (moshiko / moshika 7B, Mimi e351c8d8)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

DEFAULT_REPO = os.environ.get("MOSHI_HF_REPO", "kyutai/moshiko-pytorch-bf16")
MOSHI_NAME = "model.safetensors"
MIMI_NAME = "tokenizer-e351c8d8-checkpoint125.safetensors"
TOKENIZER_NAME = "tokenizer_spm_32k_3.model"

SAMPLE_RATE = 24000
FRAME_RATE = 12.5
FRAME = 1920  # samples per 80 ms frame


@dataclass
class LMConfig:
    dim: int = 4096
    num_heads: int = 32
    num_layers: int = 32
    hidden: int = 11264  # gated FFN hidden (2/3 of 4.125*dim)
    context: int = 3000
    text_card: int = 32000
    card: int = 2048
    n_q: int = 16
    dep_q: int = 8
    dep_dim: int = 1024
    dep_heads: int = 16
    dep_layers: int = 6
    dep_hidden: int = 2816
    max_period: float = 10000.0
    eps: float = 1e-8
    delays: list = field(default_factory=lambda: [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1])

    @property
    def head_dim(self):
        return self.dim // self.num_heads

    @property
    def num_codebooks(self):
        return self.n_q + 1


@dataclass
class MimiConfig:
    n_filters: int = 64
    dimension: int = 512
    ratios: tuple = (8, 6, 5, 4)  # decoder order; encoder uses reversed
    kernel_size: int = 7
    last_kernel_size: int = 3
    residual_kernel_size: int = 3
    compress: int = 2
    tr_layers: int = 8
    tr_heads: int = 8
    tr_ffn: int = 2048
    tr_context: int = 250
    max_period: float = 10000.0
    num_codebooks: int = 8
    n_q_semantic: int = 1
    bins: int = 2048
    vq_dim: int = 256


def checkpoint_paths(repo: str = DEFAULT_REPO, model_dir: str | None = None, need_lm: bool = True):
    """Return (moshi_weights, mimi_weights, tokenizer) local paths.

    Order: MOSHI_WEIGHTS / MIMI_WEIGHTS / MOSHI_TOKENIZER env overrides, then files in `model_dir` (a local
    snapshot, e.g. the serving model directory), then the HF Hub repo (downloaded if needed)."""
    from huggingface_hub import hf_hub_download

    def get(env, name):
        p = os.environ.get(env)
        if p:
            return p
        if model_dir and os.path.isfile(os.path.join(model_dir, name)):
            return os.path.join(model_dir, name)
        if not need_lm and name == MOSHI_NAME:
            return None
        return hf_hub_download(repo, name)

    return get("MOSHI_WEIGHTS", MOSHI_NAME), get("MIMI_WEIGHTS", MIMI_NAME), get("MOSHI_TOKENIZER", TOKENIZER_NAME)
