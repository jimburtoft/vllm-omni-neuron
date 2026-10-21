"""Small shared helpers."""
from __future__ import annotations

import os

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


def model_path(name_or_path: str = DEFAULT_MODEL) -> str:
    """Local directory for a HF repo id or path (downloads on first use)."""
    if os.path.isdir(name_or_path):
        return name_or_path
    from huggingface_hub import snapshot_download
    return snapshot_download(name_or_path)


def assistant_text(text: str) -> str:
    return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"


def instruct_text(instruct: str) -> str:
    return f"<|im_start|>user\n{instruct}<|im_end|>\n"
