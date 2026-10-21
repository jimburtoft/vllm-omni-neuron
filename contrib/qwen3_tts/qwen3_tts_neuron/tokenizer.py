"""Text tokenizer for Qwen3-TTS without a `transformers` dependency.

The checkpoint ships only `vocab.json` + `merges.txt` (Qwen2 byte-level BPE). We build an equivalent fast tokenizer with
the `tokenizers` library (present in both the torch-neuronx and vLLM-Omni DLAMI venvs), using the Qwen2 pre-tokenizer
regex and the special tokens from `tokenizer_config.json`. If `transformers` is importable we can use it instead
(`QTTS_TOKENIZER=hf`). Verified token-identical to `AutoTokenizer` on the test prompts (tests/test_tokenizer.py).
"""
from __future__ import annotations

import json
import os

# Qwen2 pre-tokenization pattern (transformers Qwen2TokenizerFast / tokenization_qwen2.PRETOKENIZE_REGEX)
QWEN2_PATTERN = (r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"""
                 r"""\s+(?!\S)|\s+""")


class TextTokenizer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        mode = os.environ.get("QTTS_TOKENIZER", "auto")
        self._hf = None
        if mode in ("hf", "auto"):
            try:
                from transformers import AutoTokenizer
                self._hf = AutoTokenizer.from_pretrained(model_dir)
            except Exception:
                if mode == "hf":
                    raise
        if self._hf is None:
            self._tok = self._build_fast(model_dir)

    @staticmethod
    def _build_fast(model_dir):
        from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers
        vocab = json.load(open(os.path.join(model_dir, "vocab.json"), encoding="utf-8"))
        merges = []
        with open(os.path.join(model_dir, "merges.txt"), encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#version"):
                    continue
                a, b = line.split(" ")
                merges.append((a, b))
        tok = Tokenizer(models.BPE(vocab=vocab, merges=merges, byte_fallback=False, unk_token=None))
        tok.pre_tokenizer = pre_tokenizers.Sequence([
            pre_tokenizers.Split(Regex(QWEN2_PATTERN), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
        ])
        tok.decoder = decoders.ByteLevel()
        cfg = json.load(open(os.path.join(model_dir, "tokenizer_config.json"), encoding="utf-8"))
        added = sorted(((int(i), d) for i, d in cfg.get("added_tokens_decoder", {}).items()), key=lambda x: x[0])
        tok.add_special_tokens([AddedToken(d["content"], special=True, normalized=False, lstrip=False, rstrip=False,
                                           single_word=False) for _, d in added])
        for i, d in added:  # ids must match the checkpoint
            assert tok.token_to_id(d["content"]) == i, (d["content"], tok.token_to_id(d["content"]), i)
        return tok

    def encode(self, text: str) -> list[int]:
        if self._hf is not None:
            return self._hf(text, padding=False)["input_ids"]
        return self._tok.encode(text, add_special_tokens=False).ids

    @property
    def backend(self) -> str:
        return "transformers" if self._hf is not None else "tokenizers"
