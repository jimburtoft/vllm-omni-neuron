"""`tokenizers`-only tokenizer must be token-identical to transformers' AutoTokenizer."""
import pytest

from conftest import CASES
from qwen3_tts_neuron.utils import assistant_text, instruct_text

EXTRA = ["Hello, world!  Multiple   spaces\nand newlines.\n\n", "数字123和English混合。", "It's  don't  we've ...?!",
         "Émile a mangé des crêpes — très bien.", "  leading and trailing  "]


def test_tokenizer_matches_hf(model_dir, monkeypatch):
    transformers = pytest.importorskip("transformers")
    hf = transformers.AutoTokenizer.from_pretrained(model_dir)
    monkeypatch.setenv("QTTS_TOKENIZER", "fast")
    from qwen3_tts_neuron.tokenizer import TextTokenizer
    tok = TextTokenizer(model_dir)
    assert tok.backend == "tokenizers"
    texts = [assistant_text(c["text"]) for c in CASES] + [instruct_text(c["instruct"]) for c in CASES] + EXTRA
    for t in texts:
        assert tok.encode(t) == hf(t, padding=False)["input_ids"], t
