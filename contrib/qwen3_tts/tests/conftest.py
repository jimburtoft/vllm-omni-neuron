import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = os.environ.get("QTTS_MODEL", "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")

CASES = [
    dict(tag="en_calm", text="She said she would be here by noon, and she always keeps her word.",
         language="English", instruct="Calm, warm adult female voice, steady pace, gentle and reassuring."),
    dict(tag="en_excited", text="No way! Did you really win the whole tournament on your first try?",
         language="English", instruct="Energetic young male voice, bright timbre, fast and excited."),
    dict(tag="zh_soft", text="哥哥，你回来啦，人家等了你好久好久了。",
         language="Chinese", instruct="体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显。"),
]


@pytest.fixture(scope="session")
def model_dir():
    from qwen3_tts_neuron.utils import model_path
    return model_path(MODEL)
