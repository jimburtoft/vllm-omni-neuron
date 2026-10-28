"""Talker prompt construction for all Qwen3-TTS task types, reusing vllm-omni's `Qwen3TTSPromptEmbedsBuilder`.

The builder is the same code the GPU talker uses (VoiceDesign, CustomVoice, Base x-vector / ICL, streaming and
non-streaming text, uploaded voices through the speaker cache, ref-audio artifact cache). We hand it host-side views
of our engine's embedding tables, so prompt construction runs on CPU and only the resulting embeddings go to the
device. Base checkpoints additionally load the ECAPA speaker encoder and the speech-tokenizer encoder (CPU, fp32,
lazily on the first voice-clone request).
"""
from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)


class _Ids:
    """Token ids passed from the cached text-embedding stand-in to the cached projection."""

    def __init__(self, ids):
        self.ids = ids


class _NoSpeakerEncoder(nn.Module):
    def forward(self, *_a, **_k):
        raise ValueError("this checkpoint has no speaker encoder (voice clone needs a Base checkpoint)")


class _LazyCodecEncoder:
    """Speech-tokenizer encoder (wav -> [T,16] codes) for Base ICL prompts; CPU fp32."""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self._enc = self._fe = None
        self._lock = threading.Lock()

    def _load(self):
        from safetensors import safe_open
        from transformers import AutoFeatureExtractor
        from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.configuration_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Config)
        from vllm_omni.model_executor.models.qwen3_tts.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import (
            Qwen3TTSTokenizerV2Encoder)
        tc = Qwen3TTSTokenizerV2Config.from_pretrained(self.model_path, subfolder="speech_tokenizer")
        enc = Qwen3TTSTokenizerV2Encoder._from_config(tc.encoder_config)
        f = safe_open(os.path.join(self.model_path, "speech_tokenizer", "model.safetensors"), "pt")
        sd = {k[len("encoder."):]: f.get_tensor(k) for k in f.keys() if k.startswith("encoder.")}
        enc.load_state_dict(sd, strict=True)
        self._enc = enc.eval().float()
        self._nq, self._down = int(tc.encoder_valid_num_quantizers), int(tc.encode_downsample_rate)
        self._fe = AutoFeatureExtractor.from_pretrained(self.model_path, subfolder="speech_tokenizer")

    def __call__(self, wavs, sr, *, device):
        import numpy as np
        with self._lock:
            if self._enc is None:
                self._load()
        fe = self._fe
        target = int(fe.sampling_rate)
        if int(sr) != target:
            from vllm.multimodal.audio import AudioResampler
            rs = AudioResampler(target_sr=target)
            wavs = [rs.resample(np.asarray(w, dtype=np.float32), orig_sr=int(sr)) for w in wavs]
        inp = fe(raw_audio=wavs, sampling_rate=target, return_tensors="pt")
        x, mask = inp["input_values"].squeeze(1).float(), inp["padding_mask"].squeeze(1)
        with torch.inference_mode():
            codes = self._enc.encode(input_values=x.unsqueeze(1), return_dict=True).audio_codes[:, : self._nq]
        return [c[..., : -(-int(m.sum()) // self._down)].transpose(0, 1).to(device=device, dtype=torch.long)
                for c, m in zip(codes, mask)]


class PromptBuilder:
    """build(info) -> (prompt [1,T,H], trailing [N,H], tts_pad [1,1,H], ref_code [R,16] | None), all CPU tensors."""

    def __init__(self, eng, model_path: str):
        from vllm_omni.model_executor.models.qwen3_tts.configuration_qwen3_tts import Qwen3TTSConfig
        from vllm_omni.model_executor.models.qwen3_tts.prompt_embeds_builder import Qwen3TTSPromptEmbedsBuilder
        self.eng, self.model_path = eng, model_path
        cfg = Qwen3TTSConfig.from_pretrained(model_path)
        self.cfg, self.model_type = cfg, str(getattr(cfg, "tts_model_type", "") or "")
        hp = torch.float32  # prompt math in fp32 on host; cast to the engine dtype at the end
        self._text_table = nn.Embedding.from_pretrained(eng.text_emb.to(hp), freeze=True)
        w1, b1, w2, b2 = (t.to(hp) for t in (eng.tp_w1, eng.tp_b1, eng.tp_w2, eng.tp_b2))
        raw_proj = lambda e: F.linear(F.silu(F.linear(e.to(hp), w1, b1)), w2, b2)
        # The text projection is a per-position MLP over a fixed table, so proj(text_emb(ids)) is a per-token lookup.
        # Cache it by token id (exact; ~6 ms of fp32 matmuls per prompt otherwise, on the talker's critical path).
        self._tp_cache: dict[int, torch.Tensor] = {}
        self._tp_cache_max = int(os.environ.get("QTTS_TEXT_PROJ_CACHE", "65536"))
        table = self._text_table

        class _CachedText(nn.Module):
            """text_embedding stand-in that returns token ids tagged for the cached projection."""

            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(table.weight, requires_grad=False)  # builder reads .parameters() for device

            def forward(self, ids):
                return _Ids(ids)

        def proj(e):
            if not isinstance(e, _Ids):
                return raw_proj(e)
            ids = e.ids
            flat = ids.reshape(-1).tolist()
            miss = sorted({t for t in flat if t not in self._tp_cache})
            if miss:
                rows = raw_proj(F.embedding(torch.tensor(miss), table.weight))
                if len(self._tp_cache) + len(miss) > self._tp_cache_max:
                    self._tp_cache.clear()
                for t, r in zip(miss, rows):
                    self._tp_cache[t] = r
            return torch.stack([self._tp_cache[t] for t in flat]).view(*ids.shape, -1)
        text_mod = _CachedText()
        codec = eng.codec_emb.to(hp)
        codec_embed = lambda ids: F.embedding(ids.to(torch.long), codec)
        self._res = None

        def residual():  # code-predictor codec embeddings (ICL ref codes only); host copy loaded on first use
            if self._res is None:
                from ..engine import SafeTensors
                st = SafeTensors(os.path.join(model_path, "model.safetensors"))
                tabs = [st.get(f"talker.code_predictor.model.codec_embedding.{i}.weight").to(hp) for i in range(15)]
                self._res = [(lambda ids, t=t: F.embedding(ids.to(torch.long), t)) for t in tabs]
            return self._res

        spk = _NoSpeakerEncoder()
        if self.model_type == "base":
            from safetensors import safe_open
            from vllm_omni.model_executor.models.qwen3_tts.qwen3_tts_talker import Qwen3TTSSpeakerEncoder
            spk = Qwen3TTSSpeakerEncoder(cfg.speaker_encoder_config)
            f = safe_open(os.path.join(model_path, "model.safetensors"), "pt")
            spk.load_state_dict({k[len("speaker_encoder."):]: f.get_tensor(k) for k in f.keys()
                                 if k.startswith("speaker_encoder.")}, strict=True)
            spk = spk.eval().to(hp)
        pad_id = torch.tensor([[cfg.tts_pad_token_id]])
        with torch.no_grad():
            self.tts_pad = proj(self._text_table(pad_id)).reshape(1, -1)
        try:
            from vllm_omni.utils.speaker_cache import get_speaker_cache
            cache = get_speaker_cache()
        except Exception:
            cache = None
        self.b = Qwen3TTSPromptEmbedsBuilder(
            config=cfg, talker_config=cfg.talker_config, model_path=model_path, text_embedding=text_mod,
            text_projection=proj, codec_embed=codec_embed, residual_code_embeddings=residual, speaker_encoder=spk,
            tts_pad_embed=self.tts_pad, encode_ref_audio_batch=_LazyCodecEncoder(model_path), speaker_cache=cache)
        self.b._embedding_dtype = hp
        if self.model_type == "base":  # load the codec encoder and code-predictor tables now, not on the first request
            self.b._encode_ref_audio_batch_fn._load()
            residual()

    @torch.no_grad()
    def build(self, info: dict):
        from .common import info_first
        task = info_first(info, "task_type", None) or {"base": "Base", "custom_voice": "CustomVoice"}.get(
            self.model_type, "VoiceDesign")
        if task == "Base" and self.model_type not in ("base", ""):
            raise ValueError(f"task_type=Base needs a Base checkpoint (loaded: {self.model_type})")
        if task == "CustomVoice" and self.model_type not in ("custom_voice", ""):
            raise ValueError(f"task_type=CustomVoice needs a CustomVoice checkpoint (loaded: {self.model_type})")
        dt = self.eng.dtype
        prompt, trailing, ref_len, ref_code = self.b.build_prompt_embeds(task_type=task, info_dict=dict(info))
        return (prompt.unsqueeze(0).to(dt), trailing.reshape(-1, prompt.shape[-1]).to(dt),
                self.tts_pad.reshape(1, 1, -1).to(dt), ref_code)
