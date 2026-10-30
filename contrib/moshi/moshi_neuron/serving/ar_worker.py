"""Stage 0 (LLM_AR) Neuron worker for Moshi on the vLLM-Omni PersonaPlex duplex pipeline.

Duplex protocol of vllm-omni 0.29.0rc1 (`personaplex/duplex/runtime_extension.py`): every 80 ms user frame is one
resumable segment of the session's stage-0 request, with `max_tokens=1` and the frame's PCM in
`model_intermediate_buffer["duplex"]["payload"]` (base64 pcm_f32le, 1920 samples). For each segment the worker
Mimi-encodes the frame, runs one Moshi LM frame (Helium + depformer, one NEFF for all B rows), and returns
  sampled token  = the text token (inner monologue)
  inter_stage    = {"codes.audio": [1, 8] raw depformer codes of this step}
The stock `talker2code2wav_async_chunk` processor de-delays the agent codes for stage 1.

Moshi (8-step depformer) follows `moshi.server`: no voice/persona prefill, the session starts from the initial-token
frame, and the stage-0 KV and encoder state are model-owned per row. `get_kv_cache_spec()` returns {}.

Offline requests (no `duplex` key) carry the whole user wav in `additional_information["moshi_user_pcm"]`
(float32 [N]); the worker runs one frame per scheduler step until the wav is consumed or `max_tokens` is reached.
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin, needs_omni_connector

from ..config import FRAME
from .common import compile_fn, cpu_mode, model_dir, neuron_runtime_env, torch_device

logger = init_logger(__name__)
EOS_TEXT = 2  # never sampled in duplex use; reported to stop an offline request


def _info(new_req) -> dict:
    buf = getattr(new_req, "model_intermediate_buffer", None)
    if isinstance(buf, dict) and buf:
        return buf
    info = getattr(new_req, "additional_information", None)
    if info is None:
        return {}
    from vllm_omni.engine.serialization import deserialize_additional_information
    return deserialize_additional_information(info) or {}


def _pcm_from_payload(payload) -> torch.Tensor:
    if not isinstance(payload, dict) or payload.get("format") != "pcm_f32le":
        raise ValueError("Moshi stage 0 expects pcm_f32le payloads")
    raw = base64.b64decode(payload["audio"])
    x = np.frombuffer(raw, dtype="<f4")
    if x.size != FRAME:
        raise ValueError(f"Moshi stage 0 expects {FRAME} samples per append, got {x.size}")
    return torch.from_numpy(x.copy())


class MoshiStage0Runner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.B = int(vllm_config.scheduler_config.max_num_seqs)
        self.requests: dict[str, dict] = {}
        self.model_intermediate_buffer: dict[str, Any] = {}
        self.slot_of: dict[str, int] = {}          # stage-0 request id -> row
        self.session_of: dict[str, tuple] = {}     # request id -> (session_id, incarnation)
        self.row_of_session: dict[tuple, int] = {}
        self.free = list(range(self.B))
        self.offline: dict[str, dict] = {}         # rid -> dict(pcm, t, max)
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    # ------------------------------------------------------------------ load / warmup
    def load_model(self):
        from ..engine import MoshiStage0
        t0 = time.time()
        greedy = os.environ.get("MOSHI_GREEDY", "0") == "1"
        self.s0 = MoshiStage0(B=self.B, device=self.device, compile_fn=compile_fn(), greedy=greedy,
                              seed=int(getattr(self.model_config, "seed", 0) or 0),
                              model_dir=model_dir(self.model_config.model))
        logger.info("[MoshiS0] loaded in %.1fs (B=%d greedy=%s device=%s)", time.time() - t0, self.B, greedy,
                    self.device)

    @torch.inference_mode()
    def warmup(self):
        t0 = time.time()
        z = torch.zeros(self.B, FRAME)
        for _ in range(3):
            self.s0.step(z, torch.ones(self.B, dtype=torch.bool))
        self.s0.reset(range(self.B))
        logger.info("[MoshiS0] warmup %.1fs", time.time() - t0)
        return time.time() - t0

    # ------------------------------------------------------------------ rows
    def _row_for(self, rid, info) -> int:
        if rid in self.slot_of:
            return self.slot_of[rid]
        duplex = info.get("duplex") if isinstance(info, dict) else None
        key = None
        if isinstance(duplex, dict):
            key = (duplex.get("session_id"), int(duplex.get("incarnation") or 0), int(duplex.get("epoch") or 0))
            if key in self.row_of_session:
                b = self.row_of_session[key]
                self.slot_of[rid] = b
                self.session_of[rid] = key
                return b
        if not self.free:
            raise RuntimeError(f"Moshi stage 0: all {self.B} rows are busy (raise max_num_seqs / max_sessions)")
        b = self.free.pop(0)
        self.s0.reset([b])
        self.slot_of[rid] = b
        if key is not None:
            self.row_of_session[key] = b
            self.session_of[rid] = key
        return b

    def _release(self, rid):
        b = self.slot_of.pop(rid, None)
        key = self.session_of.pop(rid, None)
        self.offline.pop(rid, None)
        if key is not None and any(k == key for k in self.session_of.values()):
            return  # another request id of the same session still holds the row
        if key is not None:
            self.row_of_session.pop(key, None)
        if b is not None and b not in self.free:
            self.free.append(b)

    # ------------------------------------------------------------------ engine-core API
    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
        for rid in scheduler_output.finished_req_ids:
            self.requests.pop(rid, None)
            self._release(rid)
        for nr in scheduler_output.scheduled_new_reqs:
            # duplex appends arrive again as "new" requests with the same id; prompt_len is this segment's slots
            st = self.requests.setdefault(nr.req_id, {})
            st.update(info=_info(nr), prompt_len=len(nr.prompt_token_ids or []), computed=int(nr.num_computed_tokens or 0),
                      sp=nr.sampling_params, fresh=True)
        pcm = torch.zeros(self.B, FRAME)
        active = torch.zeros(self.B, dtype=torch.bool)
        rows: dict[str, int] = {}
        emitted: dict[str, Any] = {}
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            st = self.requests.get(rid)
            if st is None:
                continue
            info = st.get("info") or {}
            duplex = info.get("duplex") if isinstance(info, dict) else None
            try:
                b = self._row_for(rid, info)
            except Exception as e:  # noqa: BLE001
                logger.warning("[MoshiS0] rid=%s: %s", rid, e)
                emitted[rid] = (EOS_TEXT, None)
                continue
            if isinstance(duplex, dict) and duplex.get("data_plane"):
                if not st.pop("fresh", False):
                    emitted[rid] = None  # nothing new for this row this step
                    continue
                st["computed"] = st.get("computed", 0) + n
                if st["computed"] < st["prompt_len"]:
                    emitted[rid] = None
                    continue
                try:
                    pcm[b] = _pcm_from_payload(duplex.get("payload"))
                except Exception as e:  # noqa: BLE001
                    logger.warning("[MoshiS0] rid=%s bad payload: %s", rid, e)
                    emitted[rid] = (EOS_TEXT, None)
                    continue
            else:  # offline request: one frame of the provided wav per step
                off = self.offline.get(rid)
                if off is None:
                    x = info.get("moshi_user_pcm")
                    x = torch.as_tensor(x if x is not None else [], dtype=torch.float32).reshape(-1)
                    mt = getattr(st.get("sp"), "max_tokens", None) or 10 ** 9
                    off = self.offline[rid] = dict(pcm=x, t=0, max=int(mt))
                    st.pop("fresh", None)
                t = off["t"]
                if (t + 1) * FRAME <= off["pcm"].numel():
                    pcm[b] = off["pcm"][t * FRAME:(t + 1) * FRAME]
                off["t"] += 1
            active[b] = True
            rows[rid] = b
        if rows:
            raw, out, ready, (te, tl) = self.s0.step(pcm, active)
            for rid, b in rows.items():
                emitted[rid] = (int(raw[b, 0]), raw[b, 1:].clone())
                off = self.offline.get(rid)
                if off is not None and (off["t"] * FRAME >= off["pcm"].numel() + 2 * FRAME or off["t"] >= off["max"]):
                    emitted[rid] = (EOS_TEXT, raw[b, 1:].clone())
            if te + tl > 0.08 and os.environ.get("MOSHI_STEP_TRACE", "0") == "1":
                logger.info("[MoshiS0] step rows=%d enc %.1f ms lm %.1f ms", len(rows), te * 1e3, tl * 1e3)
        self._last = (list(scheduler_output.num_scheduled_tokens.keys()), emitted)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        req_ids, emitted = self._last
        sampled, inter = [], []
        for rid in req_ids:
            e = emitted.get(rid)
            if e is None:
                sampled.append([])
                inter.append(None)
                continue
            tok, codes = e
            sampled.append([tok])
            inter.append(None if codes is None else {"codes.audio": codes.view(1, -1).to(torch.long)})
        out = OmniModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                    sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={},
                                    pooler_output=None, multimodal_outputs=None,
                                    inter_stage_outputs=inter if any(x is not None for x in inter) else None)
        out.omni_connector_output = self.get_omni_connector_output() if hasattr(self, "_lock") else None
        return out


class _ThinWorker(OmniWorkerMixin, WorkerBase):
    model_runner_cls = None

    def init_device(self):
        neuron_runtime_env()
        from .backports import apply as _backports
        applied = _backports()
        if applied:
            logger.info("[Moshi] applied vllm-omni backports: %s", applied)
        self.device = torch_device()
        self.model_runner = self.model_runner_cls(self.vllm_config, self.device)

    def load_model(self, *a, **k):
        self.model_runner.load_model()

    def get_kv_cache_spec(self):
        return {}

    def determine_available_memory(self) -> int:
        return 0

    def initialize_from_config(self, kv_cache_configs) -> None:
        return None

    def initialize_cache(self, num_gpu_blocks, num_cpu_blocks) -> None:
        return None

    def compile_or_warm_up_model(self):
        return CompilationTimes(language_model=self.model_runner.warmup(), encoder=0.0)

    def execute_model(self, scheduler_output, *a, **k):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output=None, *a, **k):
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        return None

    def get_supported_tasks(self):
        return ("generate",)

    def get_model(self):
        return None

    def profile(self, *a, **k):
        return None

    def synchronize_device(self):
        return None

    def check_health(self):
        return None

    def shutdown(self):
        if hasattr(self.model_runner, "shutdown_omni_connectors"):
            try:
                self.model_runner.shutdown_omni_connectors()
            except Exception:
                pass

    def handle_sleep_task(self, task):
        return None

    def handle_wake_task(self, task):
        return None


class MoshiStage0Worker(_ThinWorker):
    model_runner_cls = MoshiStage0Runner
