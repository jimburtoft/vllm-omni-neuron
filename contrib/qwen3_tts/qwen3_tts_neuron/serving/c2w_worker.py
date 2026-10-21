"""Stage 1 (code2wav, LLM_GENERATION) for vLLM-Omni on Neuron.

Receives flat codebook-major codec chunks (`prompt_token_ids`, [Q*F]) plus `meta` (left_context_size, finished,
request_id) via the stock chunk adapter, decodes them with our windowed Code2Wav graphs, and returns 24 kHz fp32 audio
per request as `multimodal_outputs[{"model_outputs": wav, "sr": 24000}]`.

The installed 0.29.0rc1 processor ships only NEW frames after the first chunk (left_context_size=0) and expects the
decoder to keep per-request state. Our decoder is stateless-windowed, so we keep the last `ctx` frames per request
and prepend them (exact, see docs/008 C2).
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch
from vllm.logger import init_logger
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin, needs_omni_connector

from .common import cpu_mode, model_dtype, neuron_runtime_env, torch_device

logger = init_logger(__name__)
Q = 16


def _meta(info: dict, key, default=None):
    m = info.get("meta") if isinstance(info, dict) else None
    v = None
    if isinstance(m, dict):
        v = m.get(key)
    if v is None:
        v = info.get(f"meta.{key}", info.get(key)) if isinstance(info, dict) else None
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    if isinstance(v, torch.Tensor):
        v = v.reshape(-1)[0].item() if v.numel() else None
    return default if v is None else v


class NeuronCode2WavModelRunner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.requests: dict[str, Any] = {}
        self.model_intermediate_buffer: dict[str, Any] = {}
        self.hist: dict[str, torch.Tensor] = {}   # req -> last ctx frames [<=ctx, Q]
        self.last_chunk: dict[str, torch.Tensor] = {}
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    def load_model(self):
        from ..code2wav import Code2Wav, Code2WavRunner
        from ..utils import model_path
        path = model_path(self.model_config.model)
        st = os.path.join(path, "speech_tokenizer")
        t0 = time.time()
        comp = not (cpu_mode() or os.environ.get("QTTS_EAGER", "0") == "1")
        if comp:
            import torch as _t
            from libtorch_neuronx_lite.compile.native_backend import register
            backend = register()
            _orig = _t.compile
            _t.compile = lambda m, backend_=None, dynamic=None, **k: _orig(m, backend=backend, dynamic=False)
            try:
                self.c2w = Code2WavRunner(st, device=self.device, dtype=model_dtype(), compile=True)
            finally:
                _t.compile = _orig
        else:
            self.c2w = Code2WavRunner(st, device=self.device, dtype=model_dtype(), compile=False)
        self.ctx, self.up = self.c2w.ctx, self.c2w.up
        logger.info("[NeuronC2W] code2wav loaded in %.1fs device=%s", time.time() - t0, self.device)

    def warmup(self):
        t0 = time.time()
        w = torch.randint(0, 2048, (self.c2w.ctx + self.c2w.chunk, Q))
        self.c2w.decode_window(w[:4])
        self.c2w.decode_window(w, ctx_frames=self.ctx)
        self.c2w.decode_window(w)
        return time.time() - t0

    def _decode(self, rid, flat: list[int], lctx: int):
        n = len(flat) // Q
        if n == 0:
            return torch.zeros(0)
        codes = torch.tensor(flat, dtype=torch.long).view(Q, n).t().contiguous()  # [F,Q]
        last = self.last_chunk.get(rid)
        if lctx == 0 and last is not None and last.shape == codes.shape and torch.equal(last, codes):
            # talker2code2wav_async_chunk re-ships the final full chunk when a request ends exactly on a chunk
            # boundary (finished flush with length % chunk == 0). Decoding it again would repeat ~2 s of audio.
            logger.debug("[NeuronC2W] rid=%s duplicate final chunk (%d frames) skipped", rid, n)
            return torch.zeros(0)
        self.last_chunk[rid] = codes
        if lctx > 0:  # processor-provided context (e.g. ICL ref codes): drop it from output
            codes_new = codes[lctx:]
            prev = codes[:lctx][-self.ctx:]
        else:
            codes_new = codes
            prev = self.hist.get(rid)
        outs = []
        s = 0
        while s < codes_new.shape[0]:
            e = min(s + self.c2w.chunk, codes_new.shape[0])
            k = 0 if prev is None else prev.shape[0]
            win = codes_new[s:e] if k == 0 else torch.cat([prev, codes_new[s:e]])
            w = self.c2w.decode_window(win, ctx_frames=k)
            w = w if (self.c2w.crop and k == self.ctx) else w[:, k * self.up:]
            outs.append(w.cpu().float().reshape(-1))
            prev = win[-self.ctx:]
            s = e
        self.hist[rid] = prev
        return torch.cat(outs)

    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
        for rid in scheduler_output.finished_req_ids:
            self.requests.pop(rid, None)
            self.hist.pop(rid, None)
            self.last_chunk.pop(rid, None)
        from vllm_omni.engine.serialization import deserialize_additional_information
        ncomp = {}
        for nr in scheduler_output.scheduled_new_reqs:
            info = getattr(nr, "additional_information", None)
            info = deserialize_additional_information(info) if info is not None else {}
            self.requests[nr.req_id] = dict(tokens=list(nr.prompt_token_ids or []), info=info or {})
            ncomp[nr.req_id] = int(nr.num_computed_tokens or 0)
        cr = scheduler_output.scheduled_cached_reqs
        ptoks = getattr(cr, "prompt_token_ids", None) or {}
        ainfo = getattr(cr, "additional_information", None) or {}
        for i, rid in enumerate(getattr(cr, "req_ids", [])):
            st = self.requests.setdefault(rid, dict(tokens=[], info={}))
            try:
                ncomp[rid] = int(cr.num_computed_tokens[i])
            except Exception:
                pass
            if rid in ptoks and ptoks[rid] is not None:
                st["tokens"] = list(ptoks[rid])
            if rid in ainfo and ainfo[rid]:
                try:
                    st["info"] = deserialize_additional_information(ainfo[rid]) or st["info"]
                except Exception:
                    st["info"] = ainfo[rid] if isinstance(ainfo[rid], dict) else st["info"]
        req_ids, wavs = [], []
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            st = self.requests.get(rid)
            if st is None:
                continue
            req_ids.append(rid)
            lctx = int(_meta(st["info"], "left_context_size", 0) or 0)
            nc = ncomp.get(rid, 0)
            toks = st["tokens"][nc: nc + n]
            wavs.append(self._decode(rid, toks, lctx))
        self._last = (req_ids, wavs)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        from vllm_omni.utils.mm_outputs import partition_payload_list
        req_ids, wavs = self._last
        payloads = [{"model_outputs": w, "sr": torch.tensor(24000)} for w in wavs]
        if self._async_chunk:
            inter, mm = partition_payload_list(payloads) if payloads else (None, None)
        else:
            inter, mm = payloads, payloads
        out = OmniModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                    sampled_token_ids=[], logprobs=None, prompt_logprobs_dict={}, pooler_output=None,
                                    multimodal_outputs=mm, inter_stage_outputs=inter, num_nans_in_logits={})
        out.omni_connector_output = self.get_omni_connector_output() if hasattr(self, "_lock") else None
        return out


class NeuronCode2WavWorker(OmniWorkerMixin, WorkerBase):
    model_runner_cls = NeuronCode2WavModelRunner

    def init_device(self):
        neuron_runtime_env()
        if not cpu_mode():
            import libtorch_neuronx_lite  # noqa: F401
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
        return self.model_runner.c2w.m

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
