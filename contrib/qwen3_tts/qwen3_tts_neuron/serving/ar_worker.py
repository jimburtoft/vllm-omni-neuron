"""Stage 0 (talker, LLM_AR) for vLLM-Omni on Neuron: worker + runner around our compiled fused-frame engine.

Protocol (one scheduler token == one audio frame):
  * new request (prefill, possibly chunked by the scheduler): when the full placeholder prompt has been scheduled, we
    build the VoiceDesign prompt embeds from `additional_information` (text/instruct/language), run the slot prefill,
    then frame 0. Returned: sampled token = CB0(frame0), inter_stage `codes.audio=[1,16]`, `meta.codec_frame_valid`.
  * decode step: one FusedFrame over all B slots. Each scheduled request returns its CB0 and frame.
  * EOS: when the fused frame samples CB0 == 2150 we report token 2150 (pipeline stop id) and no frame.
The KV cache is owned by the model (static per-slot buffers); `get_kv_cache_spec()` returns {} (attention-free stage).
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import CompilationTimes, WorkerBase

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.mixins import OmniWorkerMixin
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin, needs_omni_connector

from .common import compile_fn, cpu_mode, info_first, model_dtype, neuron_runtime_env, torch_device

logger = init_logger(__name__)


def _deser_info(new_req) -> dict:
    info = getattr(new_req, "additional_information", None)
    if info is None:
        return dict(getattr(new_req, "model_intermediate_buffer", None) or {})
    from vllm_omni.engine.serialization import deserialize_additional_information
    return deserialize_additional_information(info) or {}


class NeuronTTSARModelRunner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.requests: dict[str, Any] = {}
        self.model_intermediate_buffer: dict[str, Any] = {}
        self.B = int(vllm_config.scheduler_config.max_num_seqs)
        self.S = int(os.environ.get("QTTS_SLOT_LEN", "512"))
        self.max_new = int(os.environ.get("QTTS_MAX_NEW", str(self.S - 128)))
        self._state = None
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    # ---------------------------------------------------------------- load / warmup
    def load_model(self):
        from transformers import AutoTokenizer

        from ..engine import TTSEngine
        from ..fused import FusedEngine

        from ..utils import model_path
        path = model_path(self.model_config.model)
        self.model_path = path
        dt = model_dtype()
        t0 = time.time()
        self.eng = TTSEngine(path, device=self.device, max_ctx=self.S, dtype=dt, compile=False,
                             prefill_buckets=tuple(int(x) for x in os.environ.get("QTTS_PREFILL_BUCKETS", "64,128,256").split(",")))
        # Sampling: stage-0 default_sampling_params (talker) + subtalker_sampling_params, read from the deploy YAML via env
        # (set by make_deploy.py). Static per server: temperature/top-k are compiled into the frame graph.
        t, k = float(os.environ.get("QTTS_TEMPERATURE", "0")), int(os.environ.get("QTTS_TOP_K", "0"))
        st, sk = float(os.environ.get("QTTS_SUB_TEMPERATURE", "0")), int(os.environ.get("QTTS_SUB_TOP_K", "0"))
        sampling = dict(temperature=t, top_k=k, sub_temperature=st, sub_top_k=sk) if (t > 0 or st > 0) else None
        self.fe = FusedEngine(self.eng, self.B, S=self.S, vec_cp=True, compile=False, sampling=sampling,
                              seed=int(os.environ.get("QTTS_SEED", "0")))
        c = compile_fn()
        self.prefill_fn = c(self.fe.prefill)
        self.frame_fn = c(self.fe.frame)
        self.tok = AutoTokenizer.from_pretrained(path)
        self.eos = int(self.eng.eos)
        assert self.eos == 2150, f"unexpected codec eos {self.eos}"
        self.V = self.eng.talker.cfg.vocab
        self.H = self.eng.talker.cfg.hidden
        dev, dt = self.device, self.eng.dtype
        self.h = torch.zeros(self.B, 1, self.H, dtype=dt, device=dev)
        self.lg = torch.zeros(self.B, self.V, device=dev)
        self.seen = torch.zeros(self.B, self.V, dtype=torch.bool, device=dev)
        self.pad = torch.zeros(self.B, 1, self.H, dtype=dt, device=dev)
        self.pos = torch.zeros(self.B, dtype=torch.long)
        self.steps = [0] * self.B
        self.slot_of: dict[str, int] = {}
        self.free = list(range(self.B))
        self.pending_frames: dict[str, torch.Tensor] = {}
        logger.info("[NeuronTTS] talker loaded in %.1fs (B=%d S=%d dtype=%s device=%s)", time.time() - t0, self.B, self.S, dt, dev)

    def warmup(self):
        t0 = time.time()
        pad = torch.zeros(1, 1, self.H, dtype=self.eng.dtype)
        for Tb in self.eng.prefill_buckets:
            self._prefill_slot(0, torch.zeros(1, Tb - 1, self.H, dtype=self.eng.dtype), pad)
        self._run_frame()
        self._reset_slot(0)
        logger.info("[NeuronTTS] talker warmup %.1fs", time.time() - t0)
        return time.time() - t0

    # ---------------------------------------------------------------- slot ops
    def _reset_slot(self, b):
        self.steps[b] = 0

    def _prefill_slot(self, b, emb, pad):
        e, dev, dt, S = self.eng, self.device, self.eng.dtype, self.S
        T = emb.shape[1]
        Tb = next(x for x in e.prefill_buckets if x >= T)
        np_ = Tb - T
        x = torch.zeros(1, Tb, self.H, dtype=dt); x[:, np_:] = emb.to(dt)
        p = torch.cat([torch.arange(S - np_, S), torch.arange(T)]).long()
        bi = torch.tensor([b], device=dev)
        hl, lg = self.prefill_fn(x.to(dev), bi, p.to(dev))
        self.h = self.h.index_copy(0, bi, hl)
        self.lg = self.lg.index_copy(0, bi, lg)
        self.seen = self.seen.index_copy(0, bi, torch.zeros(1, self.V, dtype=torch.bool, device=dev))
        self.pad = self.pad.index_copy(0, bi, pad.to(dt).reshape(1, 1, -1).to(dev))
        self.pos[b] = T
        self.steps[b] = 0

    def _run_frame(self):
        allow = torch.tensor([s >= 2 for s in self.steps])
        g0, gs = self.fe.noise()
        self.h, self.lg, self.seen, codes = self.frame_fn(self.h, self.lg, self.seen, allow.to(self.device), self.pad,
                                                          self.pos.to(self.device), g0, gs)
        occupied = set(self.slot_of.values())
        for b in range(self.B):
            if b in occupied:
                self.pos[b] += 1
                self.steps[b] += 1
            else:
                self.pos[b] = 0  # park free slots: writes go to their own (dead) KV slot 0, never out of bounds
        if (self.pos >= self.S).any():
            raise RuntimeError(f"slot position overflow (max {int(self.pos.max())} >= S={self.S}); raise QTTS_SLOT_LEN")
        return codes.cpu()

    def _build_prompt(self, info):
        text = info_first(info, "text", "")
        instruct = info_first(info, "instruct", "") or ""
        language = info_first(info, "language", "Auto") or "Auto"
        ids = self.tok(f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").input_ids
        iids = self.tok(f"<|im_start|>user\n{instruct}<|im_end|>\n", return_tensors="pt").input_ids if instruct.strip() else None
        with torch.no_grad():
            emb, pad = self.eng.build_prompt(ids, iids, language)
        return emb, pad

    # ---------------------------------------------------------------- engine-core API
    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        _t_enter = time.perf_counter()
        n_prefill = 0
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
        for rid in scheduler_output.finished_req_ids:
            b = self.slot_of.pop(rid, None)
            if b is not None:
                self.free.append(b)
                self.pos[b] = 0
            self.requests.pop(rid, None)
        for nr in scheduler_output.scheduled_new_reqs:
            self.requests[nr.req_id] = dict(info=_deser_info(nr), prompt_len=len(nr.prompt_token_ids or []),
                                            computed=nr.num_computed_tokens, external_req_id=getattr(nr, "external_req_id", None))
        emitted: dict[str, tuple[int, torch.Tensor | None]] = {}
        decode_rids = []
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            st = self.requests.get(rid)
            if st is None:
                continue
            if rid not in self.slot_of:
                st["computed"] = st.get("computed", 0) + n
                if st["computed"] >= st["prompt_len"]:  # full prompt scheduled -> prefill + frame 0
                    b = self.free.pop(0)
                    self.slot_of[rid] = b
                    emb, pad = self._build_prompt(st["info"])
                    self._prefill_slot(b, emb, pad)
                    st["new_slot"] = True
                    n_prefill += 1
                else:
                    emitted[rid] = None  # partial prefill: no token
            else:
                decode_rids.append(rid)
        active = [rid for rid in scheduler_output.num_scheduled_tokens if rid in self.slot_of]
        _t_pre = time.perf_counter()
        if active:
            codes = self._run_frame()
            for rid in active:
                b = self.slot_of[rid]
                c = codes[b]
                cb0 = int(c[0])
                if int(self.pos[b]) >= self.S - 2:  # KV slot full: force-stop this request
                    cb0 = self.eos
                emitted[rid] = (cb0, c)
                self.requests[rid].pop("new_slot", None)
        _t_frame = time.perf_counter()
        self._last = (list(scheduler_output.num_scheduled_tokens.keys()), emitted)
        if os.environ.get("QTTS_STEP_STATS") == "1":
            now = time.perf_counter()
            self._ss = getattr(self, "_ss", dict(n=0, t_prev=now, gap=0.0, active=0, frame=0.0, pre=0.0, prefills=0))
            self._ss["n"] += 1; self._ss["gap"] += now - self._ss["t_prev"]; self._ss["t_prev"] = now
            self._ss["active"] += len(active); self._ss["frame"] += _t_frame - _t_pre; self._ss["pre"] += _t_pre - _t_enter
            self._ss["prefills"] += n_prefill
            if self._ss["n"] % 200 == 0:
                logger.info("[NeuronTTS] steps=%d step interval %.2f ms | in-execute: prefill+setup %.2f ms, frame %.2f ms "
                            "| outside execute %.2f ms | active rows %.2f, prefills %d", self._ss["n"], 1000 * self._ss["gap"] / 200,
                            1000 * self._ss["pre"] / 200, 1000 * self._ss["frame"] / 200,
                            1000 * (self._ss["gap"] - self._ss["pre"] - self._ss["frame"]) / 200, self._ss["active"] / 200,
                            self._ss["prefills"])
                self._ss.update(gap=0.0, active=0, frame=0.0, pre=0.0, prefills=0)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        req_ids, emitted = self._last
        out_ids, sampled, inter = [], [], []
        for rid in req_ids:
            out_ids.append(rid)
            e = emitted.get(rid)
            if e is None:
                sampled.append([])
                inter.append(None)
                continue
            cb0, frame = e
            if cb0 == self.eos:
                sampled.append([self.eos])
                inter.append({"codes.audio": torch.zeros(1, 16, dtype=torch.long),
                              "meta.codec_frame_valid": torch.tensor([False])})
            else:
                sampled.append([cb0])
                inter.append({"codes.audio": frame.view(1, 16).to(torch.long),
                              "meta.codec_frame_valid": torch.tensor([True])})
        out = OmniModelRunnerOutput(req_ids=out_ids, req_id_to_index={r: i for i, r in enumerate(out_ids)},
                                    sampled_token_ids=sampled, logprobs=None, prompt_logprobs_dict={},
                                    pooler_output=None, multimodal_outputs=None,
                                    inter_stage_outputs=inter if any(x is not None for x in inter) else None)
        out.omni_connector_output = self.get_omni_connector_output() if hasattr(self, "_lock") else None
        return out


class NeuronTTSARWorker(OmniWorkerMixin, WorkerBase):
    model_runner_cls = NeuronTTSARModelRunner

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
        t = self.model_runner.warmup()
        return CompilationTimes(language_model=t, encoder=0.0)

    def execute_model(self, scheduler_output, *a, **k):
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output=None, *a, **k):
        return self.model_runner.sample_tokens(grammar_output)

    def take_draft_token_ids(self):
        return None

    def get_supported_tasks(self):
        return ("generate",)

    def get_model(self):
        return self.model_runner.eng.talker

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
