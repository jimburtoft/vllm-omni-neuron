"""Stage 0 (talker, LLM_AR) for vLLM-Omni on Neuron: worker + runner around our compiled fused-frame engine.

Protocol (one scheduler token == one audio frame):
  * new request (prefill, possibly chunked by the scheduler): when the full placeholder prompt has been scheduled, we
    build the prompt embeds from `additional_information` with vllm-omni's own prompt builder (VoiceDesign,
    CustomVoice, Base x-vector / ICL; streaming or non-streaming text), run the slot prefill, then frame 0.
    Streaming-text prompts leave a per-request queue of text embeddings; each decode step adds the next one (then
    tts_pad) to the frame input through the per-row `pad` tensor. ICL requests also ship their reference codes
    (`codes.ref`) with frame 0 so Code2Wav can use them as left context. Returned: sampled token = CB0(frame0), inter_stage `codes.audio=[1,16]`, `meta.codec_frame_valid`.
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
        # KV slots per request (prompt + generated frames). Must equal stage-0 max_model_len (make_deploy sets both).
        self.S = int(os.environ.get("QTTS_SLOT_LEN", str(getattr(vllm_config.model_config, "max_model_len", 2048))))
        # decode attends only the first `ctx` slots, bucketed by the furthest active row (one frame graph per bucket)
        self.ctx_buckets = sorted({min(int(x), self.S) for x in os.environ.get("QTTS_CTX_BUCKETS", "256,512,1024,2048").split(",")} | {self.S})
        self.max_new = int(os.environ.get("QTTS_MAX_NEW", str(self.S - 128)))
        self._state = None
        # Prompts longer than this (talker positions incl. instruct + codec prefix) are rejected. Greedy HF itself
        # derails on ~500-token prompts; 400 keeps ~1500 frames (~2 min) of slot room at S=2048.
        self.max_prompt = int(os.environ.get("QTTS_MAX_PROMPT", "400"))
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    # ---------------------------------------------------------------- load / warmup
    def load_model(self):
        from ..engine import TTSEngine
        from ..fused import FusedEngine, FusedFrameRows, RowNoise
        from ..tokenizer import TextTokenizer

        from ..utils import model_path
        path = model_path(self.model_config.model)
        self.model_path = path
        dt = model_dtype()
        t0 = time.time()
        self.eng = TTSEngine(path, device=self.device, max_ctx=self.S, dtype=dt, compile=False,
                             prefill_buckets=tuple(int(x) for x in os.environ.get("QTTS_PREFILL_BUCKETS", "64,128,256").split(",")))
        # Per-request sampling: the frame graph takes per-row temperature / top-k / repetition-penalty tensors and
        # host-drawn per-slot Gumbel noise, so one compiled graph serves mixed greedy + sampled batches.
        # Talker params come from each request's SamplingParams (stage-0 default_sampling_params in the deploy YAML);
        # sub-talker params from the stage's subtalker_sampling_params, overridable per request through
        # additional_information keys subtalker_temperature / subtalker_top_k / subtalker_dosample.
        self.fe = FusedEngine(self.eng, self.B, S=self.S, vec_cp=True, compile=False)
        c = compile_fn()
        self.prefill_fn = c(self.fe.prefill)
        self.frame_fns = {cb: c(FusedFrameRows(self.eng, self.fe.bt, vec_cp=True, ctx=cb).to(self.device))
                          for cb in self.ctx_buckets}
        self.noise = RowNoise(self.B, self.eng.talker.cfg.vocab, self.eng.cp.cfg.vocab, self.device,
                              base_seed=int(getattr(self.model_config, "seed", 0) or 0))
        sub = getattr(self.model_config, "subtalker_sampling_params", None) or {}
        self.sub_default = dict(do_sample=bool(sub.get("do_sample", True)), temperature=float(sub.get("temperature", 0.9)),
                                top_k=int(sub.get("top_k", 50)))
        self.row_rep = torch.ones(self.B); self.row_temp = torch.zeros(self.B); self.row_topk = torch.zeros(self.B, dtype=torch.long)
        self.row_stemp = torch.zeros(self.B); self.row_stopk = torch.zeros(self.B, dtype=torch.long)
        self.row_min = [2] * self.B; self.row_max = [self.S] * self.B
        self.tok = TextTokenizer(path)
        from .prompts import PromptBuilder
        self.prompts = PromptBuilder(self.eng, path)
        self.trailing: dict[int, torch.Tensor] = {}  # slot -> remaining streaming-text rows [N,H] (host)
        self.ref_codes: dict[str, torch.Tensor] = {}  # rid -> ICL ref codes, shipped with frame 0
        self.eos = int(self.eng.eos)
        assert self.eos == 2150, f"unexpected codec eos {self.eos}"
        self.V = self.eng.talker.cfg.vocab
        self.H = self.eng.talker.cfg.hidden
        dev, dt = self.device, self.eng.dtype
        self.h = torch.zeros(self.B, 1, self.H, dtype=dt, device=dev)
        self.lg = torch.zeros(self.B, self.V, device=dev)
        self.seen = torch.zeros(self.B, self.V, dtype=torch.bool, device=dev)
        self.pad = torch.zeros(self.B, 1, self.H, dtype=dt, device=dev)
        self.pad_host = torch.zeros(self.B, 1, self.H, dtype=dt)
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
        for cb in self.ctx_buckets:  # compile every decode-context bucket
            self._run_frame(ctx=cb)
        self._reset_slot(0)
        self.pos.zero_()
        logger.info("[NeuronTTS] talker warmup %.1fs", time.time() - t0)
        return time.time() - t0

    # ---------------------------------------------------------------- slot ops
    def _reset_slot(self, b):
        self.steps[b] = 0

    def _prefill_slot(self, b, emb, pad):
        """Prefill slot b with a prompt of any length (< S): full largest-bucket chunks, then one padded chunk."""
        from ..batched import prefill_plan
        e, dev, dt, S = self.eng, self.device, self.eng.dtype, self.S
        T = emb.shape[1]
        if T >= S - 2:
            raise ValueError(f"prompt of {T} positions does not fit the {S}-slot KV cache")
        bi = torch.tensor([b], device=dev)
        for start, n, Tb, p in prefill_plan(T, e.prefill_buckets, S):
            x = torch.zeros(1, Tb, self.H, dtype=dt); x[:, Tb - n:] = emb[:, start:start + n].to(dt)
            hl, lg = self.prefill_fn(x.to(dev), bi, p.to(dev))
        self.h = self.h.index_copy(0, bi, hl)
        self.lg = self.lg.index_copy(0, bi, lg)
        self.seen = self.seen.index_copy(0, bi, torch.zeros(1, self.V, dtype=torch.bool, device=dev))
        self.pad_host[b] = pad.to(dt).reshape(1, -1)
        self.pad = self.pad.index_copy(0, bi, pad.to(dt).reshape(1, 1, -1).to(dev))
        self.pos[b] = T
        self.steps[b] = 0

    def _set_row_params(self, b, sp, info):
        """Bind a request's sampling parameters to slot b."""
        from .common import info_first
        temp = float(getattr(sp, "temperature", 0.0) or 0.0) if sp is not None else 0.0
        topk = int(getattr(sp, "top_k", 0) or 0) if sp is not None else 0
        if topk < 0:
            topk = 0
        rep = float(getattr(sp, "repetition_penalty", 1.05) or 1.0) if sp is not None else 1.05
        self.row_temp[b], self.row_topk[b], self.row_rep[b] = temp, min(topk, 64), rep
        ex = (getattr(sp, "extra_args", None) or {}) if sp is not None else {}
        pick = lambda k: ex.get(k) if ex.get(k) is not None else info_first(info, k, None)
        sd, st, sk = pick("subtalker_dosample"), pick("subtalker_temperature"), pick("subtalker_top_k")
        do = self.sub_default["do_sample"] if sd is None else bool(sd)
        self.row_stemp[b] = (float(st) if st is not None else self.sub_default["temperature"]) if do else 0.0
        self.row_stopk[b] = min(int(sk) if sk is not None else self.sub_default["top_k"], 64) if do else 0
        self.row_min[b] = int(getattr(sp, "min_tokens", 2) or 0) if sp is not None else 2
        mt = getattr(sp, "max_tokens", None) if sp is not None else None
        mn = info_first(info, "max_new_tokens", None)
        lim = [x for x in (mt, mn) if x]
        self.row_max[b] = min(lim) if lim else self.S
        seed = getattr(sp, "seed", None) if sp is not None else None
        if seed is None:
            seed = ex.get("tts_local_seed")
        if seed is not None:
            self.noise.reseed(b, seed)
        else:  # fresh stream per request (non-reproducible), independent of the slot's previous request
            self.noise.reseed(b, int.from_bytes(os.urandom(4), "little"))

    def _advance_text(self):
        """Streaming text: frame k of a row adds trailing[k] (then tts_pad) to its next talker input."""
        if not self.trailing:
            return
        for b in list(self.trailing):
            tail, k = self.trailing[b], self.steps[b]
            if k < tail.shape[0]:
                self.pad_host[b, 0] = tail[k]
            else:
                self.pad_host[b, 0] = self.prompts.tts_pad.reshape(-1).to(self.pad_host.dtype)
                del self.trailing[b]
        self.pad = self.pad_host.to(self.device)

    def _run_frame(self, ctx=None):
        self._advance_text()
        allow = torch.tensor([s >= max(self.row_min[b], 1) for b, s in enumerate(self.steps)])
        g0, gs = self.noise.draw()
        d = self.device
        need = int(self.pos.max()) + 1  # every active row writes and attends slot pos
        ctx = ctx or next(cb for cb in self.ctx_buckets if cb >= need)
        self.h, self.lg, self.seen, codes = self.frame_fns[ctx](self.h, self.lg, self.seen, allow.to(d), self.pad,
                                                          self.pos.to(d), g0, gs, self.row_rep.to(d),
                                                          self.row_temp.to(d), self.row_topk.to(d),
                                                          self.row_stemp.to(d), self.row_stopk.to(d))
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
        """-> (prompt [1,T,H], trailing [N,H], tts_pad [1,1,H], ref_code [R,16] | None)."""
        return self.prompts.build(info)

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
                self.trailing.pop(b, None)
                self.free.append(b)
                self.pos[b] = 0
            self.requests.pop(rid, None)
            self.ref_codes.pop(rid, None)
        for nr in scheduler_output.scheduled_new_reqs:
            self.requests[nr.req_id] = dict(info=_deser_info(nr), prompt_len=len(nr.prompt_token_ids or []),
                                            computed=nr.num_computed_tokens, sp=nr.sampling_params,
                                            external_req_id=getattr(nr, "external_req_id", None))
        emitted: dict[str, tuple[int, torch.Tensor | None]] = {}
        decode_rids = []
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            st = self.requests.get(rid)
            if st is None:
                continue
            if rid not in self.slot_of:
                st["computed"] = st.get("computed", 0) + n
                if st["computed"] >= st["prompt_len"]:  # full prompt scheduled -> prefill + frame 0
                    try:
                        emb, trailing, pad, ref_code = self._build_prompt(st["info"])
                    except Exception as e:  # bad request (unknown speaker, wrong task for checkpoint, ...)
                        logger.warning("[NeuronTTS] rid=%s prompt build failed: %s", rid, e)
                        emitted[rid] = (self.eos, None)
                        continue
                    if emb.shape[1] > self.max_prompt:
                        # Too long for one utterance: the checkpoint degrades badly here (HF reference included). Stop
                        # immediately with EOS so the request finishes cleanly; clients should split long text.
                        logger.warning("[NeuronTTS] rid=%s prompt %d positions > QTTS_MAX_PROMPT=%d; rejecting", rid,
                                       emb.shape[1], self.max_prompt)
                        emitted[rid] = (self.eos, None)
                        continue
                    b = self.free.pop(0)
                    self.slot_of[rid] = b
                    self._set_row_params(b, st.get("sp"), st["info"])
                    self._prefill_slot(b, emb, pad)
                    if trailing.shape[0] > 1 or (trailing.shape[0] == 1 and not torch.equal(trailing[0], pad.reshape(-1))):
                        self.trailing[b] = trailing
                    else:
                        self.trailing.pop(b, None)
                    if isinstance(ref_code, torch.Tensor) and ref_code.numel():
                        self.ref_codes[rid] = ref_code.to(torch.long).cpu().contiguous()
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
                if int(self.pos[b]) >= self.S - 1 or self.steps[b] > self.row_max[b]:  # safety; scheduler stops at max_model_len
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
            if cb0 == self.eos or frame is None:
                sampled.append([self.eos])
                inter.append({"codes.audio": torch.zeros(1, 16, dtype=torch.long),
                              "meta.codec_frame_valid": torch.tensor([False])})
            else:
                sampled.append([cb0])
                d = {"codes.audio": frame.view(1, 16).to(torch.long), "meta.codec_frame_valid": torch.tensor([True])}
                ref = self.ref_codes.pop(rid, None)
                if ref is not None:
                    d["codes.ref"] = ref
                inter.append(d)
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
        from .backports import apply as _backports
        applied = _backports()  # before the scheduler resolves the stage-input processor (same process, created later)
        if applied:
            logger.info("[NeuronTTS] applied vllm-omni backports: %s", applied)
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
