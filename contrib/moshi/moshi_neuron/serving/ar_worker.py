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
        # the connector's chunk adapter resolves the stage input processor by import; apply the processor / data
        # plane backports in this process first (TP workers are separate processes from the stage engine)
        from .backports import apply as _bp
        logger.info("[MoshiS0] runner backports: %s (pid %d)", _bp(), os.getpid())
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
        # every (batch bucket x context bucket) graph: a session crossing into a context bucket that was not warmed
        # would otherwise compile it on demand inside the serving step (minutes; all sessions stall)
        ctxs = list(getattr(self.s0, "ctx_buckets", [None]))
        for bb in self.s0.buckets:
            act = torch.zeros(self.B, dtype=torch.bool)
            act[:bb] = True
            for c in ctxs:
                self.s0.force_ctx = c if len(ctxs) > 1 else None
                for _ in range(2):
                    self.s0.step(z, act)
                if getattr(self.s0, "emb_inputs", False):
                    # the prompting-row variant of the frame call (embedding override active on a row) -- compile
                    # it here, not inside the first PersonaPlex session
                    self.s0.__dict__.setdefault("_pq", {})[0] = dict(
                        emb=torch.zeros(2, self.s0.cfg.dim, dtype=self.s0._ze.dtype), ring=None, P=10 ** 9,
                        offset=0, i=0)
                    for _ in range(2):
                        self.s0.step(z, act)
                    self.s0._pq.clear()
                self.s0.reset(range(self.B))
            self.s0.force_ctx = None
            logger.info("[MoshiS0] warmed bucket %d x ctx %s (%.1fs)", bb, ctxs, time.time() - t0)
        if getattr(self.s0, "emb_inputs", False) and os.environ.get("MOSHI_PP_PREFILL", "0") == "1":
            from ..lm import MoshiPrefill
            pmax = int(os.environ.get("MOSHI_PP_PMAX", "192"))
            self.s0._prefill = self.s0._cf(MoshiPrefill(self.s0.lm, pmax))
            dv = self.s0.device
            self.s0._prefill(torch.zeros(pmax, self.s0.cfg.dim, dtype=self.s0.lm.dtype, device=dv),
                             torch.tensor(1, device=dv), torch.tensor(0, device=dv)).cpu()
            self.s0.reset(range(self.B))
            logger.info("[MoshiS0] warmed prefill graph (P_MAX %d) (%.1fs)", pmax, time.time() - t0)
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
        if b is not None:
            self.__dict__.setdefault("_prompted", set()).discard(b)
        if b is not None and b not in self.free:
            self.free.append(b)
            self.free.sort()  # always hand out the lowest free row -> smallest batch bucket

    # ------------------------------------------------------------------ engine-core API
    def _trace(self, ev, rows):
        """MOSHI_TRACE=path: append 'event wallclock rows' lines (one file per process) for latency attribution."""
        p = os.environ.get("MOSHI_TRACE")
        if not p:
            return
        f = self.__dict__.get("_tf")
        if f is None:
            f = self._tf = open(f"{p}.s0.{os.getpid()}", "a", buffering=1)
        f.write(f"{ev} {time.time():.6f} {rows}\n")

    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        _tx0 = time.perf_counter()
        self._trace("s0_enter", len(scheduler_output.num_scheduled_tokens))
        if os.environ.get("MOSHI_STEP_STATS", "0") == "1":
            st_ = self.__dict__.setdefault("_gap", dict(last=None, gap=0.0, n=0, pre=0.0, post=0.0))
            if st_["last"] is not None:
                st_["gap"] += _tx0 - st_["last"]
            st_["n"] += 1
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
                rc = duplex.get("runtime_config") if isinstance(duplex.get("runtime_config"), dict) else {}
                voice = rc.get("personaplex_voice_prompt")
                if (voice and int(duplex.get("seq") or 0) <= 1 and b not in self.__dict__.setdefault("_prompted", set())
                        and os.environ.get("MOSHI_PP_PROMPT", "1") == "1"):
                    # PersonaPlex system prompt (voice + persona) for this session's row, before its first user frame
                    if os.environ.get("MOSHI_PP_PREFILL", "0") == "1":
                        t_p = time.perf_counter()
                        np_ = self.s0.prefill_batched(b, str(voice), str(rc.get("personaplex_persona") or ""))
                        logger.info("[MoshiS0] row %d: PersonaPlex prompt prefilled (%s, %d frames) in %.0f ms", b,
                                    voice, np_, 1e3 * (time.perf_counter() - t_p))
                    elif os.environ.get("MOSHI_PP_INLINE", "0") == "1":
                        t_p = time.perf_counter()
                        np_ = self.s0.prefill_prompt(b, str(voice), str(rc.get("personaplex_persona") or ""))
                        logger.info("[MoshiS0] row %d: PersonaPlex prompt inline (%s, %d steps) in %.0f ms", b,
                                    voice, np_, 1e3 * (time.perf_counter() - t_p))
                    else:
                        # queued: the row consumes one prompt frame per serving step inside the batch (user frames
                        # arriving meanwhile are dropped -- the reference runs the prompt before any user audio)
                        np_ = self.s0.start_prompt(b, str(voice), str(rc.get("personaplex_persona") or ""))
                        logger.info("[MoshiS0] row %d: PersonaPlex prompt queued (%s, %d steps)", b, voice, np_)
                    self._prompted.add(b)
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
        _tx1 = time.perf_counter()
        if os.environ.get("MOSHI_STEP_STATS", "0") == "1":
            self._gap["pre"] += _tx1 - _tx0
        if rows:
            raw, out, ready, (te, tl) = self.s0.step(pcm, active)
            dec_pcm = getattr(self.s0, "last_pcm", None) if self.s0.decode_here else None
            for rid, b in rows.items():
                if dec_pcm is not None:
                    # stage 0 decoded the frame: emit (text of the de-delayed frame, its PCM) when ready
                    emitted[rid] = (int(raw[b, 0]), ("pcm", dec_pcm[b].clone() if bool(ready[b]) else None,
                                                     int(out[b, 0]) if bool(ready[b]) else -1))
                    continue
                emitted[rid] = (int(raw[b, 0]), raw[b, 1:].clone())
                off = self.offline.get(rid)
                if off is not None and (off["t"] * FRAME >= off["pcm"].numel() + 2 * FRAME or off["t"] >= off["max"]):
                    emitted[rid] = (EOS_TEXT, raw[b, 1:].clone())
            if os.environ.get("MOSHI_STEP_STATS", "0") == "1":
                now = time.perf_counter()
                st = self.__dict__.setdefault("_st", dict(n=0, enc=0.0, lm=0.0, dec=0.0, rows=0, t0=now))
                st["n"] += 1; st["enc"] += te; st["lm"] += tl; st["rows"] += len(rows)
                st["dec"] += getattr(self.s0, "t_dec", 0.0)
                if st["n"] % 200 == 0:
                    logger.info("[MoshiS0] 200 steps: enc %.1f ms, lm %.1f ms, dec %.1f ms, %.2f rows/step, "
                                "%.1f ms/step wall, bucket %s", 1e3 * st["enc"] / 200, 1e3 * st["lm"] / 200,
                                1e3 * st["dec"] / 200, st["rows"] / 200, 1e3 * (now - st["t0"]) / 200,
                                getattr(self.s0, "last_bucket", "-"))
                    st.update(enc=0.0, lm=0.0, dec=0.0, rows=0, t0=now)
        self._last = (list(scheduler_output.num_scheduled_tokens.keys()), emitted)
        self._trace("s0_exit", len(rows))
        if os.environ.get("MOSHI_TRACE_ROWS") == "1" and os.environ.get("MOSHI_TRACE"):
            # per-session progress: request id -> frames computed so far (one line per step, rank 0 only)
            if getattr(self, "rank", 0) == 0 or not hasattr(self, "rank"):
                f = self.__dict__.get("_tfr")
                if f is None:
                    f = self._tfr = open(f"{os.environ['MOSHI_TRACE']}.rows.{os.getpid()}", "a", buffering=1)
                now = time.time()
                for rid in rows:
                    st_ = self.requests.get(rid)
                    f.write(f"{now:.6f} {rid.split('.i.')[0][-16:]} {st_.get('computed', -1) if isinstance(st_, dict) else -1}\n")
        if os.environ.get("MOSHI_STEP_STATS", "0") == "1":
            g = self._gap
            g["last"] = time.perf_counter()
            if g["n"] % 200 == 0 and g["n"]:
                logger.info("[MoshiS0 overhead/200] outside execute_model %.1f ms, execute pre-step %.1f ms",
                            1e3 * g["gap"] / 200, 1e3 * g["pre"] / 200)
                g.update(gap=0.0, pre=0.0)
        if os.environ.get("MOSHI_ONE_RPC", "1") == "1":
            # return the output from execute_model: EngineCore.step skips the sample_tokens RPC (one broadcast +
            # gather round trip to every TP worker less per step)
            return self.sample_tokens(None)
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
            if isinstance(codes, tuple) and codes and codes[0] == "pcm":
                _, wav, txt = codes
                inter.append({"codes.ref": (wav if wav is not None else torch.zeros(0)).float(),
                              "codes.audio": torch.tensor([[txt]], dtype=torch.long)})
                continue
            # row layout [8 audio codes, text token]: the backported processor de-delays the 8 audio codes and carries
            # the text token (delay 0) to stage 1, which detokenizes it for the client text stream
            inter.append(None if codes is None else
                         {"codes.audio": torch.cat([codes.view(-1), torch.tensor([tok])]).view(1, -1).to(torch.long)})
        if os.environ.get("MOSHI_DEBUG_INTER", "0") == "1" and inter:
            x = next((i for i in inter if i), None)
            logger.warning("[MoshiS0 dbg] inter keys=%s", {k: (tuple(v.shape) if hasattr(v, "shape") else v)
                                                          for k, v in (x or {}).items()})
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
        # stage-1 pass-through (MOSHI_S0_DECODE=1) never touches the device: stay on CPU so it holds no NeuronCore
        if os.environ.get("MOSHI_S0_DECODE", "0") == "1" and self.model_runner_cls.__name__ == "MoshiCode2WavRunner":
            self.device = torch.device("cpu")
        else:
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


def _make_tp_worker():
    """Stage-0 worker for tensor_parallel_size > 1: vllm-neuron's NeuronWorker does device pinning (one core per
    rank from `devices:`), the gloo world, the CCOM rendezvous and runtime init; we then replace its model runner with
    MoshiStage0Runner (every rank runs the same frame graph with its shard and the 2 all-reduces per layer; every
    rank runs the replicated Mimi encoder / depformer / sampling so all ranks hold identical tokens). Only TP rank 0's
    output is returned to the scheduler (vLLM's executor uses rank 0's output)."""
    from vllm_neuron.vllm.worker.neuron_worker import NeuronWorker

    class MoshiStage0TPWorker(OmniWorkerMixin, NeuronWorker):
        model_runner_cls = MoshiStage0Runner

        def init_device(self):
            from .backports import apply as _backports
            _backports()
            tp = int(self.vllm_config.parallel_config.tensor_parallel_size)
            os.environ["MOSHI_TP"] = str(tp)
            os.environ["OMP_NUM_THREADS"] = "1"
            torch.set_num_threads(int(os.environ.get("MOSHI_HOST_THREADS", "2")))
            from vllm_neuron.vllm.patches.replica_mesh_patch import apply_replica_mesh_patches
            apply_replica_mesh_patches()
            # NeuronWorker's distributed init only (core pinning, gloo world, CCOM rendezvous, runtime init); its
            # NeuronModelRunner needs a HF text config, so we build our own runner instead.
            self._init_neuron_distributed_environment_and_runtime(
                self.vllm_config, self.rank, self.distributed_init_method, self.local_rank)
            import libtorch_neuronx_lite  # noqa: F401
            # host CPUs are shared by all TP ranks (8 at LNC=1 on 12 vCPUs): pin each rank to its own CPU slice so the
            # Mimi/sampling host work of the ranks does not contend
            ncpu = os.cpu_count() or 1
            per = max(1, ncpu // max(1, tp))
            cpus = set(range((self.local_rank * per) % ncpu, (self.local_rank * per) % ncpu + per))
            try:
                os.sched_setaffinity(0, cpus)
            except Exception:  # noqa: BLE001
                pass
            # MOSHI_WORKER_THREADS: torch intra-op threads per TP rank (host ops here are tiny; >1 makes them wait
            # on helper threads that share CPUs with other ranks / the API server)
            torch.set_num_threads(int(os.environ.get("MOSHI_WORKER_THREADS", str(len(cpus)))))
            # vllm-neuron convention: with NEURON_VISIBLE_DEVICES set, the device index is the local rank
            self.device = torch.device(f"neuron:{self.local_rank}")
            self.model_runner = MoshiStage0Runner(self.vllm_config, self.device)

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

        def profile(self, *a, **k):
            return None

        def check_health(self):
            return None

    return MoshiStage0TPWorker


# The stage engine process imports this module to resolve worker_cls (connector validation) and runs the stage
# input processors; with a multiproc executor (TP>1) the workers are separate processes, so apply the backports here.
try:
    from .backports import apply as _apply_backports
    _apply_backports()
except Exception:  # noqa: BLE001
    pass

try:
    MoshiStage0TPWorker = _make_tp_worker()
except Exception:  # vllm_neuron not importable (CPU tests)
    MoshiStage0TPWorker = None
