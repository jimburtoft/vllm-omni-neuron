"""Stage 1 (LLM_GENERATION) Neuron worker for Moshi: streaming Mimi decoder rows.

Input per request (stock `talker2code2wav_async_chunk`): flat codebook-major de-delayed agent codes [8 * F] as
`prompt_token_ids` (new frames only) or `additional_information["codes"]["audio"]`. Each request owns one decoder
row, so every frame is decoded exactly once with streaming state carried on device. Rows from different requests
decode together in one graph call per frame. Output: `multimodal_outputs[{"model_outputs": wav fp32, "sr": 24000}]`
(new samples only, like the stock 0.29 PersonaPlexCode2Wav; MOSHI_C2W_CUMULATIVE=1 returns the running total).
"""
from __future__ import annotations

import os
import time
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_omni.outputs import OmniModelRunnerOutput
from vllm_omni.worker.omni_connector_model_runner_mixin import OmniConnectorModelRunnerMixin, needs_omni_connector

from ..config import FRAME
from .ar_worker import _ThinWorker
from .common import compile_fn, model_dir

logger = init_logger(__name__)
K = 8


class MoshiCode2WavRunner(OmniConnectorModelRunnerMixin):
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.device = device
        self.B = int(vllm_config.scheduler_config.max_num_seqs)
        self.requests: dict[str, dict] = {}
        self.model_intermediate_buffer: dict[str, Any] = {}
        self.row_of: dict[str, int] = {}
        self.free = list(range(self.B))
        self.seen: dict[str, int] = {}       # rid -> frames already decoded
        self._async_chunk = getattr(self.model_config, "async_chunk", False)
        # duplex data plane slices the *cumulative* audio of a request by the samples already sent
        self.cumulative = os.environ.get("MOSHI_C2W_CUMULATIVE", "0") == "1"
        self.acc: dict[str, list] = {}
        if needs_omni_connector(self.model_config):
            self.init_omni_connectors(model_config=self.model_config)

    def load_model(self):
        from ..engine import MoshiDecoderRows
        t0 = time.time()
        self.dec = MoshiDecoderRows(B=self.B, device=self.device, compile_fn=compile_fn(),
                                    model_dir=model_dir(self.model_config.model))
        logger.info("[MoshiC2W] loaded in %.1fs (B=%d)", time.time() - t0, self.B)

    @torch.inference_mode()
    def warmup(self):
        t0 = time.time()
        a = torch.ones(self.B, dtype=torch.bool)
        for i in range(3):
            self.dec.decode(torch.zeros(self.B, K, dtype=torch.long), a, a if i == 0 else ~a)
        return time.time() - t0

    def _codes(self, st, nc, n):
        info = st.get("info") or {}
        codes = info.get("codes") if isinstance(info, dict) else None
        audio = codes.get("audio") if isinstance(codes, dict) else None
        if isinstance(audio, torch.Tensor) and audio.numel():
            return audio.reshape(-1).to(torch.long)
        return torch.tensor(st["tokens"][nc: nc + n], dtype=torch.long)

    @torch.inference_mode()
    def execute_model(self, scheduler_output, intermediate_tensors=None):
        from vllm_omni.engine.serialization import deserialize_additional_information
        if hasattr(self, "_omni_connector"):
            for request in getattr(scheduler_output, "pending_input_registrations", []):
                self.register_chunk_recv(request)
            self.recv_full_payload_inputs(scheduler_output)
        for rid in scheduler_output.finished_req_ids:
            self.requests.pop(rid, None)
            self.seen.pop(rid, None)
            self.acc.pop(rid, None)
            b = self.row_of.pop(rid, None)
            if b is not None:
                self.free.append(b)
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
        # gather new frames per request
        req_ids, frames = [], {}
        for rid, n in scheduler_output.num_scheduled_tokens.items():
            st = self.requests.get(rid)
            if st is None:
                continue
            req_ids.append(rid)
            flat = self._codes(st, ncomp.get(rid, 0), n)
            if flat.numel() == 0 or flat.numel() % K:
                continue
            f = flat.view(K, -1).t()  # [F,8]
            if f.shape[0] == 0 or bool((f >= 2048).any()) or bool((f < 0).any()):
                f = f[((f >= 0) & (f < 2048)).all(1)]
            if f.shape[0]:
                if rid not in self.row_of:
                    if not self.free:
                        logger.warning("[MoshiC2W] no free decoder row for %s", rid)
                        continue
                    self.row_of[rid] = self.free.pop(0)
                frames[rid] = f
        out_wav = {rid: [] for rid in req_ids}
        # decode frame-major: step j decodes the j-th new frame of every request that has one
        maxf = max((f.shape[0] for f in frames.values()), default=0)
        for j in range(maxf):
            codes = torch.zeros(self.B, K, dtype=torch.long)
            active = torch.zeros(self.B, dtype=torch.bool)
            reset = torch.zeros(self.B, dtype=torch.bool)
            for rid, f in frames.items():
                if j < f.shape[0]:
                    b = self.row_of[rid]
                    codes[b], active[b] = f[j], True
                    reset[b] = self.seen.get(rid, 0) == 0
                    self.seen[rid] = self.seen.get(rid, 0) + 1
            pcm = self.dec.decode(codes, active, reset)
            for rid, f in frames.items():
                if j < f.shape[0]:
                    out_wav[rid].append(pcm[self.row_of[rid]])
        wavs = []
        for rid in req_ids:
            new = torch.cat(out_wav[rid]) if out_wav[rid] else torch.zeros(0)
            if self.cumulative:
                a = self.acc.setdefault(rid, [])
                if new.numel():
                    a.append(new)
                wavs.append(torch.cat(a) if a else torch.zeros(0))
            else:
                wavs.append(new)
        self._last = (req_ids, wavs)
        return None

    @torch.inference_mode()
    def sample_tokens(self, grammar_output=None):
        from vllm_omni.utils.mm_outputs import partition_payload_list
        req_ids, wavs = self._last
        payloads = [{"model_outputs": w.float(), "sr": torch.tensor(24000)} for w in wavs]
        if self._async_chunk:
            inter, mm = partition_payload_list(payloads) if payloads else (None, None)
        else:
            inter, mm = payloads, payloads
        out = OmniModelRunnerOutput(req_ids=req_ids, req_id_to_index={r: i for i, r in enumerate(req_ids)},
                                    sampled_token_ids=[], logprobs=None, prompt_logprobs_dict={}, pooler_output=None,
                                    multimodal_outputs=mm, inter_stage_outputs=inter, num_nans_in_logits={})
        out.omni_connector_output = self.get_omni_connector_output() if hasattr(self, "_lock") else None
        return out


class MoshiCode2WavWorker(_ThinWorker):
    model_runner_cls = MoshiCode2WavRunner
