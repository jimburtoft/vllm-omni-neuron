"""Qwen3-TTS (12Hz, 1.7B) on AWS Neuron via PyTorch Native (torch-neuronx 2.14, SDK 2.33).

Task 003 engine. Self-contained: loads weights straight from the HF safetensors, no transformers /
qwen_tts dependency on the hot path.

Graphs (all `torch.compile(backend="neuron", dynamic=False)`, static shapes):
  * talker_prefill(T_bucket)   : 28L Qwen3 over the prompt embeds, writes KV[0:T], returns last hidden + CB0 logits
  * talker_decode              : 28L Qwen3 one token at `pos`, writes KV[pos], returns hidden + CB0 logits
  * cp_frame                   : the whole 15-step Code Predictor loop for ONE frame, unrolled, greedy,
                                 KV in local tensors (S<=17), returns codes[15] + summed codec embedding
                                 (= next Talker input minus the text term)
  * code2wav(chunk)            : RVQ + SWA transformer + conv upsampler for a fixed number of frames

KV caches are per-layer `register_buffer`s updated with `index_copy_` (the voxtral-native Beta 6 pattern).
RMSNorm uses `torch.rms_norm(x, (H,), w, eps)` with explicit eps (-> AwsNeuronRmsNorm custom call).

mRoPE note: for TTS all three position streams (t/h/w) are equal, so interleaved mRoPE with sections
[24,20,20] degenerates to standard 1-D RoPE over the full head_dim. Verified numerically in tests.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

# ---------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------


def rms(x, w, eps):
    return torch.rms_norm(x, (x.shape[-1],), w, eps)


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


def rope_tables(head_dim, theta, max_pos, dtype=torch.float32):
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    t = torch.arange(max_pos, dtype=torch.float32)
    f = torch.outer(t, inv)
    emb = torch.cat((f, f), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class SafeTensors:
    def __init__(self, path):
        self.f = safe_open(path, framework="pt")
        self.keys = set(self.f.keys())

    def get(self, k):
        return self.f.get_tensor(k)


# ---------------------------------------------------------------------------------------------
# Qwen3 decoder layer with static KV cache (shared by Talker and Code Predictor)
# ---------------------------------------------------------------------------------------------


class Qwen3Layer(nn.Module):
    def __init__(self, hidden, n_heads, n_kv, head_dim, inter, eps, max_ctx, dtype, kv_buffers=True):
        super().__init__()
        self.h, self.nh, self.nkv, self.d, self.eps = hidden, n_heads, n_kv, head_dim, eps
        self.rep = n_heads // n_kv
        self.scale = head_dim ** -0.5
        q = n_heads * head_dim
        kv = n_kv * head_dim
        # weights stored (in, out) so forward is x @ W
        self.w_qkv = nn.Parameter(torch.empty(hidden, q + 2 * kv, dtype=dtype), requires_grad=False)
        self.w_o = nn.Parameter(torch.empty(q, hidden, dtype=dtype), requires_grad=False)
        self.w_gate_up = nn.Parameter(torch.empty(hidden, 2 * inter, dtype=dtype), requires_grad=False)
        self.w_down = nn.Parameter(torch.empty(inter, hidden, dtype=dtype), requires_grad=False)
        self.ln1 = nn.Parameter(torch.ones(hidden, dtype=dtype), requires_grad=False)
        self.ln2 = nn.Parameter(torch.ones(hidden, dtype=dtype), requires_grad=False)
        self.qn = nn.Parameter(torch.ones(head_dim, dtype=dtype), requires_grad=False)
        self.kn = nn.Parameter(torch.ones(head_dim, dtype=dtype), requires_grad=False)
        self.q_sz, self.kv_sz, self.inter = q, kv, inter
        if kv_buffers:
            self.register_buffer("ck", torch.zeros(1, n_kv, max_ctx, head_dim, dtype=dtype), persistent=False)
            self.register_buffer("cv", torch.zeros(1, n_kv, max_ctx, head_dim, dtype=dtype), persistent=False)

    def load(self, st: SafeTensors, prefix):
        g = lambda n: st.get(f"{prefix}.{n}")
        q, k, v = g("self_attn.q_proj.weight"), g("self_attn.k_proj.weight"), g("self_attn.v_proj.weight")
        self.w_qkv.data = torch.cat([q, k, v], 0).t().contiguous().to(self.w_qkv.dtype)
        self.w_o.data = g("self_attn.o_proj.weight").t().contiguous().to(self.w_o.dtype)
        gate, up = g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")
        self.w_gate_up.data = torch.cat([gate, up], 0).t().contiguous().to(self.w_gate_up.dtype)
        self.w_down.data = g("mlp.down_proj.weight").t().contiguous().to(self.w_down.dtype)
        self.ln1.data = g("input_layernorm.weight").to(self.ln1.dtype)
        self.ln2.data = g("post_attention_layernorm.weight").to(self.ln2.dtype)
        self.qn.data = g("self_attn.q_norm.weight").to(self.qn.dtype)
        self.kn.data = g("self_attn.k_norm.weight").to(self.kn.dtype)

    def _qkv(self, x, cos, sin):
        # x [B,T,H]; cos/sin [T,D] -> q [B,nh,T,D], k/v [B,nkv,T,D]
        B, T, _ = x.shape
        hn = rms(x, self.ln1, self.eps)
        qkv = hn @ self.w_qkv
        q, k, v = qkv.split([self.q_sz, self.kv_sz, self.kv_sz], dim=-1)
        q = rms(q.view(B, T, self.nh, self.d), self.qn, self.eps).transpose(1, 2)
        k = rms(k.view(B, T, self.nkv, self.d), self.kn, self.eps).transpose(1, 2)
        v = v.view(B, T, self.nkv, self.d).transpose(1, 2)
        c, s = cos.view(1, 1, T, self.d), sin.view(1, 1, T, self.d)
        q = q * c + rotate_half(q) * s
        k = k * c + rotate_half(k) * s
        return q, k, v

    def _attn(self, q, k, v, mask_add):
        # q [B,nh,T,D], k/v [B,nkv,S,D], mask_add [T,S] additive (fp32)
        B, _, T, D = q.shape
        S = k.shape[2]
        qg = q.reshape(B, self.nkv, self.rep * T, D)  # group heads sharing a kv head
        sc = (qg @ k.transpose(-1, -2)).float() * self.scale  # [B,nkv,rep*T,S]
        sc = sc.view(B, self.nkv, self.rep, T, S) + mask_add.view(1, 1, 1, T, S)
        p = torch.softmax(sc, dim=-1).to(q.dtype).view(B, self.nkv, self.rep * T, S)
        o = (p @ v).view(B, self.nh, T, D)
        return o.transpose(1, 2).reshape(B, T, self.nh * D)

    def _mlp(self, h):
        hn = rms(h, self.ln2, self.eps)
        gu = hn @ self.w_gate_up
        g, u = gu.split([self.inter, self.inter], dim=-1)
        return (F.silu(g) * u) @ self.w_down

    def forward_cached(self, x, cos, sin, pos_idx, mask_add):
        """x [1,T,H]; writes KV at positions pos_idx (1-D, len T) into the static cache, attends over full cache."""
        q, k, v = self._qkv(x, cos, sin)
        self.ck.index_copy_(2, pos_idx, k)
        self.cv.index_copy_(2, pos_idx, v)
        o = self._attn(q, k=self.ck, v=self.cv, mask_add=mask_add)
        h = x + o @ self.w_o
        return h + self._mlp(h)

    def forward_local(self, x, cos, sin, k_cache, v_cache, pos_idx, mask_add):
        """Same but with caller-provided (functional) local KV tensors; returns updated caches."""
        q, k, v = self._qkv(x, cos, sin)
        k_cache = k_cache.index_copy(2, pos_idx, k)
        v_cache = v_cache.index_copy(2, pos_idx, v)
        o = self._attn(q, k_cache, v_cache, mask_add)
        h = x + o @ self.w_o
        return h + self._mlp(h), k_cache, v_cache


# ---------------------------------------------------------------------------------------------
# Talker
# ---------------------------------------------------------------------------------------------


@dataclass
class TalkerCfg:
    hidden: int = 2048
    layers: int = 28
    heads: int = 16
    kv: int = 8
    head_dim: int = 128
    inter: int = 6144
    eps: float = 1e-6
    theta: float = 1e6
    vocab: int = 3072
    text_vocab: int = 151936


class Talker(nn.Module):
    """28L decoder + codec_head. Embedding lookups / text projection stay outside the step graphs
    (they are done once per prompt on the host or in a tiny graph)."""

    def __init__(self, cfg: TalkerCfg, max_ctx: int, dtype):
        super().__init__()
        self.cfg, self.max_ctx, self.dtype = cfg, max_ctx, dtype
        self.layers = nn.ModuleList([
            Qwen3Layer(cfg.hidden, cfg.heads, cfg.kv, cfg.head_dim, cfg.inter, cfg.eps, max_ctx, dtype)
            for _ in range(cfg.layers)])
        self.norm = nn.Parameter(torch.ones(cfg.hidden, dtype=dtype), requires_grad=False)
        self.codec_head = nn.Parameter(torch.empty(cfg.hidden, cfg.vocab, dtype=dtype), requires_grad=False)
        cos, sin = rope_tables(cfg.head_dim, cfg.theta, max_ctx, dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        # suppress mask: tokens [vocab-1024, vocab) except EOS are banned (HF generate suppress_tokens)
        self.register_buffer("logit_bias", torch.zeros(cfg.vocab, dtype=torch.float32), persistent=False)

    def load(self, st: SafeTensors, eos_id: int):
        for i, l in enumerate(self.layers):
            l.load(st, f"talker.model.layers.{i}")
        self.norm.data = st.get("talker.model.norm.weight").to(self.dtype)
        self.codec_head.data = st.get("talker.codec_head.weight").t().contiguous().to(self.dtype)
        lb = torch.zeros(self.cfg.vocab)
        lb[self.cfg.vocab - 1024:] = float("-inf")
        lb[eos_id] = 0.0
        self.logit_bias.copy_(lb)

    def reset(self):
        for l in self.layers:
            l.ck.zero_(); l.cv.zero_()

    def _mask(self, pos_idx):
        # additive causal mask [T, S]: key slot j visible iff j <= pos_i
        j = torch.arange(self.max_ctx, device=pos_idx.device).view(1, -1)
        ok = j <= pos_idx.view(-1, 1)
        return torch.where(ok, 0.0, -1e30).to(torch.float32)

    def step(self, x, pos_idx):
        """x [1,T,H] input embeds at positions pos_idx [T] (cache slot == rope position).
        Returns (post-norm last hidden [1,1,H], raw CB0 logits [1,V] fp32)."""
        cos = self.cos.index_select(0, pos_idx)
        sin = self.sin.index_select(0, pos_idx)
        m = self._mask(pos_idx)
        h = x
        for l in self.layers:
            h = l.forward_cached(h, cos, sin, pos_idx, m)
        h_last = rms(h[:, -1:, :], self.norm, self.cfg.eps)
        logits = (h_last @ self.codec_head).float().view(1, -1)
        return h_last, logits

    def decode(self, cb0_embed, res_embed, text_embed, pos_idx):
        """Next-token step: input = cb0 embed + sum(residual codebook embeds) + text/pad term."""
        return self.step(cb0_embed + res_embed + text_embed, pos_idx)


# ---------------------------------------------------------------------------------------------
# Code Predictor (5L) — whole frame loop in one graph
# ---------------------------------------------------------------------------------------------


@dataclass
class CPCfg:
    hidden: int = 1024
    layers: int = 5
    heads: int = 16
    kv: int = 8
    head_dim: int = 128
    inter: int = 3072
    eps: float = 1e-6
    theta: float = 1e6
    vocab: int = 2048
    groups: int = 16
    talker_hidden: int = 2048


class CodePredictor(nn.Module):
    def __init__(self, cfg: CPCfg, dtype):
        super().__init__()
        self.cfg, self.dtype = cfg, dtype
        S = cfg.groups + 1  # 17 positions max (hidden + cb0 + 15 generated, last one never re-fed)
        self.S = S
        self.layers = nn.ModuleList([
            Qwen3Layer(cfg.hidden, cfg.heads, cfg.kv, cfg.head_dim, cfg.inter, cfg.eps, S, dtype, kv_buffers=False)
            for _ in range(cfg.layers)])
        self.norm = nn.Parameter(torch.ones(cfg.hidden, dtype=dtype), requires_grad=False)
        G = cfg.groups - 1
        self.proj_w = nn.Parameter(torch.empty(cfg.talker_hidden, cfg.hidden, dtype=dtype), requires_grad=False)
        self.proj_b = nn.Parameter(torch.zeros(cfg.hidden, dtype=dtype), requires_grad=False)
        # heads stacked: [G, hidden, vocab]; embeds stacked: [G, vocab, talker_hidden]
        self.heads = nn.Parameter(torch.empty(G, cfg.hidden, cfg.vocab, dtype=dtype), requires_grad=False)
        self.embeds = nn.Parameter(torch.empty(G, cfg.vocab, cfg.talker_hidden, dtype=dtype), requires_grad=False)
        cos, sin = rope_tables(cfg.head_dim, cfg.theta, S, dtype)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def load(self, st: SafeTensors):
        p = "talker.code_predictor"
        for i, l in enumerate(self.layers):
            l.load(st, f"{p}.model.layers.{i}")
        self.norm.data = st.get(f"{p}.model.norm.weight").to(self.dtype)
        self.proj_w.data = st.get(f"{p}.small_to_mtp_projection.weight").t().contiguous().to(self.dtype)
        self.proj_b.data = st.get(f"{p}.small_to_mtp_projection.bias").to(self.dtype)
        G = self.cfg.groups - 1
        self.heads.data = torch.stack([st.get(f"{p}.lm_head.{i}.weight").t() for i in range(G)]).contiguous().to(self.dtype)
        self.embeds.data = torch.stack([st.get(f"{p}.model.codec_embedding.{i}.weight") for i in range(G)]).contiguous().to(self.dtype)

    def frame(self, talker_hidden, cb0_embed):
        """talker_hidden [1,1,2048] (post-norm last hidden), cb0_embed [1,1,2048] = talker codec_embedding(cb0).
        Returns codes [15] (int64) and sum of the 15 residual codec embeddings [1,1,2048]."""
        cfg = self.cfg
        dev = talker_hidden.device
        L = len(self.layers)
        kc = [torch.zeros(1, cfg.kv, self.S, cfg.head_dim, dtype=self.dtype, device=dev) for _ in range(L)]
        vc = [torch.zeros(1, cfg.kv, self.S, cfg.head_dim, dtype=self.dtype, device=dev) for _ in range(L)]
        j = torch.arange(self.S, device=dev)

        def run(x, pos):  # x [1,T,1024], pos [T]
            cos = self.cos.index_select(0, pos)
            sin = self.sin.index_select(0, pos)
            m = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32)
            h = x
            for i, l in enumerate(self.layers):
                h, kc[i], vc[i] = l.forward_local(h, cos, sin, kc[i], vc[i], pos, m)
            return rms(h[:, -1:, :], self.norm, cfg.eps)

        x0 = torch.cat([talker_hidden, cb0_embed], dim=1) @ self.proj_w + self.proj_b  # [1,2,1024]
        h = run(x0, torch.arange(2, device=dev))
        codes = []
        emb_sum = torch.zeros_like(talker_hidden)
        for g in range(cfg.groups - 1):
            logits = (h @ self.heads[g]).float().view(-1)
            c = torch.argmax(logits, dim=-1).view(1)
            codes.append(c)
            e = F.embedding(c, self.embeds[g]).view(1, 1, -1)  # [1,1,2048]
            emb_sum = emb_sum + e
            if g < cfg.groups - 2:
                xg = e @ self.proj_w + self.proj_b
                h = run(xg, torch.full((1,), g + 2, device=dev, dtype=torch.long))
        return torch.cat(codes), emb_sum


# ---------------------------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------------------------


class _Prefill(nn.Module):
    def __init__(self, t):
        super().__init__(); self.t = t

    def forward(self, x, pos):
        return self.t.step(x, pos)


class _Decode(nn.Module):
    def __init__(self, t):
        super().__init__(); self.t = t

    def forward(self, x, pos):
        return self.t.step(x, pos)


class _CPFrame(nn.Module):
    def __init__(self, cp):
        super().__init__(); self.cp = cp

    def forward(self, h, e0):
        return self.cp.frame(h, e0)



class TTSEngine:
    def __init__(self, model_dir, device="neuron", max_ctx=1024, dtype=torch.bfloat16, compile=True,
                 prefill_buckets=(64, 128, 256)):
        self.model_dir, self.device, self.dtype, self.max_ctx = model_dir, device, dtype, max_ctx
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        self.cfg = cfg
        tc = cfg["talker_config"]
        self.eos = tc["codec_eos_token_id"]
        st = SafeTensors(os.path.join(model_dir, "model.safetensors"))
        self.talker = Talker(TalkerCfg(), max_ctx, dtype)
        self.talker.load(st, self.eos)
        self.cp = CodePredictor(CPCfg(), dtype)
        self.cp.load(st)
        # host-side embedding tables (prompt construction only)
        self.codec_emb = st.get("talker.model.codec_embedding.weight").to(dtype)
        self.text_emb = st.get("talker.model.text_embedding.weight").to(dtype)
        self.tp_w1 = st.get("talker.text_projection.linear_fc1.weight").to(dtype)
        self.tp_b1 = st.get("talker.text_projection.linear_fc1.bias").to(dtype)
        self.tp_w2 = st.get("talker.text_projection.linear_fc2.weight").to(dtype)
        self.tp_b2 = st.get("talker.text_projection.linear_fc2.bias").to(dtype)
        self.codec_emb_dev = None
        self.talker.to(device)
        self.cp.to(device)
        self.codec_emb_dev = self.codec_emb.to(device)
        self.prefill_buckets = tuple(sorted(prefill_buckets))
        self._talker_step = self.talker.step
        self._cp_frame = self.cp.frame
        self._talker_step_dec = self.talker.step
        self.prefill_mod, self.decode_mod, self.cp_mod = _Prefill(self.talker), _Decode(self.talker), _CPFrame(self.cp)
        if compile:
            # distinct nn.Module wrappers (distinct code objects) per static shape: prefill bucket vs T=1 decode
            self._talker_step = torch.compile(self.prefill_mod, backend="neuron", dynamic=False)
            self._talker_step_dec = torch.compile(self.decode_mod, backend="neuron", dynamic=False)
            self._cp_frame = torch.compile(self.cp_mod, backend="neuron", dynamic=False)

    # -- prompt (host) ----------------------------------------------------------------------
    def text_project(self, ids):
        e = F.embedding(ids, self.text_emb)
        h = F.silu(F.linear(e, self.tp_w1, self.tp_b1))
        return F.linear(h, self.tp_w2, self.tp_b2)

    # -- prompt (host, VoiceDesign non-streaming path; mirrors Qwen3TTSForConditionalGeneration.generate) --
    def build_prompt(self, input_ids, instruct_ids, language):
        """input_ids: tokenized '<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n' [1,L];
        instruct_ids: tokenized '<|im_start|>user\n{instruct}<|im_end|>\n' [1,Li] or None."""
        cfg, tc = self.cfg, self.cfg["talker_config"]
        ce = lambda ids: F.embedding(torch.tensor([ids]), self.codec_emb)
        parts = []
        if instruct_ids is not None:
            parts.append(self.text_project(instruct_ids))
        bos, eos, pad = self.text_project(torch.tensor([[cfg["tts_bos_token_id"], cfg["tts_eos_token_id"],
                                                         cfg["tts_pad_token_id"]]])).chunk(3, dim=1)
        lang = None if (language is None or language.lower() == "auto") else tc["codec_language_id"][language.lower()]
        if lang is None:
            pre = [tc["codec_nothink_id"], tc["codec_think_bos_id"], tc["codec_think_eos_id"]]
        else:
            pre = [tc["codec_think_id"], tc["codec_think_bos_id"], lang, tc["codec_think_eos_id"]]
        codec_in = torch.cat([ce(pre), ce([tc["codec_pad_id"], tc["codec_bos_id"]])], dim=1)
        role = self.text_project(input_ids[:, :3])
        n = codec_in.shape[1]
        t = torch.cat([pad.expand(-1, n - 2, -1), bos], dim=1) + codec_in[:, :-1]
        body = self.text_project(input_ids[:, 3:-5])
        tail = torch.cat([torch.cat([body, eos], dim=1) + ce([tc["codec_pad_id"]] * (body.shape[1] + 1)),
                          pad + ce([tc["codec_bos_id"]])], dim=1)
        parts += [role, t, tail]
        return torch.cat(parts, dim=1), pad  # inputs_embeds, tts_pad_embed (= trailing text in non-streaming)

    # -- generation --------------------------------------------------------------------------
    @torch.no_grad()
    def generate(self, inputs_embeds, tts_pad_embed, max_new=600, rep_penalty=1.05, min_new=2, timings=None,
                 trailing=None):
        """Greedy generation (talker + code predictor). Returns codes [F,16] int64 (cpu).
        trailing: optional [N,H] streaming-text rows; decode step k adds trailing[k] (k < N) instead of tts_pad."""
        import time
        dev = self.device
        T = inputs_embeds.shape[1]
        Tb = next(b for b in self.prefill_buckets if b >= T)
        pad = Tb - T
        assert T + max_new + 1 < self.max_ctx - pad, "max_ctx too small"
        # Front-pad so the real last token is last. Pad tokens are written to the top cache slots
        # (never visible to real tokens: the mask is j <= pos and real positions stay < max_ctx - pad).
        x = torch.zeros(1, Tb, inputs_embeds.shape[-1], dtype=self.dtype)
        x[:, pad:] = inputs_embeds.to(self.dtype)
        pos = torch.cat([torch.arange(self.max_ctx - pad, self.max_ctx), torch.arange(T)]).long()
        pad_dev = tts_pad_embed.to(self.dtype).to(dev)
        self.talker.reset()
        tm = dict(prefill=0.0, cp=0.0, talker=0.0, host=0.0)
        t0 = time.perf_counter()
        h, logits = self._talker_step(x.to(dev), pos.to(dev))
        lg = logits.cpu()
        tm["prefill"] = time.perf_counter() - t0
        V = lg.shape[-1]
        ban = torch.zeros(V, dtype=torch.bool); ban[V - 1024:] = True; ban[self.eos] = False
        seen = torch.zeros(V, dtype=torch.bool)
        frames = []
        p = T
        for step in range(max_new):
            ta = time.perf_counter()
            l = lg.view(-1).clone()
            l = torch.where(seen & (l > 0), l / rep_penalty, torch.where(seen, l * rep_penalty, l))
            l[ban] = float("-inf")
            if step < min_new:
                l[self.eos] = float("-inf")
            cb0 = int(torch.argmax(l))
            tm["host"] += time.perf_counter() - ta
            if cb0 == self.eos:
                break
            seen[cb0] = True
            tb = time.perf_counter()
            cb0_t = torch.tensor([cb0], device=dev)
            cb0_e = F.embedding(cb0_t, self.codec_emb_dev).view(1, 1, -1)
            res, emb_sum = self._cp_frame(h, cb0_e)
            tc = time.perf_counter()
            txt = pad_dev if trailing is None or step >= trailing.shape[0] else \
                trailing[step].to(self.dtype).view(1, 1, -1).to(dev)
            h, logits = self._talker_step_dec(cb0_e + emb_sum + txt, torch.tensor([p], device=dev))
            lg = logits.cpu()
            td = time.perf_counter()
            frames.append(torch.cat([torch.tensor([cb0]), res.cpu()]))
            tm["cp"] += tc - tb; tm["talker"] += td - tc
            p += 1
        if timings is not None:
            timings.update(tm); timings["frames"] = len(frames)
        return torch.stack(frames) if frames else torch.zeros(0, 16, dtype=torch.long)
