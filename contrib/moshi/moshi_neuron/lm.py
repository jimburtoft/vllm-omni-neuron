"""Moshi LM (Helium-7B temporal transformer + per-step-weight depformer) as one static per-frame graph.

  MoshiFrame(tokens [B,17] long, reset [B], active [B], gumbel_text [B,32000], gumbel_audio [B,8,2048],
             force_text [B], force_audio [B,8], use_force [B]) -> out tokens [B,9] (+ debug tensors)

Semantics follow kyutai `moshi.models.lm` (LMModel.forward_text / forward_depformer / LMGen.depformer_step):
  * input = text_emb(tok[:,0]) + sum_k emb[k](tok[:,1+k]), with token -1 giving an exact zero vector
  * 32 x [rms_norm_f32 -> MHA (interleaved RoPE, ring KV with `context` slots) -> residual -> rms_norm_f32 ->
    SiLU-gated FFN -> residual], out_norm, text_linear
  * text token sampled on device (Gumbel-max over top-k of logits/T; T <= 0 means greedy)
  * depformer: 8 steps, step k uses weight set k; the input is depformer_in[k](transformer_out) +
    (depformer_text_emb(text) if k == 0 else depformer_emb[k-1](audio_{k-1})); 6 layers, no positional
    embedding, causal over the steps of this frame; head linears[k]; audio sampled on device.
The delay / token-ring bookkeeping of LMGen runs on the host (`LMGenHost`), since it is a few integer ops per row.
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .config import LMConfig
from .mimi import NEG, apply_rope_interleaved, ring_positions, rope_tables


def rms_f32(x, alpha, eps):
    xf = x.float()
    var = eps + torch.mean(xf * xf, dim=-1, keepdim=True)
    return (xf * (alpha.float() * torch.rsqrt(var))).to(x.dtype)


def sample(logits, gumbel, temp: float, top_k: int):
    """logits [..., V] fp32. Gumbel-max sampling from softmax(logits/temp) restricted to the top-k (same
    distribution as moshi `sample_top_k`). temp <= 0 means greedy argmax."""
    if temp <= 0:
        return torch.argmax(logits, dim=-1)
    l = logits / temp
    if 0 < top_k < l.shape[-1]:
        kth = torch.topk(l, top_k, dim=-1).values[..., -1:]
        l = torch.where(l >= kth, l, torch.full_like(l, NEG))
    return torch.argmax(l + gumbel, dim=-1)


class MoshiLM(nn.Module):
    def __init__(self, path, B, cfg: LMConfig = LMConfig(), dtype=torch.bfloat16, context=None):
        super().__init__()
        self.cfg, self.B, self.dtype = cfg, B, dtype
        self.C = context or cfg.context
        f = safe_open(path, framework="pt")
        g = lambda k: f.get_tensor(k).to(dtype)
        P = lambda t: nn.Parameter(t.contiguous(), requires_grad=False)
        H, D, L = cfg.num_heads, cfg.head_dim, cfg.num_layers

        # ---- input embeddings: one flat table [text (32001) | audio_0 (2049) | ... | audio_15 (2049)]
        tabs = [g("text_emb.weight")] + [g(f"emb.{k}.weight") for k in range(cfg.n_q)]
        self.emb = P(torch.cat(tabs, 0))
        offs = [0]
        for t in tabs[:-1]:
            offs.append(offs[-1] + t.shape[0])
        self.register_buffer("emb_off", torch.tensor(offs).view(1, -1), persistent=False)

        self.layers = nn.ModuleList()
        for i in range(L):
            p = f"transformer.layers.{i}."
            self.layers.append(nn.ParameterDict(dict(
                n1=P(g(p + "norm1.alpha").view(-1)), n2=P(g(p + "norm2.alpha").view(-1)),
                wqkv=P(g(p + "self_attn.in_proj_weight")), wo=P(g(p + "self_attn.out_proj.weight")),
                win=P(g(p + "gating.linear_in.weight")), wout=P(g(p + "gating.linear_out.weight")))))
            self.register_buffer(f"k{i}", torch.zeros(B * H * self.C, D, dtype=dtype), persistent=False)
            self.register_buffer(f"v{i}", torch.zeros(B * H * self.C, D, dtype=dtype), persistent=False)
        self.out_norm = P(g("out_norm.alpha").view(-1))
        self.text_linear = P(g("text_linear.weight"))
        self.register_buffer("offset", torch.zeros(B, dtype=torch.long), persistent=False)
        self.register_buffer("hoff", torch.arange(H), persistent=False)

        # ---- depformer
        Q, dd = cfg.dep_q, cfg.dep_dim
        self.dep_in = P(torch.stack([g(f"depformer_in.{k}.weight") for k in range(Q)]))         # [8,1024,4096]
        self.dep_text_emb = P(g("depformer_text_emb.weight"))                                    # [32001,1024]
        self.dep_emb = P(torch.stack([g(f"depformer_emb.{k}.weight") for k in range(Q - 1)]))   # [7,2049,1024]
        self.dep_heads = P(torch.stack([g(f"linears.{k}.weight") for k in range(Q)]))           # [8,2048,1024]
        self.dep_layers = nn.ModuleList()
        for i in range(cfg.dep_layers):
            p = f"depformer.layers.{i}."
            self.dep_layers.append(nn.ParameterDict(dict(
                n1=P(g(p + "norm1.alpha").view(-1)), n2=P(g(p + "norm2.alpha").view(-1)),
                wqkv=P(g(p + "self_attn.in_proj_weight").view(Q, 3 * dd, dd)),
                wo=P(g(p + "self_attn.out_proj.weight").view(Q, dd, dd)),
                win=P(torch.stack([g(p + f"gating.{k}.linear_in.weight") for k in range(Q)])),
                wout=P(torch.stack([g(p + f"gating.{k}.linear_out.weight") for k in range(Q)])))))
        self.onehot_emb = os.environ.get("MOSHI_ONEHOT_EMB", "0") == "1"

    # ---------------------------------------------------------------- temporal transformer
    def temporal(self, tokens, reset, active, ctx: int):
        cfg, B, H, D, C = self.cfg, self.B, self.cfg.num_heads, self.cfg.head_dim, self.C
        valid = (tokens >= 0).to(self.dtype).unsqueeze(-1)
        x = (F.embedding(tokens.clamp(min=0) + self.emb_off, self.emb) * valid).sum(1, keepdim=True)  # [B,1,dim]
        off = torch.where(reset, torch.zeros_like(self.offset), self.offset)
        cos, sin = rope_tables(off.view(B, 1), D, cfg.max_period)
        pos_k = ring_positions(off, 1, C)[:, :ctx]
        delta = off.view(B, 1) - pos_k
        ok = (pos_k >= 0) & (delta >= 0) & (delta < cfg.context)
        mask = torch.where(ok, 0.0, NEG).to(torch.float32).view(B, 1, 1, ctx)
        widx = ((torch.arange(B, device=x.device).view(B, 1) * H + self.hoff.view(1, H)) * C
                + (off % C).view(B, 1)).reshape(-1)
        for i, l in enumerate(self.layers):
            h = rms_f32(x, l["n1"], cfg.eps)
            q, k, v = F.linear(h, l["wqkv"]).view(B, 1, 3, H, D).permute(2, 0, 3, 1, 4)  # [B,H,1,D]
            q = apply_rope_interleaved(q, cos, sin)
            k = apply_rope_interleaved(k, cos, sin)
            kb, vb = getattr(self, f"k{i}"), getattr(self, f"v{i}")
            kb.index_copy_(0, widx, k.reshape(-1, D))
            vb.index_copy_(0, widx, v.reshape(-1, D))
            K = kb.view(B, H, C, D)[:, :, :ctx]
            V = vb.view(B, H, C, D)[:, :, :ctx]
            sc = (q @ K.transpose(-1, -2)).float() * (D ** -0.5) + mask
            o = (torch.softmax(sc, -1).to(V.dtype) @ V).transpose(1, 2).reshape(B, 1, H * D)
            x = x + F.linear(o, l["wo"])
            h = rms_f32(x, l["n2"], cfg.eps)
            gu = F.linear(h, l["win"]).view(B, 1, 2, -1)
            x = x + F.linear(F.silu(gu[..., 0, :]) * gu[..., 1, :], l["wout"])
        self.offset.copy_(torch.where(active, off + 1, self.offset))
        tout = rms_f32(x, self.out_norm, cfg.eps)              # [B,1,dim]
        return tout, F.linear(tout, self.text_linear).float()[:, 0]  # [B,32000]

    # ---------------------------------------------------------------- depformer
    def _lookup(self, table, idx):
        if self.onehot_emb:
            oh = (torch.arange(table.shape[0], device=idx.device).view(1, -1) == idx.view(-1, 1)).to(table.dtype)
            return oh @ table
        return F.embedding(idx, table)

    def depformer(self, tout, text, gumbel_audio, temp: float, top_k: int, force_audio=None, use_force=None):
        cfg, B = self.cfg, self.B
        Hd, Dd = cfg.dep_heads, cfg.dep_dim // cfg.dep_heads
        tout = tout[:, 0]                                      # [B,4096]
        ks, vs = [[] for _ in self.dep_layers], [[] for _ in self.dep_layers]
        prev = self._lookup(self.dep_text_emb, text)           # [B,1024]
        codes, logits_all = [], []
        for s in range(cfg.dep_q):
            x = (F.linear(tout, self.dep_in[s]) + prev).view(B, 1, -1)
            for i, l in enumerate(self.dep_layers):
                h = rms_f32(x, l["n1"], cfg.eps)
                q, k, v = F.linear(h, l["wqkv"][s]).view(B, 1, 3, Hd, Dd).permute(2, 0, 3, 1, 4)
                ks[i].append(k)
                vs[i].append(v)
                K = torch.cat(ks[i], 2) if s else k
                V = torch.cat(vs[i], 2) if s else v
                sc = (q @ K.transpose(-1, -2)).float() * (Dd ** -0.5)
                o = (torch.softmax(sc, -1).to(V.dtype) @ V).transpose(1, 2).reshape(B, 1, -1)
                x = x + F.linear(o, l["wo"][s])
                h = rms_f32(x, l["n2"], cfg.eps)
                gu = F.linear(h, l["win"][s]).view(B, 1, 2, -1)
                x = x + F.linear(F.silu(gu[..., 0, :]) * gu[..., 1, :], l["wout"][s])
            lg = F.linear(x[:, 0], self.dep_heads[s]).float()  # [B,2048]
            logits_all.append(lg)
            c = sample(lg, gumbel_audio[:, s], temp, top_k)
            if force_audio is not None:
                c = torch.where(use_force, force_audio[:, s], c)
            codes.append(c)
            if s + 1 < cfg.dep_q:
                prev = self._lookup(self.dep_emb[s], c)
        return torch.stack(codes, 1), torch.stack(logits_all, 1)


class MoshiFrame(nn.Module):
    """The compiled unit: one 80 ms LM step for all B rows. Static config: ctx bucket, sampling, debug outputs."""

    def __init__(self, lm: MoshiLM, ctx: int, temp_text=0.7, top_k_text=25, temp=0.8, top_k=250, debug=False):
        super().__init__()
        self.lm, self.ctx, self.debug = lm, ctx, debug
        self.tt, self.kt, self.ta, self.ka = float(temp_text), int(top_k_text), float(temp), int(top_k)

    def forward(self, tokens, reset, active, gumbel_text, gumbel_audio, force_text, force_audio, use_force):
        tout, tl = self.lm.temporal(tokens, reset, active, self.ctx)
        text = sample(tl, gumbel_text, self.tt, self.kt)
        text = torch.where(use_force, force_text, text)
        audio, al = self.lm.depformer(tout, text, gumbel_audio, self.ta, self.ka, force_audio, use_force)
        out = torch.cat([text.view(-1, 1), audio], 1)
        if self.debug:
            return out, tl, al, tout[:, 0].float()
        return out


class LMGenHost:
    """Host port of moshi `LMGen._step` bookkeeping (token ring with delays) for B rows.

    step(user_codes [B,8]) -> input tokens [B,17] for the device frame; then commit(out [B,9]) -> output frame
    [B,9] in the de-delayed layout the Mimi decoder consumes, or None while offset <= max_delay."""

    def __init__(self, B, cfg: LMConfig = LMConfig()):
        self.B, self.cfg = B, cfg
        self.K = cfg.num_codebooks
        self.delays = torch.tensor(cfg.delays, dtype=torch.long)
        self.max_delay = int(self.delays.max())
        self.CT = self.max_delay + 2
        self.initial = torch.tensor([cfg.text_card] + [cfg.card] * cfg.n_q, dtype=torch.long)
        self.cache = torch.full((B, self.K, self.CT), -2, dtype=torch.long)
        self.offsets = torch.zeros(B, dtype=torch.long)

    def reset(self, rows):
        m = torch.zeros(self.B, dtype=torch.bool)
        m[list(rows)] = True
        self.cache = torch.where(m.view(-1, 1, 1), torch.full_like(self.cache, -2), self.cache)
        self.offsets = torch.where(m, torch.zeros_like(self.offsets), self.offsets)

    def inputs(self, user_codes, active=None):
        cfg, B, CT = self.cfg, self.B, self.CT
        active = torch.ones(B, dtype=torch.bool) if active is None else active
        d_user = self.delays[cfg.dep_q + 1:]
        wp = (self.offsets.view(B, 1) + d_user.view(1, -1)) % CT                  # [B,8]
        self.cache = self.cache.clone()
        sub = self.cache[:, cfg.dep_q + 1:]                                        # view [B,8,CT]
        old = sub.gather(2, wp.unsqueeze(-1)).squeeze(-1)
        sub.scatter_(2, wp.unsqueeze(-1), torch.where(active.view(B, 1), user_codes.long(), old).unsqueeze(-1))
        is_init = (self.offsets.view(B, 1) <= self.delays.view(1, -1)) | ~active.view(B, 1)
        pos = (self.offsets % CT).view(B, 1, 1).expand(B, self.K, 1)
        inp = self.cache.gather(2, pos).squeeze(-1)
        return torch.where(is_init, self.initial.view(1, -1), inp)

    def commit(self, out, active=None):
        cfg, B, CT = self.cfg, self.B, self.CT
        active = torch.ones(B, dtype=torch.bool) if active is None else active
        self.offsets = torch.where(active, self.offsets + 1, self.offsets)
        pos = (self.offsets % CT).view(B, 1, 1)
        self.cache = self.cache.clone()
        gen = self.cache[:, :cfg.dep_q + 1]
        old = gen.gather(2, pos.expand(B, cfg.dep_q + 1, 1)).squeeze(-1)
        gen.scatter_(2, pos.expand(B, cfg.dep_q + 1, 1),
                     torch.where(active.view(B, 1), out.long(), old).unsqueeze(-1))
        gd = self.delays[:cfg.dep_q + 1]
        idx = (self.offsets.view(B, 1) - self.max_delay + gd.view(1, -1)) % CT
        res = self.cache.gather(2, idx.unsqueeze(-1)).squeeze(-1)                 # [B,9]
        ready = (self.offsets > self.max_delay) & active
        return res, ready
