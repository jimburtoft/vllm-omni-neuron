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


def _fp8_rowwise(w, fmax):
    """w [out, in] -> (fp8 [out, in], scale [out] fp32) with per-output-row scale = amax / fmax."""
    sc = (w.float().abs().amax(1).clamp(min=1e-8) / fmax)
    q = (w.float() / sc[:, None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn)
    return q, sc


def _fp8_group(w, fmax, g):
    """w [out, in] -> (fp8 [out, in], scale [out, in/g] fp32): per-(row, input-group) scales."""
    o, i = w.shape
    wg = w.float().view(o, i // g, g)
    sc = wg.abs().amax(-1).clamp(min=1e-8) / fmax
    q = (wg / sc[..., None]).clamp(-fmax, fmax).to(torch.float8_e4m3fn).view(o, i)
    return q, sc


def lin8g(x, wq, sc, dtype, g):
    """y = x @ W^T with W = wq * sc per (row, group): dequantize on chip (cast + group scale), then matmul."""
    o, i = wq.shape
    w = (wq.to(dtype).view(o, i // g, g) * sc.to(dtype)[..., None]).view(o, i)
    return F.linear(x, w)


def _i8_group(w, g):
    """w [out, in] -> (int8 [out, in], scale [out, in/g] fp32), symmetric per (row, group)."""
    o, i = w.shape
    wg = w.float().view(o, i // g, g)
    sc = wg.abs().amax(-1).clamp(min=1e-8) / 127.0
    q = torch.round(wg / sc[..., None]).clamp(-127, 127).to(torch.int8).view(o, i)
    return q, sc


def lin8(x, wq, sc, dtype):
    """y = x @ (wq * sc)^T with the per-row scale applied to the output (weight-only FP8, dequant on chip).
    The bf16 view of the weight is a graph temporary; for B<=8 decode the compiler streams it tile by tile, but at
    load time the runtime reserves scratchpad for the largest such temporary."""
    return F.linear(x, wq.to(dtype)) * sc.to(dtype)


def rms_f32(x, alpha, eps):
    xf = x.float()
    var = eps + torch.mean(xf * xf, dim=-1, keepdim=True)
    return (xf * (alpha.float() * torch.rsqrt(var))).to(x.dtype)


def sample(logits, gumbel, temp: float, top_k: int):
    """logits [..., V] fp32. Gumbel-max sampling from softmax(logits/temp) restricted to the top-k (same
    distribution as moshi `sample_top_k`). temp <= 0 means greedy argmax.

    MOSHI_TOPK_BISECT=N (default 0 = exact torch.topk): find the k-th largest value by N steps of bisection on the
    value range instead of a sort (count(l >= mid) is one compare + reduce per step); the kept set is the top-k up to
    ties within (max-min)/2^N."""
    if temp <= 0:
        return torch.argmax(logits, dim=-1)
    l = logits / temp
    if 0 < top_k < l.shape[-1]:
        if _TOPK_BISECT > 0:
            hi = l.amax(-1, keepdim=True)
            lo = l.amin(-1, keepdim=True)
            for _ in range(_TOPK_BISECT):
                mid = 0.5 * (lo + hi)
                ge = (l >= mid).sum(-1, keepdim=True)
                big = ge >= top_k
                lo = torch.where(big, mid, lo)
                hi = torch.where(big, hi, mid)
            kth = lo
        else:
            kth = torch.topk(l, top_k, dim=-1).values[..., -1:]
        l = torch.where(l >= kth, l, torch.full_like(l, NEG))
    return torch.argmax(l + gumbel, dim=-1)


_TOPK_BISECT = int(os.environ.get("MOSHI_TOPK_BISECT", "0"))


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
        # Interleaved RoPE == rotate-half RoPE after permuting each head's Q/K output rows from [r0,i0,r1,i1,...] to
        # [r0,r1,...,i0,i1,...]. q.k is invariant under the same permutation of q and k, so this is exact; V and
        # out_proj are untouched. MOSHI_ROPE=interleaved keeps the reference layout.
        self.rope_half = os.environ.get("MOSHI_ROPE", "half") == "half"
        perm = torch.cat([torch.arange(0, D, 2), torch.arange(1, D, 2)])
        hp = (torch.arange(H).view(H, 1) * D + perm.view(1, D)).reshape(-1)
        qkv_perm = torch.cat([hp, H * D + hp, 2 * H * D + torch.arange(H * D)])
        # MOSHI_KERNELS=1: nkilib fused decode (attention_block_tkg + mlp TKG) with a block KV ring; weights are
        # stored in the kernels' [in, out] layout and the per-layer KV is [B*NBLK, H, BLK, D].
        self.kernels = os.environ.get("MOSHI_KERNELS", "1") == "1" and not os.environ.get("VLLM_NEURON_CPU_MODE") == "1"
        if self.kernels:
            assert self.rope_half, "the fused kernel path needs MOSHI_ROPE=half"
            self.BLK = int(os.environ.get("MOSHI_KV_BLOCK", "32"))
            self.SCTX = int(os.environ.get("MOSHI_KV_SLOTS", "3072"))  # >= context + BLK, multiple of 128
            assert self.SCTX % self.BLK == 0 and self.SCTX % 128 == 0 and self.SCTX >= cfg.context + self.BLK
            self.NBLK = self.SCTX // self.BLK
            # MOSHI_FP8_MLP=1: Helium MLP weights stored FP8 e4m3 with per-output-row scales (weight-only; activations
            # bf16; dequant scale applied to the matmul output). Halves 8.9 of the 15.4 GB of weights.
            self.fp8_mlp = os.environ.get("MOSHI_FP8_MLP", "0") == "1" or os.environ.get("MOSHI_W8", "") == "int8"
            self.fp8_max = float(os.environ.get("MOSHI_FP8_WMAX", "240"))
            self.fp8_group = int(os.environ.get("MOSHI_FP8_GROUP", "0"))   # 0 = per-row scales
            self.fp8_down = os.environ.get("MOSHI_FP8_DOWN", "1") == "1"
            # MOSHI_W8=int8: int8 weight-only MLP with per-(row, 128-group) scales (4x lower error than fp8 e4m3)
            self.w8_int = os.environ.get("MOSHI_W8", "fp8") == "int8"
            if self.w8_int:
                self.fp8_mlp = True
                self.fp8_group = self.fp8_group or 128
            # nkilib mlp TKG measured slower than the compiler's matmuls here (B=3: 93 ms with column tiling,
            # 60 ms without, vs 50 ms with torch); default to the torch MLP. MOSHI_TORCH_MLP=0 -> NF.mlp.
            self.torch_mlp = os.environ.get("MOSHI_TORCH_MLP", "1") == "1"
            # MOSHI_FP8_KV=1: FP8 (e4m3) KV cache. Per-layer quant scales: scale = FP8_MAX / amax(K or V), amax from
            # MOSHI_KV_SCALES (json {"k": [32], "v": [32]} of bf16 amax per layer) or a default amax.
            self.fp8_kv = os.environ.get("MOSHI_FP8_KV", "0") == "1"
            self.attn_chunk = int(os.environ.get("MOSHI_ATTN_CHUNK", "4"))
            if self.fp8_kv:
                import json
                fmax = float(os.environ.get("MOSHI_FP8_MAX", "240"))
                path = os.environ.get("MOSHI_KV_SCALES")
                am = json.load(open(path)) if path else {"k": [float(os.environ.get("MOSHI_K_AMAX", "24"))] * L,
                                                         "v": [float(os.environ.get("MOSHI_V_AMAX", "8"))] * L}
                self.kv_scales = [(fmax / max(am["k"][i], 1e-3), fmax / max(am["v"][i], 1e-3)) for i in range(L)]
            self.mlp_mode = os.environ.get("MOSHI_MLP_MODE", "nf")  # nf | direct
            self.mlp_ct = os.environ.get("MOSHI_MLP_CT", "1") == "1"
            if self.mlp_mode == "direct":
                from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
                from nkilib.core.mlp.mlp import mlp as nk_mlp
                from nkilib.core.utils.common_types import ActFnType
                from nkilib.core.utils.common_types import ComputationMode
                self._wmlp = wrap_nki(nk_mlp)
                self._act = ActFnType.SiLU
                self._mode = getattr(ComputationMode, os.environ.get("MOSHI_MLP_CMODE", "DECODE"))
        for i in range(L):
            p = f"transformer.layers.{i}."
            w = g(p + "self_attn.in_proj_weight")
            if self.rope_half:
                w = w[qkv_perm]
            win, wout = g(p + "gating.linear_in.weight"), g(p + "gating.linear_out.weight")
            if self.kernels and self.fp8_mlp:
                I = win.shape[0] // 2
                G = self.fp8_group
                if self.w8_int:
                    quant = lambda t: _i8_group(t, G or 128)
                else:
                    quant = (lambda t: _fp8_group(t, self.fp8_max, G)) if G else (lambda t: _fp8_rowwise(t, self.fp8_max))
                gq, gs = quant(win[:I])
                uq, us = quant(win[I:])
                extra = dict(dq=P(quant(wout)[0]), ds=P(quant(wout)[1])) if self.fp8_down else dict(wd=P(wout))
                self.layers.append(nn.ParameterDict(dict(
                    n1=P(g(p + "norm1.alpha").view(1, -1)), n2=P(g(p + "norm2.alpha").view(1, -1)),
                    wqkv=P(w.t()), wo=P(g(p + "self_attn.out_proj.weight").t()),
                    gq=P(gq), gs=P(gs), uq=P(uq), us=P(us), **extra)))
                kv = torch.zeros(B * self.NBLK, H, self.BLK, D, dtype=dtype)
                self.register_buffer(f"k{i}", kv, persistent=False)
                self.register_buffer(f"v{i}", kv.clone(), persistent=False)
                continue
            if self.kernels:
                I = win.shape[0] // 2
                self.layers.append(nn.ParameterDict(dict(
                    n1=P(g(p + "norm1.alpha").view(1, -1)), n2=P(g(p + "norm2.alpha").view(1, -1)),
                    wqkv=P(w.t()), wo=P(g(p + "self_attn.out_proj.weight").t()),
                    wg=P(win[:I].t()), wu=P(win[I:].t()), wd=P(wout.t()))))
                kvdt = torch.float8_e4m3fn if self.fp8_kv else dtype
                self.register_buffer(f"k{i}", torch.zeros(B * self.NBLK, H, self.BLK, D, dtype=kvdt), persistent=False)
                self.register_buffer(f"v{i}", torch.zeros(B * self.NBLK, H, self.BLK, D, dtype=kvdt), persistent=False)
                if self.fp8_kv:
                    # kernel quantizes with x * scale and dequantizes by folding 1/scale into softmax scale and W_out
                    ks, vs = self.kv_scales[i]
                    self.register_buffer(f"ks{i}", torch.full((1, 1), ks, dtype=dtype), persistent=False)
                    self.register_buffer(f"vs{i}", torch.full((1, 1), vs, dtype=dtype), persistent=False)
                    self.layers[-1]["wo"] = P((self.layers[-1]["wo"].float() / vs).to(dtype))
                continue
            self.layers.append(nn.ParameterDict(dict(
                n1=P(g(p + "norm1.alpha").view(-1)), n2=P(g(p + "norm2.alpha").view(-1)),
                wqkv=P(w), wo=P(g(p + "self_attn.out_proj.weight")),
                win=P(win), wout=P(wout))))
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
        self.dep_in_fused = os.environ.get("MOSHI_DEP_IN_FUSED", "1") == "1"

    # ---------------------------------------------------------------- temporal transformer
    def temporal(self, tokens, reset, active, ctx: int):
        if self.kernels:
            return self.temporal_kernels(tokens, reset, active)
        cfg, B, H, D, C = self.cfg, self.B, self.cfg.num_heads, self.cfg.head_dim, self.C
        valid = (tokens >= 0).to(self.dtype).unsqueeze(-1)
        x = (F.embedding(tokens.clamp(min=0) + self.emb_off, self.emb) * valid).sum(1, keepdim=True)  # [B,1,dim]
        off = torch.where(reset, torch.zeros_like(self.offset), self.offset)
        cos, sin = rope_tables(off.view(B, 1), D, cfg.max_period)          # [B,1,1,D/2] fp32
        if self.rope_half:
            cos2 = torch.cat([cos, cos], -1)                                 # [B,1,1,D] fp32
            sin2 = torch.cat([-sin, sin], -1)
        pos_k = ring_positions(off, 1, C)[:, :ctx]
        delta = off.view(B, 1) - pos_k
        ok = (pos_k >= 0) & (delta >= 0) & (delta < cfg.context)
        mask = torch.where(ok, 0.0, NEG).to(torch.float32).view(B, 1, 1, ctx)
        abl = getattr(self, "_abl", "")  # profiling ablations only
        if "simplemask" in abl:
            mask = torch.where(torch.arange(ctx, device=x.device).view(1, ctx) <= off.view(B, 1), 0.0, NEG).float().view(B, 1, 1, ctx)
        widx = ((torch.arange(B, device=x.device).view(B, 1) * H + self.hoff.view(1, H)) * C
                + (off % C).view(B, 1)).reshape(-1)
        for i, l in enumerate(self.layers):
            h = rms_f32(x, l["n1"], cfg.eps)
            q, k, v = F.linear(h, l["wqkv"]).view(B, 1, 3, H, D).permute(2, 0, 3, 1, 4)  # [B,H,1,D]
            if "norope" in abl:
                pass
            elif self.rope_half:
                # x*cos + rotate_half(x)*sin, rotate_half(x) = [x2, x1] with the sign folded into sin2
                qf, kf = q.float(), k.float()
                q = (qf * cos2 + torch.cat([qf[..., D // 2:], qf[..., :D // 2]], -1) * sin2).to(q.dtype)
                k = (kf * cos2 + torch.cat([kf[..., D // 2:], kf[..., :D // 2]], -1) * sin2).to(k.dtype)
            else:
                q = apply_rope_interleaved(q, cos, sin)
                k = apply_rope_interleaved(k, cos, sin)
            kb, vb = getattr(self, f"k{i}"), getattr(self, f"v{i}")
            if getattr(self, "_calib", None) is not None:
                self._calib.append(torch.stack([k.float().abs().amax(), v.float().abs().amax()]))
            if not getattr(self, "_skip_write", False):  # profiling only
                kb.index_copy_(0, widx, k.reshape(-1, D))
                vb.index_copy_(0, widx, v.reshape(-1, D))
            K = kb.view(B, H, C, D)[:, :, :ctx]
            V = vb.view(B, H, C, D)[:, :, :ctx]
            if "noattn" in abl:
                o = q.transpose(1, 2).reshape(B, 1, H * D)
            else:
                sc = (q @ K.transpose(-1, -2)).float() * (D ** -0.5) + mask
                o = (torch.softmax(sc, -1).to(V.dtype) @ V).transpose(1, 2).reshape(B, 1, H * D)
            if "nomlp" in abl:
                x = x + o
                continue
            x = x + F.linear(o, l["wo"])
            h = rms_f32(x, l["n2"], cfg.eps)
            gu = F.linear(h, l["win"]).view(B, 1, 2, -1)
            x = x + F.linear(F.silu(gu[..., 0, :]) * gu[..., 1, :], l["wout"])
        self.offset.copy_(torch.where(active, off + 1, self.offset))
        tout = rms_f32(x, self.out_norm, cfg.eps)              # [B,1,dim]
        return tout, F.linear(tout, self.text_linear).float()[:, 0]  # [B,32000]

    def temporal_kernels(self, tokens, reset, active):
        """Helium step on nkilib fused decode kernels.

        Ring KV in blocks: frame f lives in slot f mod SCTX (block (f mod SCTX) // BLK, offset f % BLK) of the row's
        NBLK blocks. Each step the row's block table is rotated so that the attended window (the last context-1
        frames) is contiguous in table order, starting at offset `ofs` < BLK; the kernel's in-kernel banded mask
        (pos_ids = end, swa_start_pos_ids = start) then selects exactly moshi's RingKVCache window. The kernel places
        the current token at the table's last slot (never part of the window since SCTX >= context + BLK) and writes
        its K/V to the physical slot of frame t. Inactive rows write to uint32_max (skipped)."""
        from nkilib.core.utils.common_types import NormType
        from vllm_neuron.functional.attention.attention_decode import attention_decode
        from vllm_neuron.functional.mlp import mlp as nf_mlp

        cfg, B, H, D = self.cfg, self.B, self.cfg.num_heads, self.cfg.head_dim
        BLK, NBLK, SCTX, W = self.BLK, self.NBLK, self.SCTX, cfg.context - 1
        dev = tokens.device
        valid = (tokens >= 0).to(self.dtype).unsqueeze(-1)
        x = (F.embedding(tokens.clamp(min=0) + self.emb_off, self.emb) * valid).sum(1)  # [B,dim]
        t = torch.where(reset, torch.zeros_like(self.offset), self.offset)               # frames before this one
        s0 = torch.clamp(t - W, min=0)                                                  # oldest attended frame
        start_blk = (s0 % SCTX) // BLK
        ofs = s0 % BLK
        end = ofs + (t - s0)
        j = torch.arange(NBLK, device=dev).view(1, NBLK)
        rows = torch.arange(B, device=dev).view(B, 1)
        blocks = rows * NBLK + (start_blk.view(B, 1) + j) % NBLK                        # [B,NBLK] pool block ids
        hh = torch.arange(H, device=dev).view(1, H, 1)
        table = (blocks.view(B, 1, NBLK) * H + hh).to(torch.int32)                      # [B,H,NBLK] head-inner
        wslot = t % SCTX
        wblk = rows.view(B) * NBLK + wslot // BLK
        widx = ((wblk.view(B, 1) * H + hh.view(1, H)) * BLK + (wslot % BLK).view(B, 1))  # [B,H]
        # inactive rows: -1 in int32 == 0xFFFFFFFF as uint32 -> the kernel skips the write
        widx = torch.where(active.view(B, 1), widx.to(torch.int32), torch.full_like(widx, -1, dtype=torch.int32))
        widx = widx.view(B, H, 1).to(torch.uint32)
        pos_ids = end.view(B, 1).to(torch.float32)
        swa_start = ofs.view(B, 1).to(torch.float32)
        cos, sin = rope_tables(t.view(B, 1), D, cfg.max_period)                         # [B,1,1,D/2]
        cos = cos.view(B, 1, D // 2).permute(2, 0, 1).contiguous().to(self.dtype)       # [D/2,B,1]
        sin = sin.view(B, 1, D // 2).permute(2, 0, 1).contiguous().to(self.dtype)
        x = x.view(B, 1, -1)
        abl = getattr(self, "_abl", "")  # profiling ablations only
        for i, l in enumerate(self.layers):
            if "noattn" in abl:
                a = x
            else:
              # The kernel folds the 32 KV heads into its batch; its FP8-KV path is only correct while
              # B*32 <= 128 partitions (measured: B<=4 correct, B>=5 garbage). Chunk rows for FP8 KV; the KV pool
              # and the global block tables are shared, so per-chunk calls are exact.
              ch = self.attn_chunk if self.fp8_kv else B
              parts = []
              for r0 in range(0, B, ch):
                  r1 = min(B, r0 + ch)
                  parts.append(attention_decode(
                      X=x[r0:r1], X_hidden_dim_actual=cfg.dim, rmsnorm_X_enabled=True, rmsnorm_X_eps=cfg.eps,
                      rmsnorm_X_gamma=l["n1"], W_qkv=l["wqkv"], bias_qkv=None,
                      cos=cos[:, r0:r1].contiguous(), sin=sin[:, r0:r1].contiguous(), rope_contiguous_layout=True,
                      K_cache_transposed=False, active_blocks_table=table[r0:r1].contiguous(),
                      K_cache=getattr(self, f"k{i}"), V_cache=getattr(self, f"v{i}"), attention_mask=None,
                      softmax_scale=D ** -0.5 / (self.kv_scales[i][0] if self.fp8_kv else 1.0), sink=None,
                      pos_ids=pos_ids[r0:r1].contiguous(), swa_start_pos_ids=swa_start[r0:r1].contiguous(),
                      update_cache=True, kv_cache_update_idx=widx[r0:r1].contiguous(),
                      k_scale=getattr(self, f"ks{i}") if self.fp8_kv else None,
                      v_scale=getattr(self, f"vs{i}") if self.fp8_kv else None,
                      W_out=l["wo"], bias_out=None, transposed_out=False, out_in_sb=False))
              a = parts[0] if len(parts) == 1 else torch.cat([p.view(-1, cfg.dim) for p in parts], 0)
            if "nomlp" in abl:
                x = x + a.view(B, 1, -1)
                continue
            if self.mlp_mode == "direct":
                xr = x.view(B, -1) + a.view(B, -1)
                m = self._wmlp[2](
                    hidden_tensor=xr.view(1, B, -1), gate_proj_weights_tensor=l["wg"], up_proj_weights_tensor=l["wu"],
                    down_proj_weights_tensor=l["wd"], normalization_weights_tensor=l["n2"],
                    normalization_type=NormType.RMS_NORM, eps=cfg.eps, activation_fn=self._act,
                    use_tkg_gate_up_proj_column_tiling=self.mlp_ct, use_tkg_down_proj_column_tiling=self.mlp_ct,
                    mode=self._mode)
                m = m[0] if isinstance(m, (tuple, list)) else m
                x = (xr + m.view(B, -1)).view(B, 1, -1)
            elif self.fp8_mlp:
                x = x + a.view(B, 1, -1)
                h = rms_f32(x, l["n2"].view(-1), cfg.eps)
                G = self.fp8_group
                L8 = (lambda z, q, sc: lin8g(z, q, sc, self.dtype, G)) if G else (lambda z, q, sc: lin8(z, q, sc, self.dtype))
                gt = L8(h, l["gq"], l["gs"])
                ut = L8(h, l["uq"], l["us"])
                mid = F.silu(gt) * ut
                x = x + (L8(mid, l["dq"], l["ds"]) if self.fp8_down else F.linear(mid, l["wd"]))
            elif self.torch_mlp:
                x = x + a.view(B, 1, -1)
                h = rms_f32(x, l["n2"].view(-1), cfg.eps)
                x = x + F.linear(F.silu(F.linear(h, l["wg"].t())) * F.linear(h, l["wu"].t()), l["wd"].t())
            else:
                m, x = nf_mlp(a.view(B, -1), l["wg"], l["wu"], l["wd"], eps=cfg.eps, ln_w=l["n2"],
                              norm_type=NormType.RMS_NORM, fused_add_tensor=x.view(B, -1), store_fused_add_result=True)
                x = (x + m).view(B, 1, -1)
        self.offset.copy_(torch.where(active, t + 1, self.offset))
        tout = rms_f32(x, self.out_norm, cfg.eps)
        return tout, F.linear(tout, self.text_linear).float()[:, 0]

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
        if self.dep_in_fused:
            # all 8 per-step input projections read the same transformer output: one [8*1024, 4096] matmul
            din = F.linear(tout, self.dep_in.view(-1, self.dep_in.shape[-1])).view(B, cfg.dep_q, -1)
        for s in range(cfg.dep_q):
            x = ((din[:, s] if self.dep_in_fused else F.linear(tout, self.dep_in[s])) + prev).view(B, 1, -1)
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
