"""Streaming Mimi codec as two static per-frame graphs for B session rows.

  encode(pcm [B,1,1920], reset [B] bool, active [B] bool) -> codes [B,8]
  decode(codes [B,8],     reset [B] bool, active [B] bool) -> pcm  [B,1,1920]

This reproduces the streaming semantics of kyutai `moshi` (`StreamingConv1d`, `StreamingConvTranspose1d`,
`RingKVCache` with context 250, interleaved RoPE, `SplitResidualVectorQuantizer`) exactly. All streaming state
(conv left-context buffers, transposed-conv overlap-add tails, transformer ring-KV, per-row offsets) lives in
module buffers on the device and is updated in place, so the host only moves PCM and codes.

`reset[b]` zeroes row b's state before the frame, as `reset_streaming()` does. `active[b] == False` leaves row
b's state unchanged, as the reference `exec_mask` does. Ring-KV writes for inactive rows land in the slot the
row overwrites first on its next active step, so they are never read.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open

from .config import FRAME, MimiConfig

NEG = -1e30


def elu(x):
    return torch.where(x > 0, x, torch.exp(torch.clamp(x, max=0.0)) - 1.0)


def gelu_erf(x):
    # F.gelu graph-breaks under the lite backend with fullgraph; write the exact erf form by hand.
    return 0.5 * x * (1.0 + torch.erf(x * 0.7071067811865476))


def rope_tables(pos, head_dim, max_period):
    """pos [B,T] long -> (cos, sin) [B,1,T,D/2] fp32, matching moshi `apply_rope` (time after heads)."""
    ds = torch.arange(head_dim // 2, device=pos.device, dtype=torch.float32)
    freqs = torch.exp(ds * (-math.log(max_period) * 2 / head_dim))
    ts = pos.float().unsqueeze(-1) * freqs  # [B,T,D/2]
    return torch.cos(ts).unsqueeze(1), torch.sin(ts).unsqueeze(1)


def apply_rope_interleaved(x, cos, sin):
    """x [B,H,T,D] with [r0,i0,r1,i1,...] pairs; computed in fp32, returned in x.dtype."""
    B, H, T, D = x.shape
    xr = x.view(B, H, T, D // 2, 2).float()
    r, i = xr[..., 0], xr[..., 1]
    o = torch.stack([r * cos - i * sin, r * sin + i * cos], dim=-1)
    return o.view(B, H, T, D).to(x.dtype)


def ring_positions(end_offset, T, C):
    """moshi RingKVCache.complete positions. end_offset [B] (before this write). Returns pos_k [B,C] (-1 invalid)."""
    idx = torch.arange(C, device=end_offset.device, dtype=torch.long).view(1, C)
    last = end_offset.view(-1, 1) + T - 1
    delta = idx - last % C
    pos = torch.where(delta <= 0, last + delta, last + delta - C)
    return torch.where(idx >= end_offset.view(-1, 1) + T, torch.full_like(pos, -1), pos)


class _StreamingTransformer(nn.Module):
    """Mimi ProjectedTransformer (d=512, identity projections) with per-row ring KV of `context` slots."""

    def __init__(self, f, prefix, cfg: MimiConfig, B, dtype):
        super().__init__()
        self.B, self.H, self.L, self.C = B, cfg.tr_heads, cfg.tr_layers, cfg.tr_context
        self.D = cfg.dimension // cfg.tr_heads
        self.max_period = cfg.max_period
        self.dtype = dtype
        g = lambda k: f.get_tensor(prefix + k).to(dtype)
        P = lambda t: nn.Parameter(t.contiguous(), requires_grad=False)
        self.layers = nn.ModuleList()
        for i in range(self.L):
            p = f"transformer.layers.{i}."
            self.layers.append(nn.ParameterDict(dict(
                n1w=P(g(p + "norm1.weight")), n1b=P(g(p + "norm1.bias")),
                n2w=P(g(p + "norm2.weight")), n2b=P(g(p + "norm2.bias")),
                wqkv=P(g(p + "self_attn.in_proj_weight")), wo=P(g(p + "self_attn.out_proj.weight")),
                # LayerScale folded into the output projections (exact up to fp rounding).
                ls1=P(g(p + "layer_scale_1.scale")), ls2=P(g(p + "layer_scale_2.scale")),
                w1=P(g(p + "linear1.weight")), w2=P(g(p + "linear2.weight")))))
            self.register_buffer(f"k{i}", torch.zeros(B * self.H * self.C, self.D, dtype=dtype), persistent=False)
            self.register_buffer(f"v{i}", torch.zeros(B * self.H * self.C, self.D, dtype=dtype), persistent=False)
        self.register_buffer("offset", torch.zeros(B, dtype=torch.long), persistent=False)
        self.register_buffer("hoff", torch.arange(self.H), persistent=False)

    def forward(self, x, reset, active):
        """x [B,C,T] (conv layout) -> [B,C,T]."""
        B, H, C, D = self.B, self.H, self.C, self.D
        x = x.transpose(1, 2)  # [B,T,512]
        T = x.shape[1]
        off = torch.where(reset, torch.zeros_like(self.offset), self.offset)
        t = torch.arange(T, device=x.device)
        pos_q = off.view(B, 1) + t.view(1, T)  # [B,T]
        cos, sin = rope_tables(pos_q, D, self.max_period)
        pos_k = ring_positions(off, T, C)  # [B,C]
        delta = pos_q.view(B, T, 1) - pos_k.view(B, 1, C)
        ok = (pos_k.view(B, 1, C) >= 0) & (delta >= 0) & (delta < C)
        mask = torch.where(ok, 0.0, NEG).to(torch.float32).view(B, 1, T, C)
        widx = ((torch.arange(B, device=x.device).view(B, 1, 1) * H + self.hoff.view(1, H, 1)) * C
                + (pos_q % C).view(B, 1, T)).reshape(-1)
        for i, l in enumerate(self.layers):
            h = F.layer_norm(x, (x.shape[-1],), l["n1w"], l["n1b"], 1e-5)
            q, k, v = F.linear(h, l["wqkv"]).view(B, T, 3, H, D).permute(2, 0, 3, 1, 4)  # [B,H,T,D] each
            q = apply_rope_interleaved(q, cos, sin)
            k = apply_rope_interleaved(k, cos, sin)
            kb, vb = getattr(self, f"k{i}"), getattr(self, f"v{i}")
            kb.index_copy_(0, widx, k.reshape(-1, D))
            vb.index_copy_(0, widx, v.reshape(-1, D))
            K, V = kb.view(B, H, C, D), vb.view(B, H, C, D)
            sc = (q @ K.transpose(-1, -2)).float() * (D ** -0.5) + mask
            o = (torch.softmax(sc, -1).to(V.dtype) @ V).transpose(1, 2).reshape(B, T, H * D)
            x = x + l["ls1"] * F.linear(o, l["wo"])
            h = F.layer_norm(x, (x.shape[-1],), l["n2w"], l["n2b"], 1e-5)
            x = x + l["ls2"] * F.linear(gelu_erf(F.linear(h, l["w1"])), l["w2"])
        self.offset.copy_(torch.where(active, off + T, self.offset))
        return x.transpose(1, 2)


class _Conv(nn.Module):
    """moshi StreamingConv1d: left-context buffer of k_eff - stride samples; constant or replicate padding."""

    def __init__(self, w, b, B, stride=1, dilation=1, replicate=False):
        super().__init__()
        self.w = nn.Parameter(w.contiguous(), requires_grad=False)
        self.b = None if b is None else nn.Parameter(b.contiguous(), requires_grad=False)
        self.stride, self.dilation, self.replicate = stride, dilation, replicate
        k = (w.shape[-1] - 1) * dilation + 1
        self.tp = k - stride
        if self.tp:
            self.register_buffer("prev", torch.zeros(B, w.shape[1], self.tp, dtype=w.dtype), persistent=False)
        if replicate:
            self.register_buffer("first", torch.ones(B, dtype=torch.bool), persistent=False)

    def forward(self, x, reset, active):
        if not self.tp:
            return F.conv1d(x, self.w, self.b, stride=self.stride, dilation=self.dilation)
        r3, a3 = reset.view(-1, 1, 1), active.view(-1, 1, 1)
        prev = torch.where(r3, torch.zeros_like(self.prev), self.prev)
        if self.replicate:
            first = torch.where(reset, torch.ones_like(self.first), self.first)
            prev = torch.where(first.view(-1, 1, 1) & a3, x[..., :1].expand_as(prev), prev)
            self.first.copy_(torch.where(active, torch.zeros_like(first), first))
        xc = torch.cat([prev, x], dim=-1)
        y = F.conv1d(xc, self.w, self.b, stride=self.stride, dilation=self.dilation)
        self.prev.copy_(torch.where(a3, xc[..., -self.tp:], prev))
        return y


class _ConvTr(nn.Module):
    """moshi StreamingConvTranspose1d: overlap-add tail of K - S samples carried to the next call (bias removed)."""

    def __init__(self, w, b, B, stride, groups=1):
        super().__init__()
        self.w = nn.Parameter(w.contiguous(), requires_grad=False)
        self.b = None if b is None else nn.Parameter(b.contiguous(), requires_grad=False)
        self.stride, self.groups = stride, groups
        self.K = w.shape[-1]
        self.pt = self.K - stride
        cout = w.shape[1] * groups
        self.register_buffer("partial", torch.zeros(B, cout, self.pt, dtype=w.dtype), persistent=False)

    def forward(self, x, reset, active):
        r3, a3 = reset.view(-1, 1, 1), active.view(-1, 1, 1)
        if self.groups > 1 and x.shape[-1] == 1:
            # depthwise transposed conv on one input step: y[b,c,j] = x[b,c,0] * w[c,0,j]
            y = x * self.w[:, 0, :].unsqueeze(0)
            if self.b is not None:
                y = y + self.b.view(1, -1, 1)
        else:
            y = F.conv_transpose1d(x, self.w, self.b, stride=self.stride, groups=self.groups)
        partial = torch.where(r3, torch.zeros_like(self.partial), self.partial)
        pt = self.pt
        y = torch.cat([y[..., :pt] + partial, y[..., pt:]], dim=-1)
        tail = y[..., -pt:]
        if self.b is not None:
            tail = tail - self.b.view(1, -1, 1)
        self.partial.copy_(torch.where(a3, tail, partial))
        return y[..., :-pt]


class _ResBlock(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def forward(self, x, reset, active):
        return x + self.c2(elu(self.c1(elu(x), reset, active)), reset, active)


def _codebooks(f, prefix, n):
    out = []
    for i in range(n):
        emb = f.get_tensor(f"{prefix}.vq.layers.{i}._codebook.embedding_sum")
        use = f.get_tensor(f"{prefix}.vq.layers.{i}._codebook.cluster_usage").clamp(min=1e-5)
        out.append(emb / use[:, None])
    return torch.stack(out)  # [n,2048,256] fp32


class MimiEncoder(nn.Module):
    def __init__(self, path, B, cfg: MimiConfig = MimiConfig(), dtype=torch.float32):
        super().__init__()
        f = safe_open(path, framework="pt")
        g = lambda k: f.get_tensor(k).to(dtype)
        cv = lambda i, **kw: _Conv(g(f"encoder.model.{i}.conv.conv.weight"), g(f"encoder.model.{i}.conv.conv.bias"), B, **kw)
        res = lambda i: _ResBlock(
            _Conv(g(f"encoder.model.{i}.block.1.conv.conv.weight"), g(f"encoder.model.{i}.block.1.conv.conv.bias"), B),
            _Conv(g(f"encoder.model.{i}.block.3.conv.conv.weight"), g(f"encoder.model.{i}.block.3.conv.conv.bias"), B))
        ratios = list(reversed(cfg.ratios))  # 4,5,6,8
        mods = [cv(0)]
        idx = 1
        for r in ratios:
            mods.append(res(idx))
            mods.append(cv(idx + 2, stride=r))
            idx += 3
        self.seanet = nn.ModuleList(mods)
        self.last = cv(idx + 1)  # idx = 13 -> ELU at 13, conv at 14
        self.transformer = _StreamingTransformer(f, "encoder_transformer.", cfg, B, dtype)
        self.down = _Conv(g("downsample.conv.conv.conv.weight"), None, B, stride=2, replicate=True)
        nsem = cfg.n_q_semantic
        nac = cfg.num_codebooks - nsem
        self.register_buffer("in_sem", g("quantizer.rvq_first.input_proj.weight")[:, :, 0].float(), persistent=False)
        self.register_buffer("in_ac", g("quantizer.rvq_rest.input_proj.weight")[:, :, 0].float(), persistent=False)
        books = torch.cat([_codebooks(f, "quantizer.rvq_first", nsem), _codebooks(f, "quantizer.rvq_rest", nac)])
        self.register_buffer("books", books.float(), persistent=False)  # [8,2048,256]
        self.register_buffer("books_n2", (books.float() ** 2).sum(-1), persistent=False)  # [8,2048]
        self.register_buffer("bins", torch.arange(cfg.bins), persistent=False)
        self.nsem = nsem
        self.dtype = dtype

    def forward(self, pcm, reset, active):
        x = pcm.to(self.dtype)
        x = self.seanet[0](x, reset, active)
        for j in range(1, len(self.seanet), 2):
            x = self.seanet[j](x, reset, active)          # residual block
            x = self.seanet[j + 1](elu(x), reset, active)  # ELU + strided conv
        x = self.last(elu(x), reset, active)              # [B,512,2]
        x = self.transformer(x, reset, active)
        x = self.down(x, reset, active)                   # [B,512,1]
        emb = x[..., 0].float()                           # [B,512]
        codes = []
        for half, (proj, lo, hi) in enumerate(((self.in_sem, 0, self.nsem),
                                               (self.in_ac, self.nsem, self.books.shape[0]))):
            r = F.linear(emb, proj)                       # [B,256]
            for k in range(lo, hi):
                # argmin ||r - e||^2 == argmax(2 r.e - ||e||^2)
                score = 2.0 * (r @ self.books[k].t()) - self.books_n2[k]
                c = torch.argmax(score, dim=-1)           # [B]
                onehot = (self.bins.view(1, -1) == c.view(-1, 1)).to(r.dtype)
                r = r - onehot @ self.books[k]
                codes.append(c)
        return torch.stack(codes, dim=1)                  # [B,8]


class MimiDecoder(nn.Module):
    def __init__(self, path, B, cfg: MimiConfig = MimiConfig(), dtype=torch.float32):
        super().__init__()
        f = safe_open(path, framework="pt")
        g = lambda k: f.get_tensor(k).to(dtype)
        nsem = cfg.n_q_semantic
        nac = cfg.num_codebooks - nsem
        w_sem = f.get_tensor("quantizer.rvq_first.output_proj.weight")[:, :, 0]  # [512,256]
        w_ac = f.get_tensor("quantizer.rvq_rest.output_proj.weight")[:, :, 0]
        books = torch.cat([_codebooks(f, "quantizer.rvq_first", nsem) @ w_sem.t(),
                           _codebooks(f, "quantizer.rvq_rest", nac) @ w_ac.t()])  # [8,2048,512]
        self.nq, self.bins = books.shape[0], books.shape[1]
        self.books = nn.Parameter(books.reshape(-1, books.shape[-1]).to(dtype), requires_grad=False)
        self.register_buffer("book_off", (torch.arange(self.nq) * self.bins).view(1, -1), persistent=False)
        self.up = _ConvTr(g("upsample.convtr.convtr.convtr.weight"), None, B, stride=2, groups=cfg.dimension)
        self.transformer = _StreamingTransformer(f, "decoder_transformer.", cfg, B, dtype)
        cv = lambda i, **kw: _Conv(g(f"decoder.model.{i}.conv.conv.weight"), g(f"decoder.model.{i}.conv.conv.bias"), B, **kw)
        ct = lambda i, s: _ConvTr(g(f"decoder.model.{i}.convtr.convtr.weight"), g(f"decoder.model.{i}.convtr.convtr.bias"), B, s)
        res = lambda i: _ResBlock(
            _Conv(g(f"decoder.model.{i}.block.1.conv.conv.weight"), g(f"decoder.model.{i}.block.1.conv.conv.bias"), B),
            _Conv(g(f"decoder.model.{i}.block.3.conv.conv.weight"), g(f"decoder.model.{i}.block.3.conv.conv.bias"), B))
        self.first = cv(0)
        mods = []
        idx = 2
        for r in cfg.ratios:  # 8,6,5,4
            mods.append(ct(idx, r))
            mods.append(res(idx + 1))
            idx += 3
        self.seanet = nn.ModuleList(mods)
        self.last = cv(idx)  # 14
        self.dtype = dtype

    def forward(self, codes, reset, active):
        emb = F.embedding(codes + self.book_off, self.books).sum(1)  # [B,512]
        x = emb.unsqueeze(-1)                                        # [B,512,1]
        x = self.up(x, reset, active)                                # [B,512,2]
        x = self.transformer(x, reset, active)
        x = self.first(x, reset, active)                             # [B,1024,2]
        for j in range(0, len(self.seanet), 2):
            x = self.seanet[j](elu(x), reset, active)                # ELU + transposed conv
            x = self.seanet[j + 1](x, reset, active)                 # residual block
        return self.last(elu(x), reset, active)                      # [B,1,1920]


class MimiStreaming:
    """Host wrapper: encode / decode one frame for all B rows. `compile_fn` wraps each module (identity on CPU)."""

    def __init__(self, path, B, device, dtype=torch.float32, compile_fn=None, encoder=True, decoder=True):
        compile_fn = compile_fn or (lambda m: m)
        self.B, self.device = B, device
        self.enc = MimiEncoder(path, B, dtype=dtype).to(device).eval() if encoder else None
        self.dec = MimiDecoder(path, B, dtype=dtype).to(device).eval() if decoder else None
        self._enc = compile_fn(self.enc) if encoder else None
        self._dec = compile_fn(self.dec) if decoder else None
        self.frame = FRAME

    def _flags(self, reset, active):
        B, d = self.B, self.device
        r = torch.zeros(B, dtype=torch.bool) if reset is None else torch.as_tensor(reset, dtype=torch.bool)
        a = torch.ones(B, dtype=torch.bool) if active is None else torch.as_tensor(active, dtype=torch.bool)
        return r.to(d), a.to(d)

    @torch.no_grad()
    def encode(self, pcm, reset=None, active=None):
        """pcm [B,1920] fp32 (host or device) -> codes [B,8] (device)."""
        r, a = self._flags(reset, active)
        return self._enc(pcm.to(self.device).view(self.B, 1, -1), r, a)

    @torch.no_grad()
    def decode(self, codes, reset=None, active=None):
        """codes [B,8] long -> pcm [B,1920] (device)."""
        r, a = self._flags(reset, active)
        return self._dec(codes.to(self.device).long(), r, a).view(self.B, -1)
