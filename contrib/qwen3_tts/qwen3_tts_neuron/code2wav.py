"""Qwen3-TTS-Tokenizer-12Hz decoder ("Code2Wav") for AWS Neuron, PyTorch Native.

Faithful re-implementation of `Qwen3TTSTokenizerV2Decoder.forward` (qwen_tts 0.1.1) with static shapes:
  codes [1, 16, F] -> RVQ dequant -> causal conv -> 8L sliding-window(72) transformer -> 2x(convT x2 + ConvNeXt)
  -> conv -> 4 x (SnakeBeta, convT, 3 residual units) -> SnakeBeta -> conv -> clamp  => wav [1, 1, F*1920]

Chunked decode identical to the reference `chunked_decode(chunk_size, left_context_size)`: each call decodes
`ctx + chunk` frames and keeps the last `chunk*1920` samples. The graph is compiled once per (ctx+chunk) frame
count (we use a fixed window: left context 25 + chunk 25 = 50 frames, plus a final partial chunk padded).

NOTE: decode of a window is a pure function of its codes, so padding the *right* of a short final window with
dummy codes changes nothing for the kept samples only if the network is causal. Conv nets here are causal
(left-padded); ConvTranspose right-trims; attention is causal + sliding window -> fully causal. So right
padding is safe; we drop padded samples.
"""
from __future__ import annotations

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors import safe_open


SNAKE_FP32 = os.environ.get("C2W_SNAKE_FP32", "1") == "1"
# Fused NKI residual units (snake -> dilated conv -> snake -> 1x1 conv -> +x) for decoder blocks listed here
# (indices into the 4 upsample blocks; "" = off). Needs a Neuron device or NKI_SIMULATOR=1.
NKI_BLOCKS = {int(b) for b in os.environ.get("QTTS_C2W_NKI_BLOCKS", "").split(",") if b.strip()}


def _snake(x, a_exp, inv_b):
    # x [B,C,T]; a_exp = exp(alpha) [C,1]; inv_b = 1/(exp(beta)+1e-9) [C,1]
    # fp32 compute is ~1.5x FASTER on Neuron than bf16 here (measured: residual unit 3.9 -> 2.6 ms) and more accurate.
    if SNAKE_FP32 and x.dtype != torch.float32:
        xf = x.float()
        return (xf + inv_b.float() * torch.sin(xf * a_exp.float()).pow(2)).to(x.dtype)
    return x + inv_b * torch.sin(x * a_exp).pow(2)


class Code2Wav(nn.Module):
    def __init__(self, st_dir, dtype=torch.bfloat16):
        super().__init__()
        cfg = json.load(open(os.path.join(st_dir, "config.json")))["decoder_config"]
        self.cfg, self.dtype = cfg, dtype
        f = safe_open(os.path.join(st_dir, "model.safetensors"), framework="pt")
        g = lambda k: f.get_tensor("decoder." + k)
        P = lambda t: nn.Parameter(t.to(dtype).contiguous(), requires_grad=False)
        self.nq = cfg["num_quantizers"]
        self.upsample_total = int(math.prod(cfg["upsample_rates"] + cfg["upsampling_ratios"]))

        # ---- RVQ: codebook_dim/2 = 256-dim codebooks, output_proj 1x1 conv 256->512
        def cb(prefix, i):
            emb = g(f"quantizer.{prefix}.vq.layers.{i}._codebook.embedding_sum")
            use = g(f"quantizer.{prefix}.vq.layers.{i}._codebook.cluster_usage").clamp(min=1e-5)
            return emb / use[:, None]  # fp32
        first = torch.stack([cb("rvq_first", 0)])                       # [1,2048,256]
        rest = torch.stack([cb("rvq_rest", i) for i in range(self.nq - 1)])  # [15,2048,256]
        # fold output_proj (linear, no bias) into the codebooks: q = sum_k E_k[c_k] @ W^T
        w_first = g("quantizer.rvq_first.output_proj.weight")[:, :, 0]  # [512,256]
        w_rest = g("quantizer.rvq_rest.output_proj.weight")[:, :, 0]
        books = torch.cat([first @ w_first.t(), rest @ w_rest.t()], 0)  # [16,2048,512] fp32
        self.books = P(books.reshape(self.nq * books.shape[1], books.shape[2]))  # flat [16*2048, 512]
        self.book_size = books.shape[1]
        self.register_buffer("book_off", (torch.arange(self.nq) * self.book_size).view(1, self.nq, 1), persistent=False)

        # ---- pre_conv (causal, k=3)
        self.pre_conv_w, self.pre_conv_b = P(g("pre_conv.conv.weight")), P(g("pre_conv.conv.bias"))

        # ---- pre_transformer
        pt = "pre_transformer."
        self.in_w, self.in_b = P(g(pt + "input_proj.weight").t()), P(g(pt + "input_proj.bias"))
        self.out_w, self.out_b = P(g(pt + "output_proj.weight").t()), P(g(pt + "output_proj.bias"))
        self.fnorm = P(g(pt + "norm.weight"))
        self.L = cfg["num_hidden_layers"]
        self.nh, self.hd = cfg["num_attention_heads"], cfg["head_dim"]
        self.window = cfg["sliding_window"]
        self.eps = cfg["rms_norm_eps"]
        lay = []
        for i in range(self.L):
            p = f"{pt}layers.{i}."
            q, k, v = (g(p + f"self_attn.{n}_proj.weight") for n in "qkv")
            gate, up = g(p + "mlp.gate_proj.weight"), g(p + "mlp.up_proj.weight")
            lay.append(nn.ParameterDict(dict(
                qkv=P(torch.cat([q, k, v], 0).t()), o=P(g(p + "self_attn.o_proj.weight").t()),
                gu=P(torch.cat([gate, up], 0).t()), down=P(g(p + "mlp.down_proj.weight").t()),
                ln1=P(g(p + "input_layernorm.weight")), ln2=P(g(p + "post_attention_layernorm.weight")),
                s1=P(g(p + "self_attn_layer_scale.scale")), s2=P(g(p + "mlp_layer_scale.scale")))))
        self.layers = nn.ModuleList(lay)
        self.inter = cfg["intermediate_size"]
        self.theta = cfg["rope_theta"]

        # ---- upsample: per ratio: CausalTransConv(k=r,s=r) + ConvNeXt(dwconv7, LN, pw 4x, GELU, gamma)
        ups = []
        for i, r in enumerate(cfg["upsampling_ratios"]):
            p = f"upsample.{i}."
            ups.append(nn.ParameterDict(dict(
                tw=P(g(p + "0.conv.weight")), tb=P(g(p + "0.conv.bias")),
                dw=P(g(p + "1.dwconv.conv.weight")), db=P(g(p + "1.dwconv.conv.bias")),
                lnw=P(g(p + "1.norm.weight")), lnb=P(g(p + "1.norm.bias")),
                p1w=P(g(p + "1.pwconv1.weight").t()), p1b=P(g(p + "1.pwconv1.bias")),
                p2w=P(g(p + "1.pwconv2.weight").t()), p2b=P(g(p + "1.pwconv2.bias")),
                gamma=P(g(p + "1.gamma")))))
        self.ups = nn.ModuleList(ups)
        self.up_ratios = list(cfg["upsampling_ratios"])

        # ---- decoder conv stack
        def snake_params(prefix):
            a = g(prefix + ".alpha").exp().view(-1, 1)
            b = (1.0 / (g(prefix + ".beta").exp() + 1e-9)).view(-1, 1)
            return P(a), P(b)
        self.d0_w, self.d0_b = P(g("decoder.0.conv.weight")), P(g("decoder.0.conv.bias"))
        blocks = []
        for bi, r in enumerate(cfg["upsample_rates"]):
            p = f"decoder.{bi + 1}.block."
            a, b = snake_params(p + "0")
            blk = dict(a=a, b=b, tw=P(g(p + "1.conv.weight")), tb=P(g(p + "1.conv.bias")))
            for ri, dil in enumerate((1, 3, 9)):
                q = f"{p}{ri + 2}."
                a1, b1 = snake_params(q + "act1"); a2, b2 = snake_params(q + "act2")
                blk.update({f"r{ri}a1": a1, f"r{ri}b1": b1, f"r{ri}a2": a2, f"r{ri}b2": b2,
                            f"r{ri}w1": P(g(q + "conv1.conv.weight")), f"r{ri}c1": P(g(q + "conv1.conv.bias")),
                            f"r{ri}w2": P(g(q + "conv2.conv.weight")), f"r{ri}c2": P(g(q + "conv2.conv.bias"))})
            blocks.append(nn.ParameterDict(blk))
        self.blocks = nn.ModuleList(blocks)
        self.rates = list(cfg["upsample_rates"])
        nb = len(self.rates)
        self.fa, self.fb = snake_params(f"decoder.{nb + 1}")
        self.fw, self.fcb = P(g(f"decoder.{nb + 2}.conv.weight")), P(g(f"decoder.{nb + 2}.conv.bias"))

    nki_blocks: frozenset = frozenset()
    _nki_head = False
    _nki_out = False

    def prepare_nki(self, blocks=None):
        """Route the residual units of `blocks` (default QTTS_C2W_NKI_BLOCKS) through the NKI kernel. Call after
        .to(device); parameters are packed once into device buffers."""
        from .kernels.vocoder_nki import resunit_params
        self.nki_blocks = frozenset(NKI_BLOCKS if blocks is None else blocks)
        self._nki_p = {}
        for bi in self.nki_blocks:
            for ri in range(3):
                p = resunit_params(self.blocks[bi], ri)
                dev = self.blocks[bi][f"r{ri}w1"].device
                for k, v in p.items():
                    buf = v.to(dev)
                    self.register_buffer(f"_nki_{bi}_{ri}_{k}", buf, persistent=False)
                self._nki_p[(bi, ri)] = list(p)
        from libtorch_neuronx_lite.nki.nki_hop import wrap_nki
        from .kernels.vocoder_nki import ctconv_params, out_params, snake_conv_out, snake_ctconv, snake_resunit
        self._nki_k = wrap_nki(snake_resunit)
        from .kernels.vocoder_nki import SNAKE_IMPL
        self._snake_impl = SNAKE_IMPL
        dev = self.fw.device
        # block heads (snake + transposed conv) of the NKI blocks, and the output conv
        self._nki_head = os.environ.get("QTTS_C2W_NKI_HEAD", "1") == "1" and bool(self.nki_blocks)
        if self._nki_head:
            self._nki_k_head = wrap_nki(snake_ctconv)
            for bi in self.nki_blocks:
                for k, v in ctconv_params(self.blocks[bi]).items():
                    self.register_buffer(f"_nkih_{bi}_{k}", v.to(dev), persistent=False)
        self._nki_out = os.environ.get("QTTS_C2W_NKI_OUT", "1") == "1" and 3 in self.nki_blocks
        if self._nki_out:
            self._nki_k_out = wrap_nki(snake_conv_out)
            for k, v in out_params(self.fw, self.fcb, self.fa, self.fb).items():
                self.register_buffer(f"_nkio_{k}", v.to(dev), persistent=False)
        # grid must equal the compile LNC (NCC_EVRF066 otherwise); the kernel itself does not shard across programs
        self._nki_grid = int(os.environ.get("QTTS_C2W_NKI_GRID", os.environ.get("NEURON_LOGICAL_NC_CONFIG", "2")))

    def _nki_unit(self, x, bi, ri, dil):
        names = self._nki_p[(bi, ri)]
        args = [getattr(self, f"_nki_{bi}_{ri}_{k}") for k in names]
        return self._nki_k[self._nki_grid](x.to(torch.bfloat16).contiguous(), *args, dil,
                                            int(os.environ.get("QTTS_C2W_NKI_TILE", "512")), self._snake_impl).to(x.dtype)

    # ------------------------------------------------------------------ building blocks
    @staticmethod
    def _cconv(x, w, b, dilation=1, groups=1):
        # causal conv1d, stride 1: left pad (k-1)*dil (reference: padding = k_eff - stride; extra_padding = 0 for s=1)
        k = w.shape[-1]
        pad = (k - 1) * dilation
        return F.conv1d(F.pad(x, (pad, 0)), w, b, dilation=dilation, groups=groups)

    @staticmethod
    def _ctconv(x, w, b, stride):
        # causal transposed conv: full output then trim right by (k - stride)
        k = w.shape[-1]
        y = F.conv_transpose1d(x, w, b, stride=stride)
        trim = k - stride
        return y[..., : y.shape[-1] - trim] if trim > 0 else y

    def _rope(self, T, device):
        inv = 1.0 / (self.theta ** (torch.arange(0, self.hd, 2, device=device, dtype=torch.float32) / self.hd))
        t = torch.arange(T, device=device, dtype=torch.float32)
        fr = torch.outer(t, inv)
        e = torch.cat([fr, fr], -1)
        return e.cos().to(self.dtype), e.sin().to(self.dtype)

    def _transformer(self, x):
        # x [1,T,1024] -> [1,T,1024]
        B, T, _ = x.shape
        h = x @ self.in_w + self.in_b
        cos, sin = self._rope(T, x.device)
        cos, sin = cos.view(1, 1, T, self.hd), sin.view(1, 1, T, self.hd)
        i = torch.arange(T, device=x.device)
        d = i.view(-1, 1) - i.view(1, -1)
        mask = torch.where((d >= 0) & (d < self.window), 0.0, -1e30).to(torch.float32)
        hs = self.nh * self.hd
        for L in self.layers:
            r = h
            hn = torch.rms_norm(h, (h.shape[-1],), L["ln1"], self.eps)
            q, k, v = (hn @ L["qkv"]).split([hs, hs, hs], -1)
            q = q.view(B, T, self.nh, self.hd).transpose(1, 2)
            k = k.view(B, T, self.nh, self.hd).transpose(1, 2)
            v = v.view(B, T, self.nh, self.hd).transpose(1, 2)
            rh = lambda z: torch.cat((-z[..., self.hd // 2:], z[..., : self.hd // 2]), -1)
            q = q * cos + rh(q) * sin
            k = k * cos + rh(k) * sin
            sc = (q @ k.transpose(-1, -2)).float() * (self.hd ** -0.5) + mask
            p = torch.softmax(sc, -1).to(q.dtype)
            o = (p @ v).transpose(1, 2).reshape(B, T, hs) @ L["o"]
            h = r + L["s1"] * o
            r = h
            hn = torch.rms_norm(h, (h.shape[-1],), L["ln2"], self.eps)
            gu = hn @ L["gu"]
            gg, uu = gu.split([self.inter, self.inter], -1)
            h = r + L["s2"] * ((F.silu(gg) * uu) @ L["down"])
        h = torch.rms_norm(h, (h.shape[-1],), self.fnorm, self.eps)
        return h @ self.out_w + self.out_b

    def forward(self, codes, crop_ctx: int = 0, m1: int = 12, m2: int = 2):
        """codes [B,16,F] int64 -> wav [B, F*1920] (clamped).

        crop_ctx>0 (static int): the first crop_ctx frames are left context only. The conv stack after the
        transformer has a receptive field of ~11 frames (upsample/conv0/block0) and <1 frame after block0, so we
        drop context frames older than m1 before the upsampler and older than m2 after block0. Output is then
        wav for frames [crop_ctx, F) only. Exact as long as m1/m2 cover the receptive field (verified on CPU)."""
        Bt = codes.shape[0]
        idx = (codes + self.book_off).reshape(-1)                       # [B*16*F]
        e = F.embedding(idx, self.books).view(Bt, self.nq, -1, self.books.shape[-1])  # [B,16,F,512]
        x = e.sum(1).transpose(1, 2)                                     # [B,512,F]
        x = self._cconv(x, self.pre_conv_w, self.pre_conv_b)             # [1,1024,F]
        x = self._transformer(x.transpose(1, 2)).transpose(1, 2)         # [1,1024,F]
        a = max(0, crop_ctx - m1) if crop_ctx else 0
        if a:
            x = x[..., a:]
        for U, r in zip(self.ups, self.up_ratios):
            x = self._ctconv(x, U["tw"], U["tb"], r)
            res = x
            y = self._cconv(x, U["dw"], U["db"], groups=x.shape[1]).transpose(1, 2)
            y = F.layer_norm(y, (y.shape[-1],), U["lnw"], U["lnb"], 1e-6)
            y = F.gelu(y @ U["p1w"] + U["p1b"]) @ U["p2w"] + U["p2b"]
            x = res + (U["gamma"] * y).transpose(1, 2)
        x = self._cconv(x, self.d0_w, self.d0_b)
        b = max(a, crop_ctx - m2) if crop_ctx else 0
        per = int(math.prod(self.up_ratios))  # samples per frame after the upsampler (4)
        for bi, (Bk, r) in enumerate(zip(self.blocks, self.rates)):
            if bi == 1 and b > a:
                x = x[..., (b - a) * per * self.rates[0]:]
            if bi in self.nki_blocks and self._nki_head:
                x = self._nki_k_head[self._nki_grid](x.to(torch.bfloat16).contiguous(),
                                                     *[getattr(self, f"_nkih_{bi}_{k}") for k in ("wt", "bias", "a", "b")],
                                                     r, int(os.environ.get("QTTS_C2W_NKI_TILEQ", "256")), self._snake_impl)
            else:
                x = _snake(x, Bk["a"], Bk["b"])
                x = self._ctconv(x, Bk["tw"], Bk["tb"], r)
            for ri, dil in enumerate((1, 3, 9)):
                if bi in self.nki_blocks:
                    x = self._nki_unit(x, bi, ri, dil)
                    continue
                res = x
                y = _snake(x, Bk[f"r{ri}a1"], Bk[f"r{ri}b1"])
                y = self._cconv(y, Bk[f"r{ri}w1"], Bk[f"r{ri}c1"], dilation=dil)
                y = _snake(y, Bk[f"r{ri}a2"], Bk[f"r{ri}b2"])
                y = self._cconv(y, Bk[f"r{ri}w2"], Bk[f"r{ri}c2"])
                x = res + y
        if self._nki_out:
            x = self._nki_k_out[self._nki_grid](x.to(torch.bfloat16).contiguous(),
                                                *[getattr(self, f"_nkio_{k}") for k in ("wt", "bias", "a", "b")], 512,
                                                self._snake_impl)
        else:
            x = _snake(x, self.fa, self.fb)
            x = self._cconv(x, self.fw, self.fcb)
        if crop_ctx:
            x = x[..., (crop_ctx - b) * self.upsample_total:]
        return x.clamp(-1, 1).view(Bt, -1)


class _Crop(nn.Module):
    def __init__(self, m, ctx):
        super().__init__(); self.m, self.ctx = m, ctx

    def forward(self, codes):
        return self.m(codes, crop_ctx=self.ctx)


class _Plain(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, codes):
        return self.m(codes)


class Code2WavRunner:
    """Chunked decode with a fixed compiled window (left_ctx + chunk frames)."""

    def __init__(self, st_dir, device="neuron", dtype=torch.bfloat16, chunk=25, left_ctx=25, compile=True, model=None,
                 crop=True, small=8):
        self.m = (model or Code2Wav(st_dir, dtype)).to(device).eval()
        if NKI_BLOCKS:
            self.m.prepare_nki()
        self.dev, self.chunk, self.ctx = device, chunk, left_ctx
        self.up = self.m.upsample_total
        self.crop = crop
        self.fn = torch.compile(self.m, backend="neuron", dynamic=False) if compile else self.m
        self.fn_crop = torch.compile(_Crop(self.m, left_ctx), backend="neuron", dynamic=False) if compile else _Crop(self.m, left_ctx)
        # small window graph for short windows (first audio chunk / short tails): avoids paying a full 50-frame decode
        self.small = small
        self.fn_small = (torch.compile(_Plain(self.m), backend="neuron", dynamic=False) if compile else self.m) if small else None

    @torch.no_grad()
    def decode_window(self, codes_win, ctx_frames=None):
        """codes_win [W,16] (W <= ctx+chunk) -> wav for the window (padded internally to ctx+chunk).
        If ctx_frames == self.ctx and cropping is on, returns ONLY the new frames' audio (cheaper graph)."""
        W = self.ctx + self.chunk
        n = codes_win.shape[0]
        if self.small and n <= self.small:
            c = torch.zeros(1, 16, self.small, dtype=torch.long)
            c[0, :, :n] = codes_win.t()
            return self.fn_small(c.to(self.dev))[:, : n * self.up]
        c = torch.zeros(1, 16, W, dtype=torch.long)
        c[0, :, :n] = codes_win.t()
        if self.crop and ctx_frames == self.ctx:
            return self.fn_crop(c.to(self.dev))[:, : (n - self.ctx) * self.up]
        return self.fn(c.to(self.dev))[:, : n * self.up]

    @torch.no_grad()
    def decode_crop_batch(self, wins, buckets=(1, 2, 4), max_inflight=16):
        """Several full-context windows (each [ctx + n_i, 16], n_i <= chunk) in batched crop-graph calls.
        Returns one wav per window (only the new frames' audio). Rows are independent, so the result per window is
        the same function of its codes as decode_window(win, ctx_frames=ctx)."""
        W, out, i = self.ctx + self.chunk, [None] * len(wins), 0
        bmax = max(buckets)
        while i < len(wins):
            group = list(range(i, min(i + bmax, len(wins))))
            Bb = next(b for b in buckets if b >= len(group))
            c = torch.zeros(Bb, 16, W, dtype=torch.long)
            for r, j in enumerate(group):
                c[r, :, : wins[j].shape[0]] = wins[j].t()
            y = self.fn_crop(c.to(self.dev))
            for r, j in enumerate(group):
                out[j] = y[r : r + 1, : (wins[j].shape[0] - self.ctx) * self.up]
            i += len(group)
            calls = getattr(self, "_inflight", 0) + 1
            if calls >= max_inflight:  # bound queued device executions (runtime queue holds <= 63)
                y.cpu()
                calls = 0
            self._inflight = calls
        return out

    @torch.no_grad()
    def decode(self, codes):
        """codes [F,16] -> wav [F*1920] float32 cpu. Same windows as reference chunked_decode(chunk, ctx)."""
        outs, s, Fn = [], 0, codes.shape[0]
        while s < Fn:
            e = min(s + self.chunk, Fn)
            c = self.ctx if s - self.ctx > 0 else s
            w = self.decode_window(codes[s - c: e], ctx_frames=c)
            outs.append(w if (self.crop and c == self.ctx) else w[:, c * self.up:])
            s = e
        return torch.cat([o.cpu() for o in outs], -1).float().view(-1)
