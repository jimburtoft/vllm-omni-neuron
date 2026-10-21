"""Optimization cycle 1: ONE fused NEFF per frame for B streams.

frame(h, logits, seen, allow_eos, pad_embed, pos) ->
    on-device CB0 sampling (rep-penalty 1.05, suppress mask, min-new EOS block, greedy argmax)
    -> codec embed -> Code Predictor 15 steps (elementwise attention: S<=17 so batched tiny matmuls are replaced
       by VectorE multiply-reduce) -> Talker decode (28L, batched KV) -> (h', logits', seen', codes[B,16])
Host only reads `codes` (64 B per row) for EOS / Code2Wav; h/logits/seen never leave the device.
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .batched import BatchedTalker
from .engine import TTSEngine, rms, rotate_half


def _cp_layer_vec(l, x, cos, sin, kc, vc, pos, mask):
    """Code-Predictor layer with elementwise attention. x [B,T,H]; kc/vc [B,nkv,S,D]; mask [T,S]."""
    B, T, _ = x.shape
    hn = rms(x, l.ln1, l.eps)
    q, k, v = (hn @ l.w_qkv).split([l.q_sz, l.kv_sz, l.kv_sz], dim=-1)
    q = rms(q.view(B, T, l.nh, l.d), l.qn, l.eps).transpose(1, 2)
    k = rms(k.view(B, T, l.nkv, l.d), l.kn, l.eps).transpose(1, 2)
    v = v.view(B, T, l.nkv, l.d).transpose(1, 2)
    c, s = cos.view(1, 1, T, l.d), sin.view(1, 1, T, l.d)
    q = q * c + rotate_half(q) * s
    k = k * c + rotate_half(k) * s
    kc = kc.index_copy(2, pos, k)
    vc = vc.index_copy(2, pos, v)
    S = kc.shape[2]
    qg = q.view(B, l.nkv, l.rep, T, 1, l.d)
    sc = (qg * kc.view(B, l.nkv, 1, 1, S, l.d)).float().sum(-1) * l.scale  # [B,nkv,rep,T,S]
    p = torch.softmax(sc + mask.view(1, 1, 1, T, S), -1).to(x.dtype)
    o = (p.unsqueeze(-1) * vc.view(B, l.nkv, 1, 1, S, l.d)).sum(-2)  # [B,nkv,rep,T,D]
    o = o.view(B, l.nh, T, l.d).transpose(1, 2).reshape(B, T, l.nh * l.d)
    h = x + o @ l.w_o
    return h + l._mlp(h), kc, vc


def sample_topk(logits, gumbel, temperature, top_k):
    """Gumbel-max sampling from softmax(logits/T) restricted to top-k. logits [..., V] f32; gumbel same shape
    (host-provided -log(-log(U))). temperature<=0 -> greedy (gumbel ignored). Static python ints/floats only."""
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    l = logits / temperature
    if top_k and top_k > 0 and top_k < l.shape[-1]:
        kth = torch.topk(l, top_k, dim=-1).values[..., -1:]
        l = torch.where(l >= kth, l, float("-inf"))
    return torch.argmax(l + gumbel, dim=-1)


def cp_frame_vec(cp, talker_hidden, cb0_embed, vec=True, gumbel=None, temperature=0.0, top_k=0):
    cfg = cp.cfg
    B, dev, L = talker_hidden.shape[0], talker_hidden.device, len(cp.layers)
    kc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=dev) for _ in range(L)]
    vc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=dev) for _ in range(L)]
    j = torch.arange(cp.S, device=dev)

    def run(x, pos):
        cos, sin = cp.cos.index_select(0, pos), cp.sin.index_select(0, pos)
        m = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32)
        h = x
        for i, l in enumerate(cp.layers):
            if vec:
                h, kc[i], vc[i] = _cp_layer_vec(l, h, cos, sin, kc[i], vc[i], pos, m)
            else:
                h, kc[i], vc[i] = l.forward_local(h, cos, sin, kc[i], vc[i], pos, m)
        return rms(h[:, -1:, :], cp.norm, cfg.eps)

    h = run(torch.cat([talker_hidden, cb0_embed], 1) @ cp.proj_w + cp.proj_b, torch.arange(2, device=dev))
    codes, emb_sum = [], torch.zeros_like(talker_hidden)
    for g in range(cfg.groups - 1):
        lg = (h @ cp.heads[g]).float()  # [B,1,V]
        c = sample_topk(lg, gumbel[:, g] if gumbel is not None else None, temperature, top_k)  # [B,1]
        codes.append(c)
        e = F.embedding(c, cp.embeds[g])
        emb_sum = emb_sum + e
        if g < cfg.groups - 2:
            h = run(e @ cp.proj_w + cp.proj_b, torch.full((1,), g + 2, device=dev, dtype=torch.long))
    return torch.cat(codes, -1), emb_sum


class FusedFrame(nn.Module):
    def __init__(self, eng: TTSEngine, bt: BatchedTalker, vec_cp=True, rep=1.05, temperature=0.0, top_k=0,
                 sub_temperature=0.0, sub_top_k=0):
        super().__init__()
        self.eng_t, self.cp, self.bt = eng.talker, eng.cp, bt
        self.vec, self.rep = vec_cp, rep
        self.temperature, self.top_k = float(temperature), int(top_k)
        self.sub_temperature, self.sub_top_k = float(sub_temperature), int(sub_top_k)
        V = eng.talker.cfg.vocab
        ban = torch.zeros(V); ban[V - 1024:] = float("-inf"); ban[eng.eos] = 0.0
        self.register_buffer("ban", ban, persistent=False)
        eosm = torch.zeros(V, dtype=torch.bool); eosm[eng.eos] = True
        self.register_buffer("eos_mask", eosm, persistent=False)
        self.register_buffer("codec_emb", eng.codec_emb_dev, persistent=False)
        self.register_buffer("vocab_ids", torch.arange(V), persistent=False)

    def forward(self, h, logits, seen, allow_eos, pad_embed, pos, gumbel0=None, gumbel_sub=None):
        # h [B,1,H] bf16; logits [B,V] f32; seen [B,V] bool; allow_eos [B] bool; pad [B,1,H]; pos [B] long
        # gumbel0 [B,V] / gumbel_sub [B,15,Vcp] f32: host-drawn Gumbel noise (only used when sampling is enabled)
        l = torch.where(seen, torch.where(logits > 0, logits / self.rep, logits * self.rep), logits)
        l = l + self.ban
        l = torch.where(self.eos_mask.view(1, -1) & ~allow_eos.view(-1, 1), float("-inf"), l)
        cb0 = sample_topk(l, gumbel0, self.temperature, self.top_k)  # [B]
        seen = seen | (self.vocab_ids.view(1, -1) == cb0.view(-1, 1))
        e0 = F.embedding(cb0.view(-1, 1), self.codec_emb)  # [B,1,H]
        res, es = cp_frame_vec(self.cp, h, e0, self.vec, gumbel_sub, self.sub_temperature, self.sub_top_k)
        h2, lg2 = self.bt.decode(e0 + es + pad_embed, pos)
        return h2, lg2, seen, torch.cat([cb0.view(-1, 1), res], -1)


class FusedEngine:
    def __init__(self, eng: TTSEngine, B: int, S: int = 512, vec_cp=True, compile=True, sampling=None, seed=0):
        """sampling: None (greedy) or dict(temperature, top_k, sub_temperature, sub_top_k)."""
        from .batched import _M
        self.e, self.B, self.S = eng, B, S
        self.bt = BatchedTalker(eng.talker, B, S).to(eng.device)
        self.sampling = sampling
        self.gen = torch.Generator().manual_seed(seed)
        self.ff = FusedFrame(eng, self.bt, vec_cp, **(sampling or {})).to(eng.device)
        pre = _M(self.bt.prefill, self.bt)
        c = (lambda m: torch.compile(m, backend="neuron", dynamic=False)) if compile else (lambda m: m)
        self.prefill, self.frame = c(pre), c(self.ff)
        self.H = eng.talker.cfg.hidden

    def noise(self):
        """Host-drawn Gumbel noise for on-device sampling (None in greedy mode -> graph has no noise inputs)."""
        if not self.sampling:
            return None, None
        V, Vc = self.e.talker.cfg.vocab, self.e.cp.cfg.vocab
        g = lambda *shape: -torch.log(-torch.log(torch.rand(*shape, generator=self.gen).clamp_(1e-10, 1.0 - 1e-7)))
        return g(self.B, V).to(self.e.device), g(self.B, 15, 1, Vc).to(self.e.device)

    @torch.no_grad()
    def run(self, requests, max_new=600, c2w=None, first=1, chunk=25, ctx=25):
        e, B, S, dev, dt = self.e, self.B, self.S, self.e.device, self.e.dtype
        V = e.talker.cfg.vocab
        queue = list(range(len(requests)))
        slot = [-1] * B
        steps = [0] * B
        frames = [[] for _ in requests]
        out = [dict() for _ in requests]
        dec_upto = [0] * len(requests)
        h = torch.zeros(B, 1, self.H, dtype=dt, device=dev)
        logits = torch.zeros(B, V, device=dev)
        seen = torch.zeros(B, V, dtype=torch.bool, device=dev)
        pad = torch.zeros(B, 1, self.H, dtype=dt, device=dev)
        pos_h = torch.zeros(B, dtype=torch.long)
        t0 = time.perf_counter()

        def admit(b):
            nonlocal h, logits, seen, pad
            if not queue:
                slot[b] = -1
                return
            r = queue.pop(0)
            emb, pe = requests[r]
            T = emb.shape[1]
            Tb = next(x for x in e.prefill_buckets if x >= T)
            np_ = Tb - T
            x = torch.zeros(1, Tb, self.H, dtype=dt); x[:, np_:] = emb.to(dt)
            p = torch.cat([torch.arange(S - np_, S), torch.arange(T)]).long()
            bi = torch.tensor([b], device=dev)
            hl, lg = self.prefill(x.to(dev), bi, p.to(dev))
            h = h.index_copy(0, bi, hl)
            logits = logits.index_copy(0, bi, lg)
            seen = seen.index_copy(0, bi, torch.zeros(1, V, dtype=torch.bool, device=dev))
            pad = pad.index_copy(0, bi, pe.to(dt).to(dev))
            pos_h[b] = T
            steps[b] = 0
            slot[b] = r
            out[r]["t_admit"] = time.perf_counter() - t0

        for b in range(B):
            admit(b)
        nsteps = 0
        while any(s >= 0 for s in slot):
            allow = torch.tensor([steps[b] >= 2 for b in range(B)])
            g0, gs = self.noise()
            h, logits, seen, codes = self.frame(h, logits, seen, allow.to(dev), pad, pos_h.to(dev), g0, gs)
            cc = codes.cpu()
            nsteps += 1
            refill = []
            for b in range(B):
                r = slot[b]
                if r < 0:
                    continue
                c = cc[b]
                if int(c[0]) == e.eos or steps[b] >= max_new:
                    out[r]["codes"] = torch.stack(frames[r]) if frames[r] else torch.zeros(0, 16, dtype=torch.long)
                    out[r]["done_s"] = time.perf_counter() - t0
                    refill.append(b)
                    continue
                frames[r].append(c)
                steps[b] += 1
                pos_h[b] += 1
                if c2w is not None:
                    n = len(frames[r])
                    if dec_upto[r] == 0 and n >= first:
                        c2w.decode_window(torch.stack(frames[r][:first]))
                        dec_upto[r] = first
                    elif dec_upto[r] > 0 and n >= dec_upto[r] + chunk:
                        k = min(ctx, dec_upto[r])
                        c2w.decode_window(torch.stack(frames[r][dec_upto[r] - k: dec_upto[r] + chunk]), ctx_frames=k)
                        dec_upto[r] += chunk
                if "ttfa_s" not in out[r] and len(frames[r]) >= first:
                    out[r]["ttfa_s"] = time.perf_counter() - t0 - out[r]["t_admit"]
            for b in refill:
                admit(b)
        if c2w is not None:
            for r, o in enumerate(out):
                n = len(o.get("codes", []))
                if n > dec_upto[r]:
                    k = min(ctx, dec_upto[r])
                    c2w.decode_window(o["codes"][dec_upto[r] - k:], ctx_frames=k)
        return out, time.perf_counter() - t0, nsteps


class FusedFrameAsync(FusedFrame):
    """Same as FusedFrame but step counter and positions live on device, so the host never has to
    feed per-step tensors: the next frame can be issued before the previous codes are read."""

    def forward(self, h, logits, seen, step, pad_embed, pos):
        allow = step >= 2
        h2, lg2, seen2, codes = super().forward(h, logits, seen, allow, pad_embed, pos)
        return h2, lg2, seen2, step + 1, pos + 1, codes


class FusedEngineAsync(FusedEngine):
    def __init__(self, eng, B, S=512, vec_cp=True, compile=True):
        super().__init__(eng, B, S, vec_cp, compile)
        self.ffa = FusedFrameAsync(eng, self.bt, vec_cp).to(eng.device)
        self.frame_async = torch.compile(self.ffa, backend="neuron", dynamic=False) if compile else self.ffa

    @torch.no_grad()
    def run(self, requests, max_new=600, c2w=None, first=1, chunk=25, ctx=25):
        """One-step-lagged host loop: issue frame n, then process codes of frame n-1."""
        e, B, S, dev, dt = self.e, self.B, self.S, self.e.device, self.e.dtype
        V = e.talker.cfg.vocab
        queue = list(range(len(requests)))
        slot = [-1] * B
        nfr = [0] * B
        frames = [[] for _ in requests]
        out = [dict() for _ in requests]
        dec_upto = [0] * len(requests)
        st = dict(h=torch.zeros(B, 1, self.H, dtype=dt, device=dev), lg=torch.zeros(B, V, device=dev),
                  seen=torch.zeros(B, V, dtype=torch.bool, device=dev), pad=torch.zeros(B, 1, self.H, dtype=dt, device=dev),
                  step=torch.zeros(B, dtype=torch.long, device=dev), pos=torch.zeros(B, dtype=torch.long, device=dev))
        skip = [False] * B  # the in-flight frame for this row belongs to a finished request
        t0 = time.perf_counter()
        z1 = torch.zeros(1, dtype=torch.long, device=dev)

        def admit(b):
            if not queue:
                slot[b] = -1
                return
            r = queue.pop(0)
            emb, pe = requests[r]
            T = emb.shape[1]
            Tb = next(x for x in e.prefill_buckets if x >= T)
            np_ = Tb - T
            x = torch.zeros(1, Tb, self.H, dtype=dt); x[:, np_:] = emb.to(dt)
            p = torch.cat([torch.arange(S - np_, S), torch.arange(T)]).long()
            bi = torch.tensor([b], device=dev)
            hl, lg = self.prefill(x.to(dev), bi, p.to(dev))
            st["h"] = st["h"].index_copy(0, bi, hl)
            st["lg"] = st["lg"].index_copy(0, bi, lg)
            st["seen"] = st["seen"].index_copy(0, bi, torch.zeros(1, V, dtype=torch.bool, device=dev))
            st["pad"] = st["pad"].index_copy(0, bi, pe.to(dt).to(dev))
            st["step"] = st["step"].index_copy(0, bi, z1)
            st["pos"] = st["pos"].index_copy(0, bi, torch.full((1,), T, dtype=torch.long, device=dev))
            nfr[b] = 0
            slot[b] = r
            out[r]["t_admit"] = time.perf_counter() - t0

        def issue():
            st["h"], st["lg"], st["seen"], st["step"], st["pos"], c = self.frame_async(
                st["h"], st["lg"], st["seen"], st["step"], st["pad"], st["pos"])
            return c

        for b in range(B):
            admit(b)
        nsteps = 0
        pending = issue(); nsteps += 1
        while True:
            nxt = issue() if any(s >= 0 for s in slot) else None
            if nxt is not None:
                nsteps += 1
            cc = pending.cpu()
            refill = []
            for b in range(B):
                r = slot[b]
                if r < 0:
                    continue
                if skip[b]:
                    skip[b] = False
                    continue
                c = cc[b]
                if int(c[0]) == e.eos or nfr[b] >= max_new:
                    out[r]["codes"] = torch.stack(frames[r]) if frames[r] else torch.zeros(0, 16, dtype=torch.long)
                    out[r]["done_s"] = time.perf_counter() - t0
                    refill.append(b)
                    continue
                frames[r].append(c)
                nfr[b] += 1
                if c2w is not None:
                    n = len(frames[r])
                    if dec_upto[r] == 0 and n >= first:
                        c2w.decode_window(torch.stack(frames[r][:first]))
                        dec_upto[r] = first
                    elif dec_upto[r] > 0 and n >= dec_upto[r] + chunk:
                        k = min(ctx, dec_upto[r])
                        c2w.decode_window(torch.stack(frames[r][dec_upto[r] - k: dec_upto[r] + chunk]), ctx_frames=k)
                        dec_upto[r] += chunk
                if "ttfa_s" not in out[r] and len(frames[r]) >= first:
                    out[r]["ttfa_s"] = time.perf_counter() - t0 - out[r]["t_admit"]
            for b in refill:
                admit(b)
                if nxt is not None and slot[b] >= 0:
                    skip[b] = True  # frame `nxt` for row b was computed from the finished request's state
            if nxt is None:
                break
            pending = nxt
        if c2w is not None:
            for r, o in enumerate(out):
                n = len(o.get("codes", []))
                if n > dec_upto[r]:
                    k = min(ctx, dec_upto[r])
                    c2w.decode_window(o["codes"][dec_upto[r] - k:], ctx_frames=k)
        return out, time.perf_counter() - t0, nsteps
