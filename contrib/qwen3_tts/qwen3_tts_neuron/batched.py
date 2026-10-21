"""Batched (B concurrent streams per NeuronCore) Talker + Code Predictor with continuous slot refill.

KV layout per layer: flat buffer [B*nkv*S, D] (viewed as [B, nkv, S, D] for attention), written with a single
`index_copy_` along dim 0 at per-row positions — avoids per-row index_copy along the seq dim.

Graphs (static): prefill_slot(T_bucket) writes one request into slot b; decode(B) one step for all B rows at
per-row positions; cp_frame(B) the 15-step Code Predictor for B rows (positions shared, restart every frame).
Finished rows keep running harmlessly until refilled (their outputs are ignored).
"""
from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from .engine import TTSEngine, rms, rotate_half


def _qkv_b(l, x, cos, sin):
    # x [B,T,H]; cos/sin [B,T,D]
    B, T, _ = x.shape
    hn = rms(x, l.ln1, l.eps)
    q, k, v = (hn @ l.w_qkv).split([l.q_sz, l.kv_sz, l.kv_sz], dim=-1)
    q = rms(q.view(B, T, l.nh, l.d), l.qn, l.eps).transpose(1, 2)
    k = rms(k.view(B, T, l.nkv, l.d), l.kn, l.eps).transpose(1, 2)
    v = v.view(B, T, l.nkv, l.d).transpose(1, 2)
    c, s = cos.view(B, 1, T, l.d), sin.view(B, 1, T, l.d)
    return q * c + rotate_half(q) * s, k * c + rotate_half(k) * s, v


def _attn_b(l, q, k, v, mask):
    # q [B,nh,T,D]; k/v [B,nkv,S,D]; mask [B,T,S] additive fp32
    B, _, T, D = q.shape
    S = k.shape[2]
    qg = q.reshape(B, l.nkv, l.rep * T, D)
    sc = (qg @ k.transpose(-1, -2)).float() * l.scale
    sc = sc.view(B, l.nkv, l.rep, T, S) + mask.view(B, 1, 1, T, S)
    p = torch.softmax(sc, -1).to(q.dtype).view(B, l.nkv, l.rep * T, S)
    return (p @ v).view(B, l.nh, T, D).transpose(1, 2).reshape(B, T, l.nh * D)


class BatchedTalker(nn.Module):
    def __init__(self, talker, B, S):
        super().__init__()
        self.t, self.B, self.S = talker, B, S
        c = talker.cfg
        self.nkv, self.D = c.kv, c.head_dim
        for i in range(c.layers):
            self.register_buffer(f"ck{i}", torch.zeros(B * c.kv * S, c.head_dim, dtype=talker.dtype), persistent=False)
            self.register_buffer(f"cv{i}", torch.zeros(B * c.kv * S, c.head_dim, dtype=talker.dtype), persistent=False)
        self.register_buffer("hoff", torch.arange(c.kv), persistent=False)

    def _run(self, x, rows, pos, mask, cos, sin, ctx=None):
        """x [R,T,H] for rows `rows` [R] at positions pos [R,T]. Attention reads only the first `ctx` KV slots
        (static; default all S) -- decode context bucketing."""
        Sc = ctx or self.S
        R, T, _ = x.shape
        # flat write index: ((row*nkv + h)*S + pos)  ordered (r, h, t) to match k.reshape(R*nkv*T, D)
        widx = ((rows.view(R, 1, 1) * self.nkv + self.hoff.view(1, -1, 1)) * self.S + pos.view(R, 1, T)).reshape(-1)
        h = x
        for i, l in enumerate(self.t.layers):
            ck, cv = getattr(self, f"ck{i}"), getattr(self, f"cv{i}")
            q, k, v = _qkv_b(l, h, cos, sin)
            ck.index_copy_(0, widx, k.reshape(-1, self.D))
            cv.index_copy_(0, widx, v.reshape(-1, self.D))
            K = ck.view(self.B, self.nkv, self.S, self.D)[:, :, :Sc]
            V = cv.view(self.B, self.nkv, self.S, self.D)[:, :, :Sc]
            if R != self.B:
                K, V = K.index_select(0, rows), V.index_select(0, rows)
            o = _attn_b(l, q, K, V, mask)
            h = h + o @ l.w_o
            h = h + l._mlp(h)
        hl = rms(h[:, -1:, :], self.t.norm, self.t.cfg.eps)
        return hl, (hl @ self.t.codec_head).float().view(R, -1)

    def prefill(self, x, row, pos):
        # x [1,Tb,H], row [1], pos [Tb]
        j = torch.arange(self.S, device=x.device)
        mask = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32).view(1, -1, self.S)
        cos = self.t.cos.index_select(0, pos).unsqueeze(0)
        sin = self.t.sin.index_select(0, pos).unsqueeze(0)
        return self._run(x, row, pos.view(1, -1), mask, cos, sin)

    def decode(self, x, pos, ctx=None):
        # x [B,1,H], pos [B]; ctx: static number of KV slots to attend (all rows must have pos < ctx)
        B, Sc = self.B, ctx or self.S
        j = torch.arange(Sc, device=x.device)
        mask = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32).view(B, 1, Sc)
        cos = self.t.cos.index_select(0, pos).view(B, 1, -1)
        sin = self.t.sin.index_select(0, pos).view(B, 1, -1)
        rows = torch.arange(B, device=x.device)
        return self._run(x, rows, pos.view(B, 1), mask, cos, sin, ctx=Sc)


def prefill_plan(T: int, buckets, S: int):
    """Split a T-token prompt into prefill calls over static buckets.

    Returns [(start, n, bucket, pos)] where pos is the LongTensor of KV positions for the bucket: full chunks are
    exact; the final chunk is front-padded and its pad tokens are parked in the top KV slots [S-pad, S), which stay
    masked (positions of real tokens are < S-pad until generation overwrites them in order)."""
    bmax = max(buckets)
    out, start = [], 0
    while T - start > bmax:
        out.append((start, bmax, bmax, torch.arange(start, start + bmax)))
        start += bmax
    n = T - start
    b = next(x for x in sorted(buckets) if x >= n)
    pad = b - n
    pos = torch.cat([torch.arange(S - pad, S), torch.arange(start, T)]).long()
    out.append((start, n, b, pos))
    return out


def cp_frame_b(cp, talker_hidden, cb0_embed):
    """Batched Code Predictor frame. talker_hidden/cb0_embed [B,1,2048] -> codes [B,15], emb_sum [B,1,2048]."""
    cfg = cp.cfg
    B = talker_hidden.shape[0]
    dev = talker_hidden.device
    L = len(cp.layers)
    kc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=dev) for _ in range(L)]
    vc = [torch.zeros(B, cfg.kv, cp.S, cfg.head_dim, dtype=cp.dtype, device=dev) for _ in range(L)]
    j = torch.arange(cp.S, device=dev)

    def run(x, pos):
        T = x.shape[1]
        cos = cp.cos.index_select(0, pos)
        sin = cp.sin.index_select(0, pos)
        m = torch.where(j.view(1, -1) <= pos.view(-1, 1), 0.0, -1e30).to(torch.float32)
        h = x
        for i, l in enumerate(cp.layers):
            h, kc[i], vc[i] = l.forward_local(h, cos, sin, kc[i], vc[i], pos, m)
        return rms(h[:, -1:, :], cp.norm, cfg.eps)

    x0 = torch.cat([talker_hidden, cb0_embed], 1) @ cp.proj_w + cp.proj_b
    h = run(x0, torch.arange(2, device=dev))
    codes, emb_sum = [], torch.zeros_like(talker_hidden)
    for g in range(cfg.groups - 1):
        c = torch.argmax((h @ cp.heads[g]).float(), dim=-1)  # [B,1]
        codes.append(c)
        e = F.embedding(c, cp.embeds[g])  # [B,1,2048]
        emb_sum = emb_sum + e
        if g < cfg.groups - 2:
            h = run(e @ cp.proj_w + cp.proj_b, torch.full((1,), g + 2, device=dev, dtype=torch.long))
    return torch.cat(codes, -1), emb_sum


class _M(nn.Module):
    def __init__(self, fn, *mods):
        super().__init__()
        self.fn = fn
        self.mods = nn.ModuleList(mods)

    def forward(self, *a):
        return self.fn(*a)


class BatchedEngine:
    def __init__(self, eng: TTSEngine, B: int, S: int = 512, compile=True):
        self.e, self.B, self.S = eng, B, S
        self.bt = BatchedTalker(eng.talker, B, S).to(eng.device)
        pre = _M(self.bt.prefill, self.bt)
        dec = _M(self.bt.decode, self.bt)
        cpf = _M(lambda h, e0: cp_frame_b(eng.cp, h, e0), eng.cp)
        c = (lambda m: torch.compile(m, backend="neuron", dynamic=False)) if compile else (lambda m: m)
        self.prefill, self.decode, self.cp = c(pre), c(dec), c(cpf)
        self.H = eng.talker.cfg.hidden

    @torch.no_grad()
    def run(self, requests, max_new=600, c2w=None, first=1, chunk=25, ctx=25):
        """Continuous batching over `requests` = list of (inputs_embeds [1,T,H], tts_pad [1,1,H]).
        Returns per-request dict(codes, ttfa_s, done_s) and total wall."""
        e, B, S, dev, dt = self.e, self.B, self.S, self.e.device, self.e.dtype
        V = e.talker.cfg.vocab
        ban = torch.zeros(V, dtype=torch.bool); ban[V - 1024:] = True; ban[e.eos] = False
        queue = list(range(len(requests)))
        slot_req = [-1] * B
        seen = torch.zeros(B, V, dtype=torch.bool)
        steps = [0] * B
        frames = [[] for _ in requests]
        out = [dict() for _ in requests]
        pos = torch.zeros(B, dtype=torch.long)
        h_all = torch.zeros(B, 1, self.H, dtype=dt, device=dev)
        pad_all = torch.zeros(B, 1, self.H, dtype=dt)
        logits = torch.zeros(B, V)
        dec_upto = [0] * len(requests)
        t0 = time.perf_counter()

        def admit(b):
            if not queue:
                slot_req[b] = -1
                return
            r = queue.pop(0)
            emb, pad = requests[r]
            T = emb.shape[1]
            Tb = next(x for x in e.prefill_buckets if x >= T)
            np_ = Tb - T
            x = torch.zeros(1, Tb, self.H, dtype=dt); x[:, np_:] = emb.to(dt)
            p = torch.cat([torch.arange(S - np_, S), torch.arange(T)]).long()
            hl, lg = self.prefill(x.to(dev), torch.tensor([b], device=dev), p.to(dev))
            h_all[b] = hl[0]
            logits[b] = lg[0].cpu()
            pad_all[b] = pad[0].to(dt)
            pos[b] = T
            seen[b] = False
            steps[b] = 0
            slot_req[b] = r
            out[r]["t_admit"] = time.perf_counter() - t0

        for b in range(B):
            admit(b)
        pad_dev = pad_all.to(dev)
        active_steps = 0
        while any(r >= 0 for r in slot_req):
            # host sampling for all rows
            l = logits.clone()
            l = torch.where(seen & (l > 0), l / 1.05, torch.where(seen, l * 1.05, l))
            l[:, ban] = float("-inf")
            for b in range(B):
                if steps[b] < 2:
                    l[b, e.eos] = float("-inf")
            cb0 = torch.argmax(l, -1)
            refill = []
            for b in range(B):
                r = slot_req[b]
                if r < 0:
                    continue
                if int(cb0[b]) == e.eos or steps[b] >= max_new:
                    out[r]["codes"] = torch.stack(frames[r]) if frames[r] else torch.zeros(0, 16, dtype=torch.long)
                    out[r]["done_s"] = time.perf_counter() - t0
                    refill.append(b)
                else:
                    seen[b, int(cb0[b])] = True
            for b in refill:
                admit(b)
            if refill:
                pad_dev = pad_all.to(dev)
                # re-sample freshly admitted rows from their prefill logits next iteration
                continue_rows = set(refill)
            else:
                continue_rows = set()
            if not any(r >= 0 for r in slot_req):
                break
            if continue_rows:
                # recompute cb0 for refilled rows from their prefill logits
                l = logits.clone()
                l[:, ban] = float("-inf")
                for b in continue_rows:
                    l[b, e.eos] = float("-inf")
                    cb0[b] = torch.argmax(l[b])
                    seen[b, int(cb0[b])] = True
            cb0_d = cb0.to(dev).view(B, 1)
            e0 = F.embedding(cb0_d, e.codec_emb_dev)  # [B,1,H]
            res, es = self.cp(h_all, e0)
            h_all, lg = self.decode(e0 + es + pad_dev, pos.to(dev))
            logits = lg.cpu()
            res_c = res.cpu()
            active_steps += 1
            for b in range(B):
                r = slot_req[b]
                if r < 0:
                    continue
                frames[r].append(torch.cat([cb0[b].view(1), res_c[b]]))
                steps[b] += 1
                if "ttfa_s" not in out[r] and len(frames[r]) >= first:
                    if c2w is not None:
                        c2w.decode_window(torch.stack(frames[r][:first]))
                    out[r]["ttfa_s"] = time.perf_counter() - t0 - out[r]["t_admit"]
                    dec_upto[r] = first
                elif c2w is not None and len(frames[r]) >= dec_upto[r] + chunk:
                    cc = min(ctx, dec_upto[r])
                    c2w.decode_window(torch.stack(frames[r][dec_upto[r] - cc: dec_upto[r] + chunk]))
                    dec_upto[r] += chunk
            pos += torch.tensor([1 if slot_req[b] >= 0 else 0 for b in range(B)])
        wall = time.perf_counter() - t0
        if c2w is not None:  # flush tails
            for r, o in enumerate(out):
                n = len(o.get("codes", []))
                if n > dec_upto[r]:
                    cc = min(ctx, dec_upto[r])
                    c2w.decode_window(o["codes"][dec_upto[r] - cc:])
            wall = time.perf_counter() - t0
        return out, wall, active_steps
