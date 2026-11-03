"""Standalone Moshi duplex engine on Neuron: Mimi encode -> LM frame -> Mimi decode, B rows in lockstep.

Per 80 ms tick, for every active row b:
    codes_in = mimi.encode(user_pcm[b])          (fresh encoder state on the row's frames 0 and 1: moshi/server.py
                                                   resets Mimi after encoding the first frame)
    tokens   = LMGenHost.inputs(codes_in)        (delay ring, host)
    out      = frame(tokens, ...)                (Helium + depformer + on-device sampling, one NEFF)
    ready, o = LMGenHost.commit(out)             (de-delay, host)
    pcm_out  = mimi.decode(o[:, 1:]) if ready    (decoder state advances only for ready rows)

`MoshiStage0` (encode + LM) and `MoshiDecoderRows` (decode) are the halves the two vLLM-Omni stages run;
`MoshiEngine` chains them in one process.
"""
from __future__ import annotations

import os
import time

import torch

from .config import FRAME, LMConfig, checkpoint_paths
from .lm import LMGenHost, MoshiFrame, MoshiLM
from .mimi import MimiStreaming


def gumbel(gen, shape, device):
    u = torch.rand(*shape, generator=gen).clamp_(1e-10, 1.0 - 1e-7)
    return (-torch.log(-torch.log(u))).to(device)


class MoshiStage0:
    """Mimi encoder + Moshi LM frame for B session rows. step() returns the raw generated tokens [B,9]
    (text + 8 audio codes of this step) and the de-delayed output frame [B,9] with its ready mask."""

    def __init__(self, B=1, device="cpu", compile_fn=None, lm_dtype=torch.bfloat16, mimi_dtype=torch.float32,
                 greedy=False, temp_text=0.7, top_k_text=25, temp=0.8, top_k=250, seed=0, model_dir=None):
        lm_path, mimi_path, tok_path = checkpoint_paths(model_dir=model_dir)
        cf = compile_fn or (lambda m: m)
        self.B, self.device, self.cfg = B, device, LMConfig()
        # TP>1: the Mimi encoder and (MOSHI_S0_DECODE=1) decoder are row-sharded across the TP ranks: rank r codes
        # rows [r*B/tp, (r+1)*B/tp); user codes are all-gathered (gloo, host) before the LM frame, PCM is gathered to
        # rank 0. Removes the need for a separate decoder core.
        self.tp = int(os.environ.get("MOSHI_TP", "1"))
        self.rank = 0
        if self.tp > 1:
            import torch.distributed as dist
            self.rank = dist.get_rank() % self.tp
            assert B % self.tp == 0, f"B={B} must be divisible by TP={self.tp}"
        self.Bl = B // self.tp
        self.r0, self.r1 = self.rank * self.Bl, (self.rank + 1) * self.Bl
        # Batch buckets (MOSHI_BUCKETS="4,8,16,24"): each step runs graphs compiled for the smallest bucket that
        # covers the highest active row, so step time scales with load. Rows 0..Bb-1 of the shared state are used;
        # the serving row allocator hands out the lowest free row. Buckets are multiples of TP (decoder rows are
        # interleaved across ranks: rank r owns global rows r, r+tp, r+2tp, ...).
        bl = os.environ.get("MOSHI_BUCKETS", "")
        bk = sorted({int(x) for x in bl.split(",") if x.strip()} | {B}) if bl else [B]
        # buckets < tp are allowed: rank 0 then decodes all rows (no gather)
        self.buckets = [b for b in bk if b <= B and (b % self.tp == 0 or b < self.tp)] or [B]
        self.decode_here = os.environ.get("MOSHI_S0_DECODE", "0") == "1"
        # MOSHI_ENC_ALL=1: every TP rank encodes all B rows (identical results, no host all-gather of codes, which
        # costs 10-17 ms per step through gloo while the encoder graph itself is ~2 ms)
        self.enc_all = self.tp > 1 and os.environ.get("MOSHI_ENC_ALL", "1") == "1"
        if self.enc_all:
            self.mimi = MimiStreaming(mimi_path, B, device, dtype=mimi_dtype, compile_fn=cf, decoder=False)
            self.mimi_dec = (MimiStreaming(mimi_path, self.Bl, device, dtype=mimi_dtype, compile_fn=cf, encoder=False)
                             if self.decode_here else None)
            self.interleave = True
        else:
            self.mimi = MimiStreaming(mimi_path, self.Bl, device, dtype=mimi_dtype, compile_fn=cf,
                                      decoder=self.decode_here)
            self.mimi_dec = self.mimi
            self.interleave = False
        self.lm = MoshiLM(lm_path, B, dtype=lm_dtype).to(device).eval()
        tt, ta = (0.0, 0.0) if greedy else (temp_text, temp)
        self.sampling = not greedy
        # Context buckets (MOSHI_CTX_BUCKETS="512,1536,3072", kernel path only): one frame graph per attended-slot
        # count; a step uses the smallest that covers every active row's window (fill + BLK + 1 slots).
        self._frames = {}
        cb = os.environ.get("MOSHI_CTX_BUCKETS", "")
        sctx = getattr(self.lm, "SCTX", self.lm.C)
        self.ctx_buckets = (sorted({int(x) for x in cb.split(",") if x.strip()} | {sctx})
                            if (cb and getattr(self.lm, "kernels", False)) else [sctx])
        blk = getattr(self.lm, "BLK", 1)
        for c in self.ctx_buckets:
            assert c % 128 == 0 and c % blk == 0, f"ctx bucket {c} must be a multiple of 128 and of the KV block"
            self._frames[c] = cf(MoshiFrame(self.lm, ctx=c, temp_text=tt, top_k_text=top_k_text, temp=ta,
                                            top_k=top_k).to(device))
        self.frame = self._frames[sctx]
        self.host = LMGenHost(B, self.cfg)
        self.gens = [torch.Generator().manual_seed(seed + b) for b in range(B)]
        # MOSHI_SAMPLER=multinomial (default): inverse-CDF multinomial over top-k, 1 uniform per sampler;
        # MOSHI_SAMPLER=gumbel: full-vocab Gumbel-max (B x 48k host random numbers per frame)
        self.uniform_sampler = os.environ.get("MOSHI_SAMPLER", "multinomial") == "multinomial"
        self._prof = os.environ.get("MOSHI_PARTS", "0") == "1"
        self._pt, self._pn, self._plast = {}, 0, 0.0
        self.nseen = torch.zeros(B, dtype=torch.long)
        self.lm_frames = torch.zeros(B, dtype=torch.long)   # host mirror of lm.offset (frames in each row's KV)
        z = lambda *s, **k: torch.zeros(*s, device=device, **k)
        self._zt, self._za = z(B, self.cfg.text_card), z(B, self.cfg.dep_q, self.cfg.card)
        self._ft, self._fa = z(B, dtype=torch.long), z(B, self.cfg.dep_q, dtype=torch.long)
        self._uf = z(B, dtype=torch.bool)
        self.tok_path = tok_path

    @torch.inference_mode()
    def reset(self, rows, seed=None):
        rows = list(rows)
        m = torch.zeros(self.B, dtype=torch.bool)
        m[rows] = True
        self.nseen = torch.where(m, torch.zeros_like(self.nseen), self.nseen)
        self.lm_frames = torch.where(m, torch.zeros_like(self.lm_frames), self.lm_frames)
        self.host.reset(rows)
        if seed is not None:
            for b in rows:
                self.gens[b] = torch.Generator().manual_seed(int(seed) + b)
            self._next = None

    def _draw(self):
        if self.uniform_sampler:
            # one uniform per sampler and row (inverse-CDF top-k sampling on device)
            gt = torch.stack([torch.rand(1, generator=g) for g in self.gens])
            ga = torch.stack([torch.rand(self.cfg.dep_q, 1, generator=g) for g in self.gens])
            return gt, ga
        gt = torch.stack([gumbel(g, (self.cfg.text_card,), "cpu") for g in self.gens])
        ga = torch.stack([gumbel(g, (self.cfg.dep_q, self.cfg.card), "cpu") for g in self.gens])
        return gt, ga

    def _noise(self):
        """Gumbel noise for this frame (host RNG, per-row generators). The next frame's noise is drawn while the
        device runs the current frame (`_prefetch`), so it costs only the host->device copy on the critical path."""
        if not self.sampling:
            return self._zt, self._za
        gt, ga = self._next if getattr(self, "_next", None) is not None else self._draw()
        self._next = None
        return gt.to(self.device), ga.to(self.device)

    def _prefetch(self):
        if self.sampling and getattr(self, "_next", None) is None:
            self._next = self._draw()

    def _tick(self, name):
        if self._prof:
            now = time.perf_counter()
            self._pt[name] = self._pt.get(name, 0.0) + now - self._plast
            self._plast = now

    def ctx_for(self, reset, active, Bb):
        """smallest context bucket whose table covers every active row's window: ofs + (t - s0) + 1 < ctx"""
        if len(self.ctx_buckets) == 1:
            return self.ctx_buckets[0]
        lm = self.lm
        W, blk = lm.cfg.context - 1, lm.BLK
        t = torch.where(reset, torch.zeros_like(self.lm_frames), self.lm_frames)[:Bb]
        a = active[:Bb]
        if not bool(a.any()):
            return self.ctx_buckets[0]
        tt = t[a]
        s0 = torch.clamp(tt - W, min=0)
        need = int(((s0 % blk) + (tt - s0)).max()) + 2   # window end index + current-token slot
        for c in self.ctx_buckets:
            if c >= need:
                self.last_ctx = c
                return c
        self.last_ctx = self.ctx_buckets[-1]
        return self.ctx_buckets[-1]

    def bucket_for(self, active):
        """smallest bucket covering the highest active row"""
        idx = torch.nonzero(active).view(-1)
        need = int(idx.max()) + 1 if idx.numel() else 1
        for b in self.buckets:
            if b >= need:
                return b
        return self.B

    @torch.inference_mode()
    def step(self, pcm_in, active=None):
        """pcm_in [B,1920] fp32 host. -> (raw [B,9], out [B,9], ready [B], (t_enc, t_lm)). Rows >= the chosen bucket
        must be inactive; their outputs are zeros and their state is untouched."""
        if self._prof:
            self._plast = time.perf_counter()
            self._pn += 1
            if self._pn % 200 == 0 and self.rank == 0:
                import logging
                logging.getLogger("moshi").warning("[MoshiS0 parts/200] %s", {k: round(v / 200 * 1e3, 2) for k, v in self._pt.items()})
                self._pt = {}
        B, dev = self.B, self.device
        active = torch.ones(B, dtype=torch.bool) if active is None else active.bool()
        Bb = self.bucket_for(active)
        self.last_bucket = Bb
        t0 = time.perf_counter()
        mreset = (self.nseen <= 1) & active
        reset = (self.nseen == 0) & active
        if self.enc_all:
            codes = self.mimi.encode(pcm_in[:Bb], reset=mreset[:Bb], active=active[:Bb])
            self._tick("enc_call")
            codes = codes.cpu()
            self._tick("enc_d2h")
        else:
            r0, r1 = self.r0, self.r1
            codes = self.mimi.encode(pcm_in[r0:r1], reset=mreset[r0:r1], active=active[r0:r1]).cpu()
            if self.tp > 1:
                import torch.distributed as dist
                parts = [torch.empty_like(codes) for _ in range(self.tp)]
                dist.all_gather(parts, codes)
                codes = torch.cat(parts, 0)
        if codes.shape[0] < B:
            codes = torch.cat([codes, torch.zeros(B - codes.shape[0], codes.shape[1], dtype=codes.dtype)])
        t1 = time.perf_counter()
        tokens = self.host.inputs(codes, active)
        self._tick("host_inputs")
        gt, ga = self._noise()
        self._tick("noise_h2d")
        frame = self._frames[self.ctx_for(reset, active, Bb)]
        raw_b = frame(tokens[:Bb].to(dev), reset[:Bb].to(dev), active[:Bb].to(dev), gt[:Bb], ga[:Bb],
                      self._ft[:Bb], self._fa[:Bb], self._uf[:Bb])
        self._tick("lm_call")
        self._prefetch()  # overlaps the device frame
        self._tick("prefetch")
        raw_b = raw_b.cpu()
        raw = raw_b if Bb == B else torch.cat([raw_b, torch.zeros(B - Bb, raw_b.shape[1], dtype=raw_b.dtype)])
        self._tick("lm_d2h")
        t2 = time.perf_counter()
        out, ready = self.host.commit(raw, active)
        self.nseen = self.nseen + active.long()
        self.lm_frames = torch.where(active, torch.where(reset, torch.ones_like(self.lm_frames), self.lm_frames + 1),
                                     self.lm_frames)
        self._tick("host_commit")
        if self.decode_here:
            t3 = time.perf_counter()
            self.last_pcm = self._decode_rows(out, ready, Bb)
            self.t_dec = time.perf_counter() - t3
            self._tick("decode+gather")
        return raw, out, ready, (t1 - t0, t2 - t1)

    @torch.inference_mode()
    def _decode_rows(self, out, ready, Bb=None):
        """Mimi-decode this rank's rows of the de-delayed frame; gather [B,1920] PCM to rank 0 (gloo).
        Rows are interleaved across ranks (rank r: rows r, r+tp, ...) so every rank has Bb/tp rows in any bucket."""
        Bb = Bb or self.B
        tp, r = self.tp, self.rank
        dreset = ready & (self.host.offsets == self.host.max_delay + 1)
        small = tp > 1 and Bb < tp
        if small:
            sel = torch.arange(Bb) if r == 0 else torch.arange(0)
        elif self.interleave:
            sel = torch.arange(r, Bb, tp)
        else:
            sel = torch.arange(self.r0, self.r1)
        if small:
            full = torch.zeros(self.B, FRAME)
            if r == 0 and bool(ready[sel].any()):
                full[sel] = self.mimi_dec.decode(out[sel, 1:].clamp(min=0), reset=dreset[sel], active=ready[sel]).cpu()
            return full
        if bool(ready[sel].any()):
            pcm = self.mimi_dec.decode(out[sel, 1:].clamp(min=0), reset=dreset[sel], active=ready[sel]).cpu()
        else:
            pcm = torch.zeros(sel.numel(), FRAME)
        if tp > 1:
            import torch.distributed as dist
            if r == 0:
                parts = [torch.empty_like(pcm) for _ in range(tp)]
                dist.gather(pcm, parts, dst=0)
                full = torch.zeros(self.B, FRAME)
                for q in range(tp):
                    rows = torch.arange(q, Bb, tp) if self.interleave else torch.arange(q * self.Bl, (q + 1) * self.Bl)
                    full[rows] = parts[q]
                pcm = full
            else:
                dist.gather(pcm, None, dst=0)
                pcm = torch.zeros(self.B, FRAME)
        elif Bb < self.B:
            full = torch.zeros(self.B, FRAME)
            full[sel] = pcm
            pcm = full
        return pcm


class MoshiDecoderRows:
    """Streaming Mimi decoder for B rows; decode(codes [B,8], active [B], reset [B]) -> pcm [B,1920]."""

    def __init__(self, B=1, device="cpu", compile_fn=None, dtype=torch.float32, model_dir=None):
        _, mimi_path, _ = checkpoint_paths(model_dir=model_dir, need_lm=False)
        self.B, self.device = B, device
        self.mimi = MimiStreaming(mimi_path, B, device, dtype=dtype, compile_fn=compile_fn, encoder=False)

    @torch.inference_mode()
    def decode(self, codes, active, reset):
        return self.mimi.decode(codes.clamp(min=0), reset=reset, active=active).cpu()


class MoshiEngine:
    def __init__(self, B=1, device="cpu", compile_fn=None, lm_dtype=torch.bfloat16, mimi_dtype=torch.float32,
                 greedy=False, seed=0, model_dir=None, **sampling):
        self.s0 = MoshiStage0(B, device, compile_fn, lm_dtype, mimi_dtype, greedy=greedy, seed=seed,
                              model_dir=model_dir, **sampling)
        self.dec = MoshiDecoderRows(B, device, compile_fn, mimi_dtype, model_dir=model_dir)
        self.B = B
        import sentencepiece
        self.tok = sentencepiece.SentencePieceProcessor(self.s0.tok_path)

    def reset(self, rows, seed=None):
        self.s0.reset(rows, seed)

    @torch.inference_mode()
    def step(self, pcm_in, active=None):
        """-> (pcm_out [B,1920], text [B], ready [B], (t_enc, t_lm, t_dec))."""
        B = self.B
        active = torch.ones(B, dtype=torch.bool) if active is None else active.bool()
        raw, out, ready, (te, tl) = self.s0.step(pcm_in, active)
        t2 = time.perf_counter()
        pcm = torch.zeros(B, FRAME)
        if bool(ready.any()):
            dreset = ready & (self.s0.host.offsets == self.s0.host.max_delay + 1)  # the row's first output frame
            pcm = self.dec.decode(out[:, 1:], ready, dreset)
        return pcm, out[:, 0], ready, (te, tl, time.perf_counter() - t2)

    def piece(self, t):
        t = int(t)
        if t in (0, 3) or t < 0:
            return ""
        return self.tok.id_to_piece(t).replace("\u2581", " ")


def run_file(engine: MoshiEngine, wavs, verbose=True):
    """wavs: list of B 1-D fp32 tensors (24 kHz). Runs them in lockstep; returns per-row (pcm, text) + stats."""
    B = engine.B
    T = max(w.shape[0] for w in wavs) // FRAME
    outs = [[] for _ in range(B)]
    texts = [[] for _ in range(B)]
    tim = []
    for t in range(T):
        x = torch.zeros(B, FRAME)
        act = torch.zeros(B, dtype=torch.bool)
        for b, w in enumerate(wavs):
            if (t + 1) * FRAME <= w.shape[0]:
                x[b] = w[t * FRAME:(t + 1) * FRAME]
                act[b] = True
        t0 = time.perf_counter()
        pcm, text, ready, parts = engine.step(x, act)
        tim.append((time.perf_counter() - t0, *parts))
        for b in range(B):
            if bool(ready[b]):
                outs[b].append(pcm[b])
                texts[b].append(engine.piece(text[b]))
        if verbose and t % 50 == 0:
            print(f"  t={t} step={tim[-1][0] * 1e3:.1f} ms (enc {parts[0] * 1e3:.1f} lm {parts[1] * 1e3:.1f} "
                  f"dec {parts[2] * 1e3:.1f})", flush=True)
    res = [(torch.cat(o) if o else torch.zeros(0), "".join(tx)) for o, tx in zip(outs, texts)]
    return res, torch.tensor(tim)
