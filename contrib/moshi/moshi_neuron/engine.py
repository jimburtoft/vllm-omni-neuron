"""Standalone Moshi duplex engine on Neuron: Mimi encode -> LM frame -> Mimi decode, B rows in lockstep.

Per 80 ms tick, for every row b:
    codes_in = mimi.encode(user_pcm[b])          (row reset on its first frame, as moshi/server.py does)
    tokens   = LMGenHost.inputs(codes_in)        (delay ring, host)
    out      = frame(tokens, ...)                (Helium + depformer + on-device sampling, one NEFF)
    ready, o = LMGenHost.commit(out)             (de-delay, host)
    pcm_out  = mimi.decode(o[:, 1:]) if ready    (decoder state advances only for ready rows)

The host moves 1920 fp32 samples in and out per row, plus about 100 bytes of tokens.
"""
from __future__ import annotations

import os
import time

import torch

from .config import FRAME, LMConfig, checkpoint_paths
from .lm import LMGenHost, MoshiFrame, MoshiLM
from .mimi import MimiStreaming


class MoshiEngine:
    def __init__(self, B=1, device="cpu", compile_fn=None, lm_dtype=torch.bfloat16, mimi_dtype=torch.float32,
                 greedy=False, temp_text=0.7, top_k_text=25, temp=0.8, top_k=250, seed=0, context=None):
        lm_path, mimi_path, tok_path = checkpoint_paths()
        cf = compile_fn or (lambda m: m)
        self.B, self.device = B, device
        self.cfg = LMConfig()
        self.mimi = MimiStreaming(mimi_path, B, device, dtype=mimi_dtype, compile_fn=cf)
        self.lm = MoshiLM(lm_path, B, dtype=lm_dtype, context=context).to(device).eval()
        tt, ta = (0.0, 0.0) if greedy else (temp_text, temp)
        self.sampling = not greedy
        self.frame = cf(MoshiFrame(self.lm, ctx=self.lm.C, temp_text=tt, top_k_text=top_k_text, temp=ta,
                                   top_k=top_k).to(device))
        self.host = LMGenHost(B, self.cfg)
        self.gen = torch.Generator().manual_seed(seed)
        # frames seen per row: moshi/server.py resets Mimi after encoding the first frame, so frames 0 AND 1
        # are both encoded from fresh state.
        self.nseen = torch.zeros(B, dtype=torch.long)
        self._zt = torch.zeros(B, self.cfg.text_card, device=device)
        self._za = torch.zeros(B, self.cfg.dep_q, self.cfg.card, device=device)
        self._ft = torch.zeros(B, dtype=torch.long, device=device)
        self._fa = torch.zeros(B, self.cfg.dep_q, dtype=torch.long, device=device)
        self._uf = torch.zeros(B, dtype=torch.bool, device=device)
        import sentencepiece
        self.tok = sentencepiece.SentencePieceProcessor(tok_path)

    def _gumbel(self, *shape):
        u = torch.rand(*shape, generator=self.gen).clamp_(1e-10, 1.0 - 1e-7)
        return (-torch.log(-torch.log(u))).to(self.device)

    @torch.inference_mode()
    def reset(self, rows):
        rows = list(rows)
        m = torch.zeros(self.B, dtype=torch.bool)
        m[rows] = True
        self.nseen = torch.where(m, torch.zeros_like(self.nseen), self.nseen)
        self.host.reset(rows)

    @torch.inference_mode()
    def step(self, pcm_in, active=None):
        """pcm_in [B,1920] fp32 host tensor. Returns (pcm_out [B,1920] host, text [B] host, ready [B] bool, timings)."""
        B = self.B
        active = torch.ones(B, dtype=torch.bool) if active is None else active
        t0 = time.perf_counter()
        # Mimi encoder: fresh state on frames 0 and 1 (server.py resets after the first frame); LM: frame 0 only.
        mreset = (self.nseen <= 1) & active
        reset = (self.nseen == 0) & active
        codes = self.mimi.encode(pcm_in, reset=mreset, active=active).cpu()
        t1 = time.perf_counter()
        tokens = self.host.inputs(codes, active)
        gt = self._gumbel(B, self.cfg.text_card) if self.sampling else self._zt
        ga = self._gumbel(B, self.cfg.dep_q, self.cfg.card) if self.sampling else self._za
        dev = self.device
        out = self.frame(tokens.to(dev), reset.to(dev), active.to(dev), gt, ga, self._ft, self._fa, self._uf).cpu()
        t2 = time.perf_counter()
        res, ready = self.host.commit(out, active)
        pcm = torch.zeros(B, FRAME)
        if bool(ready.any()):
            # decoder state advances only for rows that produced a frame; it resets on the row's first output
            dreset = ready & (self.host.offsets == self.host.max_delay + 1)
            pcm = self.mimi.decode(res[:, 1:].clamp(min=0), reset=dreset, active=ready).cpu()
        t3 = time.perf_counter()
        self.nseen = self.nseen + active.long()
        return pcm, res[:, 0], ready, (t1 - t0, t2 - t1, t3 - t2)

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
