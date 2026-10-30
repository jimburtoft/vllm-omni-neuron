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
        self.mimi = MimiStreaming(mimi_path, B, device, dtype=mimi_dtype, compile_fn=cf, decoder=False)
        self.lm = MoshiLM(lm_path, B, dtype=lm_dtype).to(device).eval()
        tt, ta = (0.0, 0.0) if greedy else (temp_text, temp)
        self.sampling = not greedy
        self.frame = cf(MoshiFrame(self.lm, ctx=self.lm.C, temp_text=tt, top_k_text=top_k_text, temp=ta,
                                   top_k=top_k).to(device))
        self.host = LMGenHost(B, self.cfg)
        self.gens = [torch.Generator().manual_seed(seed + b) for b in range(B)]
        self.nseen = torch.zeros(B, dtype=torch.long)
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
        self.host.reset(rows)
        if seed is not None:
            for b in rows:
                self.gens[b] = torch.Generator().manual_seed(int(seed) + b)

    def _noise(self):
        if not self.sampling:
            return self._zt, self._za
        gt = torch.stack([gumbel(g, (self.cfg.text_card,), "cpu") for g in self.gens]).to(self.device)
        ga = torch.stack([gumbel(g, (self.cfg.dep_q, self.cfg.card), "cpu") for g in self.gens]).to(self.device)
        return gt, ga

    @torch.inference_mode()
    def step(self, pcm_in, active=None):
        """pcm_in [B,1920] fp32 host. -> (raw [B,9], out [B,9], ready [B], (t_enc, t_lm))."""
        B, dev = self.B, self.device
        active = torch.ones(B, dtype=torch.bool) if active is None else active.bool()
        t0 = time.perf_counter()
        mreset = (self.nseen <= 1) & active
        reset = (self.nseen == 0) & active
        codes = self.mimi.encode(pcm_in, reset=mreset, active=active).cpu()
        t1 = time.perf_counter()
        tokens = self.host.inputs(codes, active)
        gt, ga = self._noise()
        raw = self.frame(tokens.to(dev), reset.to(dev), active.to(dev), gt, ga, self._ft, self._fa, self._uf).cpu()
        t2 = time.perf_counter()
        out, ready = self.host.commit(raw, active)
        self.nseen = self.nseen + active.long()
        return raw, out, ready, (t1 - t0, t2 - t1)


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
