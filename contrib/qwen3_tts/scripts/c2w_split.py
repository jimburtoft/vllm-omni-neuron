#!/usr/bin/env python3
"""Device time of Code2Wav crop-window stages (cumulative prefixes compiled as separate graphs).
usage: c2w_split.py <model_dir>"""
import math, os, sys, time
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen3_tts_neuron.serving.common import neuron_runtime_env, compile_fn
neuron_runtime_env()
import libtorch_neuronx_lite  # noqa: F401
import qwen3_tts_neuron.code2wav as cw
dev = torch.device("neuron:0")
m = cw.Code2Wav(os.path.join(sys.argv[1], "speech_tokenizer"), dtype=torch.bfloat16).to(dev)
c = compile_fn()


class Prefix(nn.Module):
    """forward() of Code2Wav truncated after `stop` (0=transformer, 1=upsampler, 2..5=after decoder block k-2, 6=all)."""
    def __init__(self, stop):
        super().__init__(); self.stop = stop

    def forward(self, codes):
        self_ = m; crop_ctx, m1, m2 = 25, 12, 2
        Bt = codes.shape[0]
        idx = (codes + self_.book_off).reshape(-1)
        e = F.embedding(idx, self_.books).view(Bt, self_.nq, -1, self_.books.shape[-1])
        x = e.sum(1).transpose(1, 2)
        x = self_._cconv(x, self_.pre_conv_w, self_.pre_conv_b)
        x = self_._transformer(x.transpose(1, 2)).transpose(1, 2)
        if self.stop == 0:
            return x.sum()
        a = crop_ctx - m1
        x = x[..., a:]
        for U, r in zip(self_.ups, self_.up_ratios):
            x = self_._ctconv(x, U["tw"], U["tb"], r)
            res = x
            y = self_._cconv(x, U["dw"], U["db"], groups=x.shape[1]).transpose(1, 2)
            y = F.layer_norm(y, (y.shape[-1],), U["lnw"], U["lnb"], 1e-6)
            y = F.gelu(y @ U["p1w"] + U["p1b"]) @ U["p2w"] + U["p2b"]
            x = res + (U["gamma"] * y).transpose(1, 2)
        x = self_._cconv(x, self_.d0_w, self_.d0_b)
        if self.stop == 1:
            return x.sum()
        b = crop_ctx - m2
        per = int(math.prod(self_.up_ratios))
        for bi, (Bk, r) in enumerate(zip(self_.blocks, self_.rates)):
            if bi == 1:
                x = x[..., (b - a) * per * self_.rates[0]:]
            x = cw._snake(x, Bk["a"], Bk["b"])
            x = self_._ctconv(x, Bk["tw"], Bk["tb"], r)
            for ri, dil in enumerate((1, 3, 9)):
                res = x
                y = cw._snake(x, Bk[f"r{ri}a1"], Bk[f"r{ri}b1"])
                y = self_._cconv(y, Bk[f"r{ri}w1"], Bk[f"r{ri}c1"], dilation=dil)
                y = cw._snake(y, Bk[f"r{ri}a2"], Bk[f"r{ri}b2"])
                y = self_._cconv(y, Bk[f"r{ri}w2"], Bk[f"r{ri}c2"])
                x = res + y
            if self.stop == 2 + bi:
                return x.sum()
        x = cw._snake(x, self_.fa, self_.fb)
        x = self_._cconv(x, self_.fw, self_.fcb)
        return x.sum()


codes = torch.randint(0, 2048, (1, 16, 50)).to(dev)
prev = 0.0
print("rates", m.rates, "ups", m.up_ratios)
with torch.inference_mode():
    for stop, name in enumerate(["pre+transformer", "upsampler", "block0", "block1", "block2", "block3", "final"]):
        fn = c(Prefix(stop))
        for _ in range(3): fn(codes).cpu()
        ts = []
        for _ in range(20):
            t0 = time.perf_counter(); fn(codes).cpu(); ts.append(time.perf_counter() - t0)
        t = 1000 * sorted(ts)[10]
        print(f"through {name}: {t:.2f} ms  (stage {t - prev:+.2f} ms)", flush=True)
        prev = t
