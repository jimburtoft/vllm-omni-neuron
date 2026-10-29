#!/usr/bin/env python3
"""Simulator check of snake_ctconv (block head) and snake_conv_out (decoder output) vs fp32 torch."""
import os, sys
import numpy as np, ml_dtypes, torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nki
from qwen3_tts_neuron.code2wav import Code2Wav, _snake
from qwen3_tts_neuron.kernels.vocoder_nki import ctconv_params, out_params, snake_ctconv, snake_conv_out
m = Code2Wav(os.path.join(sys.argv[1], "speech_tokenizer"), dtype=torch.float32)
bf = ml_dtypes.bfloat16
cv = lambda t: t.float().numpy().astype(bf) if t.dtype == torch.bfloat16 else t.numpy()
snr = lambda o, r: 10 * torch.log10(r.pow(2).sum() / (o - r).pow(2).sum()).item()
torch.manual_seed(0)
for bi, Tin in ((3, 300), (2, 200), (1, 120)):
    Bk = m.blocks[bi]; r = m.rates[bi]
    x = (torch.randn(1, Bk["tw"].shape[0], Tin) * 0.5).to(torch.bfloat16)
    ref = m._ctconv(_snake(x.float(), Bk["a"], Bk["b"]), Bk["tw"], Bk["tb"], r)
    p = ctconv_params(Bk)
    o = nki.simulate(snake_ctconv)(cv(x), cv(p["wt"]), cv(p["bias"]), cv(p["a"]), cv(p["b"]), r, 128)
    o = torch.from_numpy(np.asarray(o).astype(np.float32))
    print(f"[sim] ctconv block {bi} r={r} {tuple(x.shape)} -> {tuple(o.shape)} ref {tuple(ref.shape)} snr {snr(o, ref):.1f} dB")
x = (torch.randn(1, 96, 1500) * 0.5).to(torch.bfloat16)
ref = m._cconv(_snake(x.float(), m.fa, m.fb), m.fw, m.fcb).clamp(-1, 1)
p = out_params(m.fw, m.fcb, m.fa, m.fb)
o = nki.simulate(snake_conv_out)(cv(x), cv(p["wt"]), cv(p["bias"]), cv(p["a"]), cv(p["b"]), 512)
o = torch.from_numpy(np.asarray(o).astype(np.float32))
print(f"[sim] conv_out {tuple(o.shape)} snr {snr(o, ref):.1f} dB")
