#!/usr/bin/env python3
"""cp_step NKI kernel vs the torch code-predictor layers (fp32 CPU reference) for one step at position `pos`
with a KV cache pre-filled by the reference for positions < pos.
usage: cp_kernel_check.py <model> [--device sim|neuron] [--B 8] [--pos 0,3,16]"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--device", default="sim")
ap.add_argument("--B", type=int, default=8); ap.add_argument("--pos", default="0,3,16")
a = ap.parse_args()
from qwen3_tts_neuron.engine import TTSEngine, rms
from qwen3_tts_neuron.kernels.cp_nki import cp_params, cp_step
eng = TTSEngine(a.model, device="cpu", max_ctx=64, dtype=torch.float32, compile=False)
cp = eng.cp; cfg = cp.cfg; B = a.B; S = cp.S; L = len(cp.layers)
torch.manual_seed(0)

def ref_step(x, kc, vc, pos):
    """x [B,1,D]; kc/vc lists of [B,nkv,S,D] (positions < pos valid); returns y [B,D] and updated caches."""
    p = torch.tensor([pos])
    cos, sin = cp.cos.index_select(0, p), cp.sin.index_select(0, p)
    m = torch.where(torch.arange(S).view(1, -1) <= p.view(-1, 1), 0.0, -1e30).float()
    h = x
    for i, l in enumerate(cp.layers):
        h, kc[i], vc[i] = l.forward_local(h, cos, sin, kc[i], vc[i], p, m)
    return rms(h[:, -1:, :], cp.norm, cfg.eps)[:, 0], kc, vc

pp = cp_params(cp)
bf = None
for pos in [int(x) for x in a.pos.split(",")]:
    kc = [torch.zeros(B, cfg.kv, S, cfg.head_dim) for _ in range(L)]
    vc = [torch.zeros(B, cfg.kv, S, cfg.head_dim) for _ in range(L)]
    for p0 in range(pos):  # fill the cache with real reference steps
        _, kc, vc = ref_step(torch.randn(B, 1, cfg.hidden) * 0.5, kc, vc, p0)
    x = torch.randn(B, 1, cfg.hidden) * 0.5
    kcl = torch.stack(kc)  # [L,B,8,S,128]
    vcl = torch.stack(vc)
    y_ref, kc2, vc2 = ref_step(x, kc, vc, pos)
    args = dict(x=x[:, 0], kc=kcl, vc=vcl, **pp)
    if a.device == "sim":
        import nki, ml_dtypes
        bf = ml_dtypes.bfloat16
        cv = lambda t, k: t.float().numpy().astype(bf) if k in ("x", "kc", "vc", "wqkv", "wo", "wgu", "wd") else t.float().numpy()
        nargs = {k: cv(v, k) for k, v in args.items()}
        t0 = time.time()
        y = nki.simulate(cp_step)(nargs["x"], nargs["kc"], nargs["vc"], nargs["wqkv"], nargs["wo"], nargs["wgu"], nargs["wd"],
                                  nargs["ln1"], nargs["ln2"], nargs["qn"], nargs["kn"], nargs["norm"], nargs["cos"], nargs["sin"],
                                  pos, cfg.eps, skip=int(os.environ.get("CPK_SKIP", "0")), attn=int(os.environ.get("CPK_ATTN", "1")))
        y = torch.from_numpy(np.asarray(y).astype(np.float32)); kout = torch.from_numpy(nargs["kc"].astype(np.float32))
        el = time.time() - t0
    # bf16-noise yardstick: the same reference run with bf16 weights/activations
    if not hasattr(sys, "_bfeng"):
        sys._bfeng = TTSEngine(a.model, device="cpu", max_ctx=64, dtype=torch.bfloat16, compile=False)
    cpb = sys._bfeng.cp
    pb = torch.tensor([pos]); mb = torch.where(torch.arange(S).view(1, -1) <= pb.view(-1, 1), 0.0, -1e30).float()
    hb = x.to(torch.bfloat16); kcb = [k.to(torch.bfloat16) for k in kc]; vcb = [v.to(torch.bfloat16) for v in vc]
    for i, l in enumerate(cpb.layers):
        hb, kcb[i], vcb[i] = l.forward_local(hb, cpb.cos.index_select(0, pb), cpb.sin.index_select(0, pb), kcb[i], vcb[i], pb, mb)
    yb = rms(hb[:, -1:, :], cpb.norm, cfg.eps)[:, 0].float()
    print(f"[cpk]   torch-bf16 vs fp32: min cos {F.cosine_similarity(yb, y_ref, dim=-1).min().item():.5f} maxabs {(yb - y_ref).abs().max().item():.3e}")
    cosm = F.cosine_similarity(y, y_ref, dim=-1).min().item()
    err = (y - y_ref).abs().max().item()
    kref = torch.stack(kc2)[:, :, :, pos]
    kerr = (kout[:, :, :, pos] - kref).abs().max().item()
    print(f"[cpk] {a.device} B={B} pos={pos}: y min cos {cosm:.5f} maxabs {err:.3e} | k-write maxabs {kerr:.3e} | {el:.1f}s", flush=True)
