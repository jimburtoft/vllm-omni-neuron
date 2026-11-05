"""Repro: vllm_neuron NF.attention_decode (nkilib attention_block_tkg) with FP8 block KV vs bf16 block KV, one layer,
Moshi shapes (H heads x 128, block KV [B*NBLK, H, 32, 128], per-head block table, pos_ids/swa_start in-kernel mask).
Prefills the cache with random K/V (same values for both dtypes), runs one decode call, compares attention output and
the written cache slot. Sweep B to find where FP8 breaks.

  python fp8kv_repro.py --H 32 --B 1,2,4,5,8 --fill 300
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device  # noqa: E402


class One(torch.nn.Module):
    def __init__(self, fp8, scale_k, scale_v):
        super().__init__()
        self.fp8, self.sk, self.sv = fp8, scale_k, scale_v

    def forward(self, x, n1, wqkv, wo, cos, sin, table, K, V, pos_ids, swa, widx, ks, vs):
        from vllm_neuron.functional.attention.attention_decode import attention_decode
        return attention_decode(
            X=x, X_hidden_dim_actual=x.shape[-1], rmsnorm_X_enabled=True, rmsnorm_X_eps=1e-8, rmsnorm_X_gamma=n1,
            W_qkv=wqkv, bias_qkv=None, cos=cos, sin=sin, rope_contiguous_layout=True, K_cache_transposed=False,
            active_blocks_table=table, K_cache=K, V_cache=V, attention_mask=None,
            softmax_scale=128 ** -0.5 / (self.sk if self.fp8 else 1.0), sink=None, pos_ids=pos_ids,
            swa_start_pos_ids=swa, update_cache=True, kv_cache_update_idx=widx,
            k_scale=ks if self.fp8 else None, v_scale=vs if self.fp8 else None,
            W_out=None if os.environ.get("RP_NOPROJ") else wo, bias_out=None, transposed_out=False, out_in_sb=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=32)
    ap.add_argument("--B", default="1,2,4,5,8")
    ap.add_argument("--fill", type=int, default=300)
    ap.add_argument("--blk", type=int, default=32)
    ap.add_argument("--slots", type=int, default=3072)
    ap.add_argument("--varfill", action="store_true", help="different fill per row (fill - 37*b)")
    a = ap.parse_args()
    neuron_runtime_env(); dev = torch_device(); cf = compile_fn()
    H, D, BLK = a.H, 128, a.blk
    NBLK = a.slots // BLK
    dim = H * D
    torch.manual_seed(0)
    bf = torch.bfloat16
    n1 = torch.ones(1, dim, dtype=bf)
    wqkv = (torch.randn(dim, 3 * dim) * dim ** -0.5).to(bf)
    wo = (torch.randn(dim, dim) * dim ** -0.5).to(bf)
    sk = sv = 240.0 / 4.0      # amax ~4
    Bs = [int(b) for b in a.B.split(",")]
    BM = max(Bs)
    x_all = torch.randn(BM, 1, dim).to(bf)
    kv_all = (torch.randn(BM * NBLK, H, BLK, D) * 1.0).clamp(-3.9, 3.9)
    results = {}
    for B in Bs:
        x = x_all[:B]
        t = torch.full((B,), a.fill, dtype=torch.long)
        if a.varfill:
            t = (a.fill - 37 * torch.arange(B)).clamp(min=1)
        kv = kv_all
        blocks = torch.arange(B).view(B, 1) * NBLK + torch.arange(NBLK).view(1, NBLK)
        table = (blocks.view(B, 1, NBLK) * H + torch.arange(H).view(1, H, 1)).to(torch.int32)
        wslot = t % a.slots
        widx = (((torch.arange(B) * NBLK + wslot // BLK).view(B, 1) * H + torch.arange(H).view(1, H)) * BLK
                + (wslot % BLK).view(B, 1)).to(torch.int32).view(B, H, 1).to(torch.uint32)
        pos_ids = t.view(B, 1).float(); swa = torch.zeros(B, 1)
        ds = torch.arange(D // 2, dtype=torch.float32)
        ang = t.view(B, 1).float() * torch.exp(ds * (-torch.log(torch.tensor(10000.0)) * 2 / D)).view(1, -1)
        cos = torch.cos(ang).T.contiguous().view(D // 2, B, 1).to(bf); sin = torch.sin(ang).T.contiguous().view(D // 2, B, 1).to(bf)
        ks = torch.full((1, 1), sk, dtype=bf); vs = torch.full((1, 1), sv, dtype=bf)
        outs = {}
        for fp8 in (False, True):
            K = (kv * sk).to(torch.float8_e4m3fn) if fp8 else kv.to(bf)
            V = (kv * sv).to(torch.float8_e4m3fn) if fp8 else kv.to(bf)
            wo_ = (wo.float() / sv).to(bf) if fp8 else wo
            fn = cf(One(fp8, sk, sv).to(dev))
            Kd, Vd = K.to(dev), V.to(dev)
            with torch.inference_mode():
                o = fn(x.to(dev), n1.to(dev), wqkv.to(dev), wo_.to(dev), cos.to(dev), sin.to(dev), table.to(dev), Kd, Vd,
                       pos_ids.to(dev), swa.to(dev), widx.to(dev), ks.to(dev), vs.to(dev))
            outs[fp8] = (o.cpu().float().view(B, -1), Kd.cpu().float() / (sk if fp8 else 1.0))
        o0, o1 = outs[False][0], outs[True][0]
        results[B] = (o0, o1)
        if os.environ.get("RP_NOPROJ"):
            e = ((o1.view(B, H, -1) - o0.view(B, H, -1)).norm(dim=-1) / o0.view(B, H, -1).norm(dim=-1))
            bad = (e > 0.1).nonzero().tolist()
            print(f"  per-(b,h) bad folded rows: {[b * H + h for b, h in bad]}", flush=True)
        rel = (o1 - o0).norm(dim=-1) / o0.norm(dim=-1)
        # written slot check
        k0, k1 = outs[False][1], outs[True][1]
        wb = (torch.arange(B) * NBLK + wslot // BLK)
        wd = [float((k1[wb[b], :, wslot[b] % BLK] - k0[wb[b], :, wslot[b] % BLK]).abs().max()) for b in range(B)]
        print(f"H={H} B={B} (B*H={B * H}): attn out rel err per row {[round(float(r), 3) for r in rel]}; "
              f"written K max|diff| per row {[round(w, 3) for w in wd]}", flush=True)
    # cross-B consistency: rows of each B vs the same rows at the largest B (row results must not depend on B)
    for B in Bs:
        if B == BM:
            continue
        for k, name in ((0, "bf16"), (1, "fp8")):
            r = results[B][k]; ref = results[BM][k][:B]
            e = ((r - ref).norm(dim=-1) / ref.norm(dim=-1)).max()
            print(f"  {name} B={B} vs B={BM} rows: max rel diff {float(e):.4f}", flush=True)


if __name__ == "__main__":
    main()
