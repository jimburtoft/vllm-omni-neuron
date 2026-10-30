"""CPU equivalence of moshi_neuron against goldens from the kyutai `moshi` reference (scripts/make_goldens.py).

  python test_cpu_equivalence.py --goldens ~/goldens --what mimi,lm [--dtype float32] [--frames 60]

Mimi:  stream-encode each clip and compare codes with the golden; stream-decode the golden codes and compare PCM.
LM:    teacher-forced over a duplex golden. Feed the golden user codes and force the golden generated tokens
       (greedy argmax of the golden logits), then compare text and depformer logits frame by frame, and check that
       the de-delayed output frames match the golden tokens.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from moshi_neuron.config import FRAME, checkpoint_paths  # noqa: E402
from moshi_neuron.lm import LMGenHost, MoshiFrame, MoshiLM  # noqa: E402
from moshi_neuron.mimi import MimiStreaming  # noqa: E402


def rel(a, b):
    return float((a.float() - b.float()).norm() / (b.float().norm() + 1e-12))


def snr(est, ref):
    n = min(len(est), len(ref))
    e, r = est[:n].double(), ref[:n].double()
    return float(10 * torch.log10((r ** 2).sum() / ((e - r) ** 2).sum().clamp(min=1e-20)))


def test_mimi(gdir, mimi_path, dtype, device="cpu", compile_fn=None):
    m = MimiStreaming(mimi_path, 1, device, dtype=dtype, compile_fn=compile_fn)
    res = {}
    for name in ("silence12", "utt1", "utt2"):
        wav = torch.from_numpy(np.load(os.path.join(gdir, f"clip_{name}.npy")))
        g = torch.load(os.path.join(gdir, f"mimi_{name}.pt"))
        T = wav.shape[0] // FRAME
        t0 = time.time()
        codes = []
        for t in range(T):
            codes.append(m.encode(wav[t * FRAME:(t + 1) * FRAME].view(1, -1), reset=[t == 0]).cpu())
        codes = torch.cat(codes, 0).t()  # [8,T]
        te = (time.time() - t0) / T
        match = (codes == g["codes"]).float()
        t0 = time.time()
        pcm = []
        for t in range(T):
            pcm.append(m.decode(g["codes"][:, t].view(1, -1), reset=[t == 0]).cpu().view(-1))
        pcm = torch.cat(pcm)
        td = (time.time() - t0) / T
        res[name] = dict(code_match=float(match.mean()), cb0_match=float(match[0].mean()),
                         first_mismatch=int((match.min(0).values < 1).nonzero()[0]) if match.min() < 1 else -1,
                         pcm_snr_db=snr(pcm, g["pcm"]), enc_ms=te * 1e3, dec_ms=td * 1e3)
        print("mimi", name, res[name], flush=True)
    return res


def test_lm(gdir, lm_path, dtype, clip, frames, device="cpu", compile_fn=None, golden_dtype="float32"):
    g = torch.load(os.path.join(gdir, f"duplex_{clip}_{golden_dtype}_greedy.pt"))
    T = min(frames, g["user_codes"].shape[0])
    lm = MoshiLM(lm_path, 1, dtype=dtype).to(device).eval()
    frame = MoshiFrame(lm, ctx=lm.C, temp_text=0.0, temp=0.0, debug=True).to(device)
    fn = (compile_fn or (lambda x: x))(frame)
    host = LMGenHost(1)
    forced_text = g["text_logits"].argmax(-1)       # [T]
    forced_audio = g["dep_logits"].argmax(-1)       # [T,8]
    z = lambda *s: torch.zeros(*s, device=device)
    stats = dict(text_rel=[], dep_rel=[], text_top1=[], dep_top1=[], out_match=[], ms=[])
    for t in range(T):
        inp = host.inputs(g["user_codes"][t].view(1, -1))
        t0 = time.time()
        out, tl, al, tout = fn(inp.to(device), torch.tensor([t == 0], device=device), torch.ones(1, dtype=torch.bool, device=device),
                               z(1, 32000), z(1, 8, 2048), forced_text[t].view(1).to(device),
                               forced_audio[t].view(1, 8).to(device), torch.ones(1, dtype=torch.bool, device=device))
        out = out.cpu()
        stats["ms"].append((time.time() - t0) * 1e3)
        tl, al = tl.cpu()[0], al.cpu()[0]
        stats["text_rel"].append(rel(tl, g["text_logits"][t]))
        stats["dep_rel"].append(rel(al, g["dep_logits"][t]))
        stats["text_top1"].append(int(tl.argmax() == forced_text[t]))
        stats["dep_top1"].append(float((al.argmax(-1) == forced_audio[t]).float().mean()))
        res, ready = host.commit(out)
        if bool(ready[0]):
            stats["out_match"].append(float((res[0] == g["tokens"][t]).float().mean()))
        elif int(g["tokens"][t][0]) != -2:
            stats["out_match"].append(0.0)
        if t % 20 == 0:
            print(f"  t={t} text_rel={stats['text_rel'][-1]:.2e} dep_rel={stats['dep_rel'][-1]:.2e} "
                  f"ms={stats['ms'][-1]:.0f}", flush=True)
    summ = {k: (float(np.mean(v)) if k != "text_rel" and k != "dep_rel" else float(np.max(v))) for k, v in stats.items()}
    summ["ms_median"] = float(np.median(stats["ms"][2:])) if T > 3 else summ["ms"]
    print("lm", clip, dtype, summ, flush=True)
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--goldens", default=os.path.expanduser("~/goldens"))
    ap.add_argument("--what", default="mimi,lm")
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--golden-dtype", default="float32")
    ap.add_argument("--clip", default="utt1")
    ap.add_argument("--frames", type=int, default=60)
    args = ap.parse_args()
    lm_path, mimi_path, _ = checkpoint_paths()
    dt = getattr(torch, args.dtype)
    if "mimi" in args.what:
        test_mimi(args.goldens, mimi_path, torch.float32)
    if "lm" in args.what:
        test_lm(args.goldens, lm_path, dt, args.clip, args.frames, golden_dtype=args.golden_dtype)


if __name__ == "__main__":
    with torch.no_grad():
        main()
