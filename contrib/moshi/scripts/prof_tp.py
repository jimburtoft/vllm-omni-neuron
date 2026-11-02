"""TP decode benchmark + parity for the Moshi LM frame (one process per logical core).

  torchrun --nproc_per_node=2 prof_tp.py --B 8            # TP2 timing
  torchrun --nproc_per_node=2 prof_tp.py --parity 60      # TP2 teacher-forced parity vs fp32 golden (rank 0 reports)
MOSHI_TP is set from WORLD_SIZE.
"""
import argparse
import os
import sys
import time

rank = int(os.environ.get("RANK", "0"))
world = int(os.environ.get("WORLD_SIZE", "1"))
local = int(os.environ.get("LOCAL_RANK", "0"))
os.environ["MOSHI_TP"] = str(world)
base = int(os.environ.get("MOSHI_CORE0", "0"))
os.environ["NEURON_RT_VISIBLE_CORES"] = str(base + local)
os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")

import torch  # noqa: E402
import torch.distributed as dist  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--parity", type=int, default=0)
    ap.add_argument("--rows", type=int, default=0, help="staggered B-row teacher-forced parity (frames)")
    a = ap.parse_args()
    dist.init_process_group("gloo", rank=rank, world_size=world)
    if world > 1:
        from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
        rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.config import checkpoint_paths
    from moshi_neuron.lm import LMGenHost, MoshiFrame, MoshiLM
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(2)
    dev = torch.device("neuron:0")
    cf = compile_fn()
    lm_path, _, _ = checkpoint_paths()
    B = 1 if a.parity else a.B
    if a.rows:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tests"))
        from test_cpu_equivalence import _test_lm_rows
        lm = MoshiLM(lm_path, B).to(dev).eval()
        fn = cf(MoshiFrame(lm, ctx=lm.C, temp_text=0.0, temp=0.0, debug=True).to(dev))
        g = torch.load(os.path.expanduser("~/goldens/duplex_utt2_float32_greedy.pt"))
        with torch.inference_mode():
            res = _test_lm_rows(g, a.rows, B, fn, dev, stagger=7)
        if rank == 0:
            import numpy as np
            tt = [v["text_top1"] for v in res.values()]; dd = [v["dep_top1"] for v in res.values()]
            print(f"TP{world} rows B={B}: text top1 min {min(tt):.4f} mean {np.mean(tt):.4f} | dep top1 min {min(dd):.4f} "
                  f"mean {np.mean(dd):.4f}", flush=True)
        return
    lm = MoshiLM(lm_path, B).to(dev).eval()
    frame = MoshiFrame(lm, ctx=lm.C, temp_text=0.0, temp=0.0, debug=bool(a.parity)).to(dev)
    fn = cf(frame)
    z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
    with torch.inference_mode():
        if a.parity:
            g = torch.load(os.path.expanduser("~/goldens/duplex_utt1_float32_greedy.pt"))
            host = LMGenHost(1)
            ft, fa = g["text_logits"].argmax(-1), g["dep_logits"].argmax(-1)
            tt, dd, ms = [], [], []
            for t in range(a.parity):
                inp = host.inputs(g["user_codes"][t].view(1, -1))
                t0 = time.perf_counter()
                out, tl, al, _ = fn(inp.to(dev), torch.tensor([t == 0], device=dev), torch.ones(1, dtype=torch.bool, device=dev),
                                    z(1, 32000), z(1, 8, 2048), ft[t].view(1).to(dev), fa[t].view(1, 8).to(dev),
                                    torch.ones(1, dtype=torch.bool, device=dev))
                out = out.cpu(); tl = tl.cpu()[0]; al = al.cpu()[0]
                ms.append(time.perf_counter() - t0)
                host.commit(out)
                tt.append(int(tl.argmax() == ft[t])); dd.append(float((al.argmax(-1) == fa[t]).float().mean()))
            if rank == 0:
                import numpy as np
                print(f"TP{world} parity {a.parity} frames: text top1 {np.mean(tt):.4f} dep top1 {np.mean(dd):.4f} "
                      f"step {np.median(ms[3:]) * 1e3:.1f} ms", flush=True)
        else:
            tok = torch.full((B, 17), 5, dtype=torch.long).to(dev)
            r = z(B, dtype=torch.bool); act = ~r
            args = (tok, r, act, z(B, 32000), z(B, 8, 2048), z(B, dtype=torch.long), z(B, 8, dtype=torch.long), r)
            for _ in range(5):
                fn(*args).cpu()
            t0 = time.perf_counter()
            for _ in range(a.n):
                fn(*args).cpu()
            ms = (time.perf_counter() - t0) / a.n * 1e3
            if rank == 0:
                print(f"TP{world} B={B} LM frame {ms:.2f} ms", flush=True)
    dist.barrier() if world > 1 and False else None


if __name__ == "__main__":
    main()
