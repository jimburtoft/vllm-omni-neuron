"""Standalone stage-0 step time vs B (TP ranks via torchrun), table: B -> step ms per 80 ms frame.
usage: torchrun --nproc_per_node=4 s0_bench.py "1,8,16,24,32,40,48" N  (env as for serving: MOSHI_B = max B, buckets)."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
os.environ.setdefault("MOSHI_S0_DECODE", "1")
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    Bs = [int(x) for x in sys.argv[1].split(",")]
    N = int(sys.argv[2])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.engine import MoshiStage0
    from moshi_neuron.serving.common import compile_fn
    os.sched_setaffinity(0, {local * 3 % 12, local * 3 % 12 + 1, local * 3 % 12 + 2}); torch.set_num_threads(1)
    Bmax = int(os.environ.get("MOSHI_B", str(max(Bs))))
    s0 = MoshiStage0(B=Bmax, device=torch.device("neuron:0"), compile_fn=compile_fn(), seed=7,
                     greedy=os.environ.get("MOSHI_GREEDY", "0") == "1")
    x = torch.randn(Bmax, 1920) * 0.05
    s0.lm._abl = os.environ.get("MOSHI_ABL", "")
    if getattr(s0.lm, "pin_max", 0):
        s0.lm.pin_n.fill_(134)
    res = {}
    with torch.inference_mode():
        for B in Bs:
            for c in getattr(s0, "ctx_buckets", [None]):
                act = torch.zeros(Bmax, dtype=torch.bool); act[:B] = True
                s0.force_ctx = c
                for _ in range(5): s0.step(x, act)
                s0._prof = os.environ.get("MOSHI_PARTS", "0") == "1"; s0._pt = {}; s0._pn = 0
                t0 = time.perf_counter()
                for _ in range(N): s0.step(x, act)
                res[(B, c)] = (time.perf_counter() - t0) / N * 1e3
                if rank == 0 and s0._prof:
                    print(f"PARTS B={B}: " + str({k: round(v / N * 1e3, 2) for k, v in s0._pt.items()}), flush=True)
                s0._prof = False
                s0.reset(range(Bmax))
            s0.force_ctx = None
    if rank == 0:
        for (B, c), v in res.items():
            print(f"RESULT greedy={os.environ.get('MOSHI_GREEDY','0')} TP{world} B={B} ctx={c}: stage-0 step {v:.1f} ms (bucket {s0.bucket_for(torch.arange(Bmax) < B)})",
                  flush=True)


if __name__ == "__main__":
    main()
