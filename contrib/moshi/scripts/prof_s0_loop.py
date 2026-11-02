"""Stage-0 step loop (MoshiStage0.step, real-time paced or back-to-back) on TPn ranks with part timers."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
os.environ.setdefault("MOSHI_S0_DECODE", "1")
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B, N = int(sys.argv[1]), int(sys.argv[2])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.engine import MoshiStage0
    from moshi_neuron.serving.common import compile_fn
    ncpu = os.cpu_count(); per = max(1, ncpu // world)
    os.sched_setaffinity(0, set(range(local * per % ncpu, local * per % ncpu + per))); torch.set_num_threads(per)
    s0 = MoshiStage0(B=B, device=torch.device(f"neuron:0"), compile_fn=compile_fn(), seed=7)
    x = torch.randn(B, 1920) * 0.05
    parts = {}
    with torch.inference_mode():
        for _ in range(10): s0.step(x)
        s0._prof = True; s0._pt = {}; s0._pn = 0
        t0 = time.perf_counter()
        for _ in range(N): s0.step(x)
        wall = (time.perf_counter() - t0) / N * 1e3
    if rank == 0:
        print(f"TP{world} B={B}: step {wall:.1f} ms; parts/step " + str({k: round(v / N * 1e3, 2) for k, v in s0._pt.items()}), flush=True)


if __name__ == "__main__":
    main()
