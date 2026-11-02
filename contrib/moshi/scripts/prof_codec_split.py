"""Per-rank breakdown of the codec around the LM frame at TPn: mimi.encode graph, .cpu(), gloo all_gather, decode."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    Bl = int(sys.argv[1])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.config import checkpoint_paths
    from moshi_neuron.mimi import MimiStreaming
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(1)
    dev, cf = torch.device("neuron:0"), compile_fn()
    _, mp, _ = checkpoint_paths()
    m = MimiStreaming(mp, Bl, dev, compile_fn=cf)
    x = torch.randn(Bl, 1920) * 0.1; c = torch.randint(0, 2048, (Bl, 8))
    with torch.inference_mode():
        for _ in range(5): m.encode(x).cpu(); m.decode(c).cpu()
        N = 50; a = b = g = d = 0.0
        for _ in range(N):
            t0 = time.perf_counter(); y = m.encode(x); t1 = time.perf_counter(); y = y.cpu(); t2 = time.perf_counter()
            parts = [torch.empty_like(y) for _ in range(world)]; dist.all_gather(parts, y); t3 = time.perf_counter()
            m.decode(c).cpu(); t4 = time.perf_counter()
            a += t1 - t0; b += t2 - t1; g += t3 - t2; d += t4 - t3
    if rank == 0:
        print(f"world {world} rows/rank {Bl}: encode-call {a/N*1e3:.2f} ms, d2h {b/N*1e3:.2f}, gloo gather {g/N*1e3:.2f}, "
              f"decode(+d2h) {d/N*1e3:.2f}", flush=True)


if __name__ == "__main__":
    main()
