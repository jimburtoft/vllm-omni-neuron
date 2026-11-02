"""Do all TP ranks produce identical sampled tokens? (They must: each rank decodes Mimi for its own rows.)"""
import os, sys
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B = int(sys.argv[1]); N = int(sys.argv[2])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.engine import MoshiStage0
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(2)
    s0 = MoshiStage0(B=B, device=torch.device("neuron:0"), compile_fn=compile_fn(), seed=7)
    import numpy as np
    wav = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy")))
    mism = 0
    with torch.inference_mode():
        for t in range(N):
            x = wav[t * 1920:(t + 1) * 1920].view(1, -1).expand(B, -1).contiguous()
            raw, out, ready, _ = s0.step(x)
            g = [torch.empty_like(raw) for _ in range(world)]
            dist.all_gather(g, raw)
            d = sum(int((g[0] != gi).any(1).sum()) for gi in g[1:])
            mism += d
            if rank == 0 and (d or t % 50 == 0):
                print(f"t={t} rows differing across ranks: {d}", flush=True)
    if rank == 0:
        print(f"TOTAL mismatching (rank,row,step) = {mism}", flush=True)


if __name__ == "__main__":
    main()
