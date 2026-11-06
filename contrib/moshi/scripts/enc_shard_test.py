"""MOSHI_ENC_SHARD parity + timing: per-rank row-sharded Mimi encode + in-graph all-gather vs all-rows encode on
every rank. torchrun --nproc_per_node=4 enc_shard_test.py B N"""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B, N = int(sys.argv[1]), int(sys.argv[2])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.mimi import MimiStreaming
    from moshi_neuron.config import checkpoint_paths
    from moshi_neuron.serving.common import compile_fn
    from moshi_neuron.engine import _EncGather
    _, mimi_path, _ = checkpoint_paths()
    dev = torch.device("neuron:0"); cf = compile_fn()
    full = MimiStreaming(mimi_path, B, dev, compile_fn=cf, decoder=False)
    part = MimiStreaming(mimi_path, B // world, dev, compile_fn=lambda m: m, decoder=False)
    g = cf(_EncGather(part.enc))
    torch.manual_seed(0)
    import numpy as np
    clip = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy"))).float()
    sel = torch.arange(rank, B, world)
    diff = tot = 0
    t_full = t_sh = 0.0
    with torch.inference_mode():
        for f in range(N):
            # each row = the clip at a different offset, so rows differ
            pcm = torch.stack([clip[((f + 7 * b) % 300) * 1920:((f + 7 * b) % 300 + 1) * 1920] for b in range(B)])
            reset = torch.zeros(B, dtype=torch.bool) if f else torch.ones(B, dtype=torch.bool)
            act = torch.ones(B, dtype=torch.bool)
            t0 = time.perf_counter()
            a = full.encode(pcm, reset=reset, active=act).cpu()
            t1 = time.perf_counter()
            cg = g(pcm[sel].to(dev).view(sel.numel(), 1, -1), reset[sel].to(dev), act[sel].to(dev)).cpu()
            t2 = time.perf_counter()
            nb = sel.numel()
            b_ = torch.zeros_like(a)
            for q in range(world):
                b_[torch.arange(q, B, world)] = cg[q * nb:(q + 1) * nb]
            if f >= 3:
                t_full += t1 - t0; t_sh += t2 - t1
            diff += int((a != b_).sum()); tot += a.numel()
    if rank == 0:
        print(f"ENC B={B}: codes differ {diff}/{tot}; all-rows {t_full / (N - 3) * 1e3:.2f} ms, "
              f"sharded+gather {t_sh / (N - 3) * 1e3:.2f} ms", flush=True)


if __name__ == "__main__":
    main()
