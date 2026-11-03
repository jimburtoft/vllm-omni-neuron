"""MOSHI_DEV_GATHER=1 vs 0 at TPn: identical decoded PCM (greedy, same inputs) and stage-0 loop step time."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local); os.environ["MOSHI_S0_DECODE"] = "1"
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B = int(sys.argv[1]); N = 60
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    import numpy as np
    from moshi_neuron.engine import MoshiStage0
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(2)
    s0 = MoshiStage0(B=B, device=torch.device("neuron:0"), compile_fn=compile_fn(), seed=7, greedy=True)
    s0.dev_gather = os.environ.get("MOSHI_DEV_GATHER", "1") == "1"
    wav = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy")))
    pcm, tm = [], []
    with torch.inference_mode():
        for t in range(N):
            x = wav[t * 1920:(t + 1) * 1920].view(1, -1).expand(B, -1).contiguous()
            t0 = time.perf_counter()
            s0.step(x)
            tm.append(time.perf_counter() - t0)
            pcm.append(s0.last_pcm.clone())
    if rank == 0:
        p = torch.stack(pcm)
        tag = int(s0.dev_gather)
        torch.save(p, os.path.expanduser(f"~/dg_{tag}.pt"))
        print(f"TP{world} B={B} dev_gather={tag}: step p50 {sorted(tm[10:])[len(tm[10:]) // 2] * 1e3:.1f} ms", flush=True)
        o = os.path.expanduser(f"~/dg_{1 - tag}.pt")
        if os.path.exists(o):
            q = torch.load(o)
            print(f"  max |pcm diff| vs other mode: {float((p - q).abs().max()):.2e}", flush=True)


if __name__ == "__main__":
    main()
