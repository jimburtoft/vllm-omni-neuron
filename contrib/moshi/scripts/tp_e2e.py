"""TP standalone end-to-end with the row-sharded codec: wav in -> wav out per row on rank 0 (for ASR checks)."""
import os, sys
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
os.environ["MOSHI_S0_DECODE"] = "1"
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B = int(sys.argv[1]); out = sys.argv[2]; greedy = len(sys.argv) > 3 and sys.argv[3] == "greedy"
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    import numpy as np, soundfile as sf
    from moshi_neuron.engine import MoshiStage0
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(2)
    s0 = MoshiStage0(B=B, device=torch.device("neuron:0"), compile_fn=compile_fn(), seed=7, greedy=greedy)
    wav = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy")))
    T = wav.shape[0] // 1920
    pcm = [[] for _ in range(B)]
    with torch.inference_mode():
        for t in range(T):
            x = wav[t * 1920:(t + 1) * 1920].view(1, -1).expand(B, -1).contiguous()
            raw, o, ready, _ = s0.step(x)
            for b in range(B):
                if bool(ready[b]):
                    pcm[b].append(s0.last_pcm[b].clone())
    if rank == 0:
        os.makedirs(out, exist_ok=True)
        for b in range(B):
            sf.write(os.path.join(out, f"row{b}.wav"), torch.cat(pcm[b]).numpy(), 24000)
        print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
