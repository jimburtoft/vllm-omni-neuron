"""MOSHI_FUSE_ENC=1 (Mimi encode fused into the LM frame graph) vs unfused: greedy tokens must be identical, and step
timing. TP1: python fuse_enc_test.py 4 ; TPn: torchrun ... fuse_enc_test.py 24"""
import os, sys, time
rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
if world > 1:
    os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def run(B, fuse, wav, N):
    os.environ["MOSHI_FUSE_ENC"] = "1" if fuse else "0"
    from moshi_neuron.engine import MoshiStage0
    from moshi_neuron.serving.common import compile_fn, torch_device
    dev = torch.device("neuron:0") if world > 1 else torch_device()
    s0 = MoshiStage0(B=B, device=dev, compile_fn=compile_fn(), seed=7, greedy=True)
    toks, tm = [], []
    with torch.inference_mode():
        for t in range(N):
            x = wav[t * 1920:(t + 1) * 1920].view(1, -1).expand(B, -1).contiguous()
            t0 = time.perf_counter()
            raw, out, ready, _ = s0.step(x)
            tm.append(time.perf_counter() - t0)
            toks.append(raw.clone())
    del s0
    return torch.stack(toks), sorted(tm[10:])[len(tm[10:]) // 2] * 1e3


def main():
    B = int(sys.argv[1]); N = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
        rendezvous_ccom_bootstrap()
    else:
        from moshi_neuron.serving.common import neuron_runtime_env
        neuron_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    import numpy as np
    torch.set_num_threads(2)
    wav = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy")))
    fuse = os.environ.get("FUSE", "0") == "1"
    a, ta = run(B, fuse, wav, N)
    if rank == 0:
        torch.save(a, os.path.expanduser(f"~/fe_tok_{world}_{int(fuse)}.pt"))
        print(f"TP{world} B={B} fuse={int(fuse)}: step {ta:.1f} ms", flush=True)
        other = os.path.expanduser(f"~/fe_tok_{world}_{1 - int(fuse)}.pt")
        if os.path.exists(other):
            b = torch.load(other)
            print(f"  greedy token rows differing vs other mode: {int((a != b).any(-1).sum())}/{a.shape[0] * a.shape[1]}")


if __name__ == "__main__":
    main()
