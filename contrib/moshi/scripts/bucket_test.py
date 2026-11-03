"""Batch buckets: (1) parity -- greedy run with a session in row 0 while other rows idle, bucketed vs full-B graphs
must give identical tokens; (2) step time per bucket. Works at TP1 (python) or TPn (torchrun)."""
import os, sys, time
rank = int(os.environ.get("RANK", "0")); world = int(os.environ.get("WORLD_SIZE", "1")); local = int(os.environ.get("LOCAL_RANK", "0"))
if world > 1:
    os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
os.environ.setdefault("MOSHI_S0_DECODE", "1")
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B = int(sys.argv[1]); buckets = sys.argv[2]; N = int(sys.argv[3]) if len(sys.argv) > 3 else 40
    if world > 1:
        dist.init_process_group("gloo", rank=rank, world_size=world)
        from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
        rendezvous_ccom_bootstrap()
    from moshi_neuron.serving.common import compile_fn, neuron_runtime_env, torch_device
    if world == 1:
        neuron_runtime_env()
    import libtorch_neuronx_lite  # noqa: F401
    import numpy as np
    from moshi_neuron.engine import MoshiStage0
    torch.set_num_threads(2)
    dev = torch.device("neuron:0") if world > 1 else torch_device()
    os.environ["MOSHI_BUCKETS"] = buckets
    s0 = MoshiStage0(B=B, device=dev, compile_fn=compile_fn(), seed=7, greedy=True)
    wav = torch.from_numpy(np.load(os.path.expanduser("~/goldens/clip_utt2.npy")))
    res = {}
    with torch.inference_mode():
        # timing per bucket (all rows of the bucket active)
        for bb in s0.buckets:
            act = torch.zeros(B, dtype=torch.bool); act[:bb] = True
            x = torch.randn(B, 1920) * 0.05
            for _ in range(5): s0.step(x, act)
            t0 = time.perf_counter()
            for _ in range(N): s0.step(x, act)
            res[bb] = (time.perf_counter() - t0) / N * 1e3
            s0.reset(range(B))
        # parity: row 0 only (smallest bucket) vs all rows active (largest bucket), same clip, greedy
        def run(active_rows):
            s0.reset(range(B))
            toks = []
            act = torch.zeros(B, dtype=torch.bool); act[:active_rows] = True
            for t in range(60):
                x = wav[t * 1920:(t + 1) * 1920].view(1, -1).expand(B, -1).contiguous()
                raw, out, ready, _ = s0.step(x, act)
                toks.append(raw[0].clone())
            return torch.stack(toks), s0.last_pcm[0].clone() if s0.decode_here else None
        a, _ = run(1)
        b, _ = run(B)
        c, _ = run(1)   # repeat: is the bucket-1 path itself deterministic?
        # also: B-row graph with only row 0 active (same graph as b, different masking)
        os.environ["MOSHI_BUCKETS_OFF"] = "1"
        s0.buckets = [B]
        d, _ = run(1)
        s0.buckets = sorted({int(x) for x in buckets.split(",")} | {B})
    if rank == 0:
        print(f"TP{world} B={B} buckets {s0.buckets}: step ms " + ", ".join(f"{k}:{v:.1f}" for k, v in res.items()), flush=True)
        f = lambda x, y: int((x != y).any(1).sum())
        print(f"row0 greedy tokens, frames differing /60: bucket1 vs fullB(all active) {f(a, b)}, bucket1 vs bucket1 {f(a, c)}, "
              f"fullB(row0 only) vs fullB(all active) {f(d, b)}, bucket1 vs fullB(row0 only) {f(a, d)}", flush=True)


if __name__ == "__main__":
    main()
