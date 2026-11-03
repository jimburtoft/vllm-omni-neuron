"""Depformer row-sharding at TPn: greedy tokens identical with MOSHI_DEP_SHARD=1 vs 0 (same frame inputs), all ranks
agree, and timing of the LM frame at B."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B = int(sys.argv[1]); ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.config import checkpoint_paths
    from moshi_neuron.lm import LMGenHost, MoshiFrame, MoshiLM
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(2)
    dev, cf = torch.device("neuron:0"), compile_fn()
    lm = MoshiLM(checkpoint_paths()[0], B).to(dev).eval()
    g = torch.load(os.path.expanduser("~/goldens/duplex_utt2_float32_greedy.pt"))
    z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
    res, tim = {}, {}
    for shard in (1, 0):
        lm.dep_shard = bool(shard)
        fn = cf(MoshiFrame(lm, ctx=ctx, temp_text=0.0, temp=0.0).to(dev))
        host = LMGenHost(B); toks = []
        with torch.inference_mode():
            for t in range(40):
                inp = host.inputs(g["user_codes"][t].view(1, -1).expand(B, -1).contiguous())
                out = fn(inp.to(dev), torch.full((B,), t == 0, device=dev), torch.ones(B, dtype=torch.bool, device=dev),
                         z(B, 32000), z(B, 8, 1), z(B, dtype=torch.long), z(B, 8, dtype=torch.long), z(B, dtype=torch.bool)).cpu()
                host.commit(out); toks.append(out)
            t0 = time.perf_counter()
            for _ in range(20):
                fn(inp.to(dev), torch.zeros(B, dtype=torch.bool, device=dev), torch.ones(B, dtype=torch.bool, device=dev),
                   z(B, 32000), z(B, 8, 1), z(B, dtype=torch.long), z(B, 8, dtype=torch.long), z(B, dtype=torch.bool)).cpu()
            tim[shard] = (time.perf_counter() - t0) / 20 * 1e3
        res[shard] = torch.stack(toks)
        lm.offset.copy_(torch.zeros_like(lm.offset))
    if rank == 0:
        d = int((res[1] != res[0]).any(-1).sum())
        print(f"TP{world} B={B} ctx={ctx}: dep_shard LM frame {tim[1]:.1f} ms vs replicated {tim[0]:.1f} ms; "
              f"greedy token rows differing (shard vs not) {d}/{40 * B}", flush=True)


if __name__ == "__main__":
    main()
