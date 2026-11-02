"""TP4 full-frame timing: LM frame (TP4) + Mimi encode of B/4 rows + Mimi decode of B/4 rows on EVERY rank
(row-sharded codec; codes would be all-gathered -- measured separately), i.e. the per-rank critical path if the
codec is sharded across the TP ranks instead of a separate decoder core."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def main():
    B = int(sys.argv[1])
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
    rendezvous_ccom_bootstrap()
    import libtorch_neuronx_lite  # noqa: F401
    from moshi_neuron.config import checkpoint_paths
    from moshi_neuron.lm import MoshiFrame, MoshiLM
    from moshi_neuron.mimi import MimiStreaming
    from moshi_neuron.serving.common import compile_fn
    torch.set_num_threads(2)
    dev, cf = torch.device("neuron:0"), compile_fn()
    lm_path, mimi_path, _ = checkpoint_paths()
    Bl = B // world
    mimi = MimiStreaming(mimi_path, Bl, dev, compile_fn=cf)
    lm = MoshiLM(lm_path, B).to(dev).eval()
    fn = cf(MoshiFrame(lm, ctx=lm.C).to(dev))
    z = lambda *s, **k: torch.zeros(*s, device=dev, **k)
    tok = torch.full((B, 17), 5, dtype=torch.long).to(dev); r = z(B, dtype=torch.bool); act = ~r
    args = (tok, r, act, z(B, 32000), z(B, 8, 2048), z(B, dtype=torch.long), z(B, 8, dtype=torch.long), r)
    x = torch.randn(Bl, 1920) * 0.1; c = torch.randint(0, 2048, (Bl, 8))
    with torch.inference_mode():
        for _ in range(5):
            mimi.encode(x).cpu(); fn(*args).cpu(); mimi.decode(c).cpu()
        te = tl = td = 0.0; N = 30
        for _ in range(N):
            t0 = time.perf_counter(); codes = mimi.encode(x).cpu()
            g = [torch.zeros_like(codes) for _ in range(world)]; dist.all_gather(g, codes)   # host gather (gloo)
            t1 = time.perf_counter(); fn(*args).cpu()
            t2 = time.perf_counter(); mimi.decode(c).cpu(); t3 = time.perf_counter()
            te += t1 - t0; tl += t2 - t1; td += t3 - t2
        if rank == 0:
            print(f"TP{world} B={B}: enc({Bl} rows)+gather {te/N*1e3:.1f} ms, LM {tl/N*1e3:.1f} ms, dec({Bl} rows) {td/N*1e3:.1f} ms, "
                  f"total {(te+tl+td)/N*1e3:.1f} ms", flush=True)


if __name__ == "__main__":
    main()
