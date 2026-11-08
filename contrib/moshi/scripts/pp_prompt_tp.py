"""TP4 standalone: prefill_prompt on row 0 (bucket 24), then 30 serving steps with all rows active; prints timing."""
import os, sys, time
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"]); local = int(os.environ["LOCAL_RANK"])
os.environ["MOSHI_TP"] = str(world); os.environ["NEURON_RT_VISIBLE_CORES"] = str(local)
import torch, torch.distributed as dist
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
dist.init_process_group("gloo", rank=rank, world_size=world)
from vllm_neuron.vllm.worker.neuron_worker import rendezvous_ccom_bootstrap
rendezvous_ccom_bootstrap()
import libtorch_neuronx_lite  # noqa
from moshi_neuron.engine import MoshiStage0
from moshi_neuron.serving.common import compile_fn
B = int(os.environ.get("MOSHI_B", "24"))
s0 = MoshiStage0(B=B, device=torch.device("neuron:0"), compile_fn=compile_fn(), seed=7, model_dir=os.environ.get("PPDIR"))
x = torch.randn(B, 1920) * 0.05
act = torch.ones(B, dtype=torch.bool)
with torch.inference_mode():
    for _ in range(3): s0.step(x, act)
    for row in (0, 1, 2):
        t = time.perf_counter(); P = s0.prefill_prompt(row, "NATM1.pt", "You work for a restaurant and your name is Owen."); dt = time.perf_counter() - t
        if rank == 0: print(f"PROMPT row {row}: {P} steps in {dt*1e3:.0f} ms", flush=True)
    t = time.perf_counter()
    for _ in range(30): s0.step(x, act)
    if rank == 0: print(f"STEPS 30 in {(time.perf_counter()-t)*1e3/30:.1f} ms/step", flush=True)
