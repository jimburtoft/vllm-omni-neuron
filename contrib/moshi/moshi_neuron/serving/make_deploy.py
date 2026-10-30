#!/usr/bin/env python3
"""Write a Neuron deploy YAML for Moshi from the installed vllm-omni `personaplex.yaml`.

env: MOSHI_B (sessions per stage-0 replica, default 1), MOSHI_DEV0 / MOSHI_DEV1 (cores, default 0 / 1),
     MOSHI_REP0 (stage-0 replicas; MOSHI_DEV0 must list one core each), MOSHI_WARMUP_FRAMES (default 0)."""
import os
import sys

import yaml
import vllm_omni

src = os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "personaplex.yaml")
d = yaml.safe_load(open(src))
B = int(os.environ.get("MOSHI_B", "1"))
r0 = int(os.environ.get("MOSHI_REP0", "1"))
d["model_runner"] = "v1"
d.pop("platforms", None)
d["duplex_session"] = dict(d.get("duplex_session") or {}, max_sessions=B * r0,
                           warmup_frames=int(os.environ.get("MOSHI_WARMUP_FRAMES", "0")))
cx = d["connectors"]["connector_of_shared_memory"]["extra"]
# every 80 ms frame goes to the decoder as soon as it exists (the 0.29 default buffers 5 frames = +320 ms latency)
cx.update(codec_chunk_frames=1, initial_codec_chunk_frames=1, codec_left_context_frames=0)
s0, s1 = d["stages"]
s0.update(worker_cls="moshi_neuron.serving.ar_worker.MoshiStage0Worker", async_scheduling=False, max_num_seqs=B,
          max_model_len=int(os.environ.get("MOSHI_MAX_MODEL_LEN", "3000")),
          devices=os.environ.get("MOSHI_DEV0", "0"), enforce_eager=True)
s1.update(worker_cls="moshi_neuron.serving.c2w_worker.MoshiCode2WavWorker", async_scheduling=False,
          max_num_seqs=int(os.environ.get("MOSHI_B1", str(B * r0))), devices=os.environ.get("MOSHI_DEV1", "1"),
          enforce_eager=True)
for s in (s0, s1):
    s.pop("gpu_memory_utilization", None)
if r0 > 1:
    devs = [x for x in s0["devices"].split(",") if x.strip()]
    if len(devs) != r0:
        raise SystemExit(f"MOSHI_DEV0 must list {r0} cores for {r0} replicas")
    s0["num_replicas"] = r0
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/moshi_neuron.yaml"
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
print(open(out).read())
