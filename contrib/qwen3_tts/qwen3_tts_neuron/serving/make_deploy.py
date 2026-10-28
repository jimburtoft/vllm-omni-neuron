#!/usr/bin/env python3
"""Write a Neuron deploy YAML for Qwen3-TTS derived from the installed vllm-omni qwen3_tts.yaml."""
import os, sys, yaml, vllm_omni
src = os.path.join(os.path.dirname(vllm_omni.__file__), "deploy", "qwen3_tts.yaml")
d = yaml.safe_load(open(src))
B = int(os.environ.get("QTTS_B", "8"))
greedy = os.environ.get("QTTS_GREEDY", "0") == "1"
d["model_runner"] = "v1"
d.pop("platforms", None)
s0, s1 = d["stages"]
s0.update(worker_cls="qwen3_tts_neuron.serving.ar_worker.NeuronTTSARWorker", async_scheduling=False,
          max_num_seqs=B, max_num_batched_tokens=4096, max_model_len=int(os.environ.get("QTTS_SLOT_LEN", "2048")),
          devices=os.environ.get("QTTS_DEV0", "0"), enforce_eager=True)
s1.update(worker_cls="qwen3_tts_neuron.serving.c2w_worker.NeuronCode2WavWorker", async_scheduling=False,
          max_num_seqs=B, devices=os.environ.get("QTTS_DEV1", "1"), enforce_eager=True)
for s in (s0, s1):
    s.pop("gpu_memory_utilization", None)
    # qwen3_tts's HF config carries a dummy vision_config, which makes vllm-neuron's platform auto-configure
    # VLM encoder buckets; give it an explicit (unused) bucket so the check passes.
    s["additional_config"] = {"encoder_neuron_config": {"num_encoder_input_buckets": [2048],
                                                        "encoder_attention_block_size": 2048}}
r0, r1 = int(os.environ.get("QTTS_REP0", "1")), int(os.environ.get("QTTS_REP1", "1"))
# Replicated stages need one explicit core per replica (vllm-omni "pool" device mode). A single core id with
# num_replicas > 1 is read as a per-replica *template* and replica r gets core r, which lands talker replicas on
# the vocoder's core (or fails NRT init). Default layout: talkers on 0,2,3,... and the vocoder on 1.
for st, r, key in ((s0, r0, "QTTS_DEV0"), (s1, r1, "QTTS_DEV1")):
    devs = [d for d in str(st["devices"]).split(",") if d.strip()]
    if r > 1:
        st["num_replicas"] = r
        if len(devs) != r:
            raise SystemExit(f"{key} must list {r} cores for {r} replicas (got {st['devices']!r}), e.g. "
                             f"QTTS_REP0=3 QTTS_DEV0=0,2,3 QTTS_DEV1=1")
used = [d.strip() for st in (s0, s1) for d in str(st["devices"]).split(",")]
if len(set(used)) != len(used):
    raise SystemExit(f"stages share a core: {used}")
if os.environ.get("QTTS_B1"):
    s1["max_num_seqs"] = int(os.environ["QTTS_B1"])
if greedy:
    s0["default_sampling_params"].update(temperature=0.0, top_k=-1)
    s0["subtalker_sampling_params"].update(do_sample=False)
out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3_tts_neuron.yaml"
yaml.safe_dump(d, open(out, "w"), sort_keys=False)
print(open(out).read())
