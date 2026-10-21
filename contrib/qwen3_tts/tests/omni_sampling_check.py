#!/usr/bin/env python3
"""Per-request sampling through vLLM-Omni (offline Omni API).

Batch mixes greedy and sampled (seeded) requests. Checks:
  * greedy requests produce identical codec frames to an all-greedy run (no cross-row interference)
  * sampled requests with the same seed reproduce identical audio across runs/batch positions
  * sampled requests with different seeds differ
  * max_tokens is honored
usage: test_sampling_omni.py <deploy_yaml>
"""
import copy
import hashlib
import json
import os
import sys

import torch

from qwen3_tts_neuron.serving.run_omni import CASES, MODEL, placeholder_len


def _inputs(cases):
    out = []
    for c in cases:
        info = {"task_type": ["VoiceDesign"], "text": [c["text"]], "language": [c["language"]],
                "instruct": [c["instruct"]], "max_new_tokens": [2048], "non_streaming_mode": [True]}
        out.append({"prompt_token_ids": [0] * placeholder_len(info), "additional_information": info})
    return out


def _run(omni, inputs, sps):
    res = {}
    for so in omni.generate(inputs, sps):
        a = so.outputs[0].multimodal_output["audio"]
        a = torch.cat(a, -1) if isinstance(a, list) else a
        idx = int(str(so.request_id).split("_")[0])
        res[idx] = a.float().cpu().reshape(-1)
    return [res[i] for i in range(len(inputs))]


def h(w):
    return hashlib.md5(w.numpy().tobytes()).hexdigest()[:10]


def main():
    from vllm import SamplingParams
    from vllm_omni import Omni
    omni = Omni(model=MODEL, deploy_config=sys.argv[1], init_timeout=3600, stage_init_timeout=3600)
    stage_sp = omni.default_sampling_params_list if hasattr(omni, "default_sampling_params_list") else None

    def sp(temp, top_k, seed=None, max_tokens=2048):
        p = SamplingParams(temperature=temp, top_k=top_k, repetition_penalty=1.05, min_tokens=2, max_tokens=max_tokens,
                           seed=seed, stop_token_ids=[2150], detokenize=False)
        c2w = SamplingParams(temperature=0.0, max_tokens=65536)
        return [p, c2w]

    c0, c1 = CASES[0], CASES[1]
    base = _run(omni, _inputs([c0, c1]), None)  # stage defaults (greedy YAML)
    greedy_sp = [sp(0.0, -1)[0], sp(0.0, -1)[1]]
    g_only = _run(omni, _inputs([c0, c1]), greedy_sp)
    # mixed batch: [greedy c0, sampled c1 seed 7, sampled c1 seed 7, sampled c1 seed 8, greedy c0 max_tokens=20]
    reqs = _inputs([c0, c1, c1, c1, c0])
    sps = [sp(0.0, -1), sp(0.9, 50, 7), sp(0.9, 50, 7), sp(0.9, 50, 8), sp(0.0, -1, max_tokens=20)]
    # Omni takes one sampling-params list per stage applied to all prompts; run requests individually batched by the
    # engine via concurrent generate is not exposed offline, so submit as separate calls but all at once via threads.
    import threading
    out = [None] * len(reqs)

    def go(i):
        out[i] = _run(omni, [reqs[i]], sps[i])[0]
    ts = [threading.Thread(target=go, args=(i,)) for i in range(len(reqs))]
    [t.start() for t in ts]; [t.join() for t in ts]
    r = dict(
        default_vs_explicit_greedy=[h(a) == h(b) for a, b in zip(base, g_only)],
        greedy_in_mixed_batch_equal=h(out[0]) == h(g_only[0]),
        same_seed_equal=h(out[1]) == h(out[2]),
        diff_seed_differ=h(out[1]) != h(out[3]),
        sampled_differs_from_greedy=h(out[1]) != h(g_only[1]),
        max_tokens_20_seconds=round(out[4].numel() / 24000, 2),
        durations=[round(o.numel() / 24000, 2) for o in out],
    )
    print("[sampling-omni]", json.dumps(r), flush=True)


if __name__ == "__main__":
    main()
