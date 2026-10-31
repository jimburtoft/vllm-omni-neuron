#!/usr/bin/env python3
"""`vllm-omni` launcher for Moshi on Neuron.

1. Pre-sets vllm-neuron's device count so the API-server process never initializes the Neuron runtime (it would
   claim every visible core); stage processes are pinned to cores by the deploy YAML `devices:`.
2. Applies the moshi backports (serving adapter) in the API-server process.
usage: MOSHI_NC_COUNT=4 python -m moshi_neuron.serving.serve_launcher serve <model> --deploy-config ... --omni
"""
import os
import sys


def _preset():
    # stage-0 max_model_len caps the session length (one token per 80 ms frame), above Helium's 3000-frame context;
    # the Neuron worker attends a 3000-frame ring, so the longer length is safe.
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(int(os.environ.get("MOSHI_NC_COUNT", "4")))
    from .backports import apply
    print("[moshi] backports:", apply(), flush=True)


if __name__ == "__main__":
    _preset()
    from vllm_omni.entrypoints.cli.main import main
    sys.argv[0] = "vllm-omni"
    sys.exit(main())
