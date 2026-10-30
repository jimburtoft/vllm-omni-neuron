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
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(int(os.environ.get("MOSHI_NC_COUNT", "4")))
    from .backports import apply
    print("[moshi] backports:", apply(), flush=True)


if __name__ == "__main__":
    _preset()
    from vllm_omni.entrypoints.cli.main import main
    sys.argv[0] = "vllm-omni"
    sys.exit(main())
