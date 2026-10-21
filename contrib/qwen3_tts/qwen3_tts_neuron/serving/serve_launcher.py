#!/usr/bin/env python3
"""`vllm-omni` launcher that keeps the API-server process off the Neuron runtime.

vllm-neuron's NeuronPlatform.device_count() initializes NRT (claiming every visible core) in whichever process calls it
first -- including the front end. Pre-setting the count makes it a pure lookup, so several servers can share a device,
each with stage processes pinned to disjoint cores via the deploy YAML `devices:`.
usage: QTTS_NC_COUNT=4 python serve_launcher.py serve <model> --deploy-config ... --omni"""
import os
import sys


def _preset():
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(int(os.environ.get("QTTS_NC_COUNT", "4")))


_preset()

if __name__ == "__main__":
    from vllm_omni.entrypoints.cli.main import main
    sys.argv[0] = "vllm-omni"
    sys.exit(main())
