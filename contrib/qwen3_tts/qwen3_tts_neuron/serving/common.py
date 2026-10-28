"""Shared helpers for the Neuron vLLM-Omni workers."""
from __future__ import annotations

import os

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


def neuron_runtime_env():
    """Runtime env our single-process graphs need (vllm-neuron's NeuronWorker normally sets these)."""
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    # vllm-omni assigns stage devices through NEURON_VISIBLE_DEVICES (the platform's device_control_env_var);
    # the Neuron runtime requires NEURON_RT_VISIBLE_CORES alongside it. One logical core per stage process.
    vis = os.environ.get("NEURON_VISIBLE_DEVICES")
    if vis and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        os.environ["NEURON_RT_VISIBLE_CORES"] = vis.split(",")[0].strip()
    if vis:
        os.environ.pop("NEURON_VISIBLE_DEVICES", None)
    # distinct collective root port per core so two stage processes don't collide
    core = os.environ.get("NEURON_RT_VISIBLE_CORES", "0").split(",")[0].split("-")[0]
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{46820 + int(core)}")
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", "127.0.0.1:46820")
    os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "128")
    os.environ.setdefault("NEURON_RT_IO_RING_CACHE_SIZE", "32")
    # Host-side torch work per step is small (noise, index ops, prompt build); the default intra-op pool spins ~3
    # cores per stage process and starves the other stages on a 12-vCPU trn2.3xlarge. 2 threads measured best.
    if not cpu_mode():
        torch.set_num_threads(int(os.environ.get("QTTS_HOST_THREADS", "2")))


def cpu_mode() -> bool:
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1"


def torch_device():
    return torch.device("cpu") if cpu_mode() else torch.device("neuron:0")


def compile_fn():
    """torch.compile wrapper with the backend vllm-neuron uses (lite native), or identity in CPU mode."""
    if cpu_mode() or os.environ.get("QTTS_EAGER", "0") == "1":
        return lambda m: m
    from libtorch_neuronx_lite.compile.native_backend import register
    backend = register()
    torch._dynamo.config.cache_size_limit = 256
    return lambda m: torch.compile(m, backend=backend, dynamic=False)


def model_dtype():
    name = os.environ.get("QTTS_DTYPE", "float32" if cpu_mode() else "bfloat16")
    return getattr(torch, name)


def info_first(info: dict, key: str, default=None):
    """additional_information values arrive as lists (or tensors) per key."""
    v = info.get(key, default) if info else default
    if isinstance(v, (list, tuple)):
        return v[0] if v else default
    if isinstance(v, torch.Tensor):
        return v.reshape(-1)[0].item() if v.numel() else default
    return v
