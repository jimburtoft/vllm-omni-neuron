"""Runtime helpers shared by the Neuron workers and test drivers (same conventions as contrib/qwen3_tts)."""
from __future__ import annotations

import os

import torch


def cpu_mode() -> bool:
    return os.environ.get("VLLM_NEURON_CPU_MODE", "0") == "1"


def neuron_runtime_env():
    """Runtime environment for single-process lite graphs (one logical core per process unless set)."""
    os.environ.setdefault("NEURON_LOGICAL_NC_CONFIG", "2")
    vis = os.environ.get("NEURON_VISIBLE_DEVICES")
    if vis and not os.environ.get("NEURON_RT_VISIBLE_CORES"):
        os.environ["NEURON_RT_VISIBLE_CORES"] = vis.split(",")[0].strip()
    if vis:
        os.environ.pop("NEURON_VISIBLE_DEVICES", None)
    os.environ.setdefault("NEURON_RT_VISIBLE_CORES", "0")
    core = os.environ["NEURON_RT_VISIBLE_CORES"].split(",")[0].split("-")[0]
    os.environ.setdefault("NEURON_RT_ROOT_COMM_ID", f"127.0.0.1:{46920 + int(core)}")
    os.environ.setdefault("NEURON_RT_XU_COMPUTE_MAX_QUEUED_REQUESTS", "128")
    if not cpu_mode():
        torch.set_num_threads(int(os.environ.get("MOSHI_HOST_THREADS", "2")))


def torch_device():
    if cpu_mode():
        return torch.device("cpu")
    import libtorch_neuronx_lite  # noqa: F401  (registers the "neuron" device)
    return torch.device("neuron:0")


def compile_fn():
    """torch.compile with the lite native backend (the one vllm-neuron uses); identity in CPU / eager mode."""
    if cpu_mode() or os.environ.get("MOSHI_EAGER", "0") == "1":
        return lambda m: m
    from libtorch_neuronx_lite.compile.native_backend import register
    backend = register()
    torch._dynamo.config.cache_size_limit = 256
    return lambda m: torch.compile(m, backend=backend, dynamic=False, fullgraph=True)


def model_dir(name_or_path: str) -> str | None:
    """Local snapshot directory for the served model (repo id or path); None if it can't be resolved offline."""
    if name_or_path and os.path.isdir(name_or_path):
        return name_or_path
    try:
        from huggingface_hub import snapshot_download
        return snapshot_download(name_or_path, allow_patterns=["*.safetensors", "*.model", "*.json"])
    except Exception:
        return None
