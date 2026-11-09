"""Frame-synchronous batching gate for Moshi stage 0 (MOSHI_FRAMESYNC=1).

vLLM's engine core runs a step as soon as any session's 80 ms frame is waiting, so sessions with different phase
land in different steps (several partial steps per tick; on Neuron each costs about a full step). This gate wraps
EngineCore.step in the stage-0 engine-core process and holds waiting frames until either every recently active
session has a frame ready or MOSHI_FS_PERIOD_MS (default 76) has passed since the previous gated step started,
i.e. about one step per tick that carries every session. Costs up to one tick of alignment latency.

Installed from `scripts/fs_site/sitecustomize.py` (on PYTHONPATH) in every process; it activates only in the engine
core whose worker is the Moshi stage-0 worker.
"""
from __future__ import annotations

import logging
import os
import time

logger = logging.getLogger("moshi.framesync")


def _patch(core_mod) -> bool:
    EC = core_mod.EngineCore
    if getattr(EC, "_moshi_fs", False):
        return False
    EC._moshi_fs = True
    orig = EC.step
    period = float(os.environ.get("MOSHI_FS_PERIOD_MS", "76")) / 1e3
    win = float(os.environ.get("MOSHI_FS_ACTIVE_S", "0.5"))
    dbg = os.environ.get("MOSHI_FS_DEBUG") == "1"

    def step(self):
        st = self.__dict__.get("_fs")
        if st is None:
            vc = getattr(self, "vllm_config", None)
            wc = str(getattr(getattr(vc, "parallel_config", None), "worker_cls", "") or "")
            st = self._fs = dict(on="ar_worker" in wc, seen={}, last=0.0, n=0, held=0)
            logger.warning("[MoshiFS] frame-sync gate %s (period %.0f ms, worker_cls %s, pid %d)",
                           "ON" if st["on"] else "off", period * 1e3, wc, os.getpid())
        if not st["on"]:
            return orig(self)
        from vllm.v1.request import RequestStatus as RS
        sched = self.scheduler
        ready = {rid for rid, r in sched.requests.items()
                 if r.status in (RS.WAITING, RS.RUNNING) and r.num_computed_tokens < r.num_tokens}
        if not ready:
            return orig(self)
        now = time.perf_counter()
        seen = st["seen"]
        for r in [r for r, t in seen.items() if now - t > win]:
            del seen[r]
        # MOSHI_FS_MIN_FRAC (default 1.0): fire as soon as this fraction of the recently active sessions has a
        # frame ready (stragglers ride the next tick) instead of waiting for all or the full period
        frac = float(os.environ.get("MOSHI_FS_MIN_FRAC", "1.0"))
        have = len(seen.keys() & ready)
        enough = (seen.keys() <= ready) or (frac < 1.0 and seen and have >= frac * len(seen))
        if not enough and now - st["last"] < period:
            st["held"] += 1
            return {}, False
        st["last"] = now
        out = orig(self)
        for r in ready:
            seen[r] = now
        st["n"] += 1
        if dbg and st["n"] % 250 == 0:
            from collections import Counter
            logger.warning("[MoshiFS] %d steps, %d held polls, %d active, %d ready; statuses %s", st["n"], st["held"],
                           len(seen), len(ready), dict(Counter(str(r.status) for r in sched.requests.values())))
        return out

    EC.step = step
    return True


def install() -> bool:
    if os.environ.get("MOSHI_FRAMESYNC", "0") != "1":
        return False
    import vllm.v1.engine.core as core
    return _patch(core)


def install_import_hook():
    """Patch vllm.v1.engine.core as soon as it is imported (any process)."""
    import importlib.abc
    import importlib.util
    import sys

    if os.environ.get("MOSHI_FRAMESYNC", "0") != "1" and not os.environ.get("MOSHI_PHASES"):
        return

    def _all(mod):
        if os.environ.get("MOSHI_FRAMESYNC", "0") == "1":
            _patch(mod)
        if os.environ.get("MOSHI_PHASES"):
            from .phases import install as _ph
            _ph()
    if "vllm.v1.engine.core" in sys.modules:
        _all(sys.modules["vllm.v1.engine.core"])
        return

    class _Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname != "vllm.v1.engine.core":
                return None
            sys.meta_path.remove(self)
            spec = importlib.util.find_spec(fullname)
            if spec is None or spec.loader is None:
                return spec
            orig_exec = spec.loader.exec_module

            def exec_module(module):
                orig_exec(module)
                try:
                    _all(module)
                except Exception as e:  # noqa: BLE001
                    sys.stderr.write(f"[MoshiFS] patch failed: {e}\n")
            spec.loader.exec_module = exec_module
            return spec

    sys.meta_path.insert(0, _Finder())
