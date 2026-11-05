"""Phase timing for the Moshi stage-0 engine core (MOSHI_PHASES=<path prefix>). Debug aid only.

Wraps, inside the stage-0 EngineCore process, the pieces of one engine iteration and writes one line per executed
step to <prefix>.ec.<pid>:
  t_loop_top  input_ms  sched_ms  bcast_ms  wait_ms  update_ms  post_ms  nreq  pkl_bytes
where
  input_ms  = _process_input_queue (client requests -> scheduler), measured per loop iteration and summed since the
              previous executed step (includes any idle/blocked time)
  sched_ms  = scheduler.schedule()
  bcast_ms  = executor.execute_model() call itself (pickle + shm enqueue to all TP workers; non_block)
  wait_ms   = future.result() (worker compute + response dequeue/unpickle from rank 0)
  update_ms = scheduler.update_from_output()
  post_ms   = rest of _process_engine_step (output_queue.put + post_step)
Worker side (MoshiStage0Runner) already traces s0_enter/s0_exit (MOSHI_TRACE).
"""
from __future__ import annotations

import os
import pickle
import time


def install():
    pfx = os.environ.get("MOSHI_PHASES")
    if not pfx:
        return False
    import vllm.v1.engine.core as core
    EC, ECP = core.EngineCore, core.EngineCoreProc
    if getattr(EC, "_moshi_ph", False):
        return False
    EC._moshi_ph = True
    st = dict(f=None, on=None, inp=0.0, cur=None)

    def on(self):
        if st["on"] is None:
            wc = str(getattr(self.vllm_config.parallel_config, "worker_cls", "") or "")
            st["on"] = "ar_worker" in wc
            if st["on"]:
                st["f"] = open(f"{pfx}.ec.{os.getpid()}", "a", buffering=1)
        return st["on"]

    orig_step = EC.step

    def step(self):
        if not on(self):
            return orig_step(self)
        sched, ex = self.scheduler, self.model_executor
        cur = st["cur"] = dict(t0=time.perf_counter(), sched=0.0, bcast=0.0, wait=0.0, update=0.0, n=0, pk=0)
        o_s, o_e, o_u = sched.schedule, ex.execute_model, sched.update_from_output

        def s_(*a, **k):
            t = time.perf_counter(); r = o_s(*a, **k); cur["sched"] += time.perf_counter() - t
            cur["n"] = len(getattr(r, "num_scheduled_tokens", {}) or {})
            if os.environ.get("MOSHI_PHASES_PKL") == "1":
                cur["pk"] = len(pickle.dumps(r, protocol=pickle.HIGHEST_PROTOCOL))
            return r

        def e_(*a, **k):
            t = time.perf_counter(); fut = o_e(*a, **k); cur["bcast"] += time.perf_counter() - t
            if hasattr(fut, "result"):
                orr = fut.result

                def res(*ra, **rk):
                    t2 = time.perf_counter(); v = orr(*ra, **rk); cur["wait"] += time.perf_counter() - t2
                    return v
                fut.result = res
            return fut

        def u_(*a, **k):
            t = time.perf_counter(); r = o_u(*a, **k); cur["update"] += time.perf_counter() - t
            return r

        sched.schedule, ex.execute_model, sched.update_from_output = s_, e_, u_
        try:
            return orig_step(self)
        finally:
            sched.schedule, ex.execute_model, sched.update_from_output = o_s, o_e, o_u
            cur["t1"] = time.perf_counter()

    EC.step = step

    orig_pes = ECP._process_engine_step

    def pes(self):
        if not on(self):
            return orig_pes(self)
        st["cur"] = None
        r = orig_pes(self)
        cur = st["cur"]
        if cur is not None and cur["n"] > 0:
            t2 = time.perf_counter()
            st["f"].write("%.6f %.3f %.3f %.3f %.3f %.3f %.3f %d %d\n" % (
                cur["t0"], st["inp"] * 1e3, cur["sched"] * 1e3, cur["bcast"] * 1e3, cur["wait"] * 1e3,
                cur["update"] * 1e3, (t2 - cur["t1"]) * 1e3, cur["n"], cur["pk"]))
            st["inp"] = 0.0
        return r

    ECP._process_engine_step = pes

    orig_piq = ECP._process_input_queue

    def piq(self):
        if not on(self):
            return orig_piq(self)
        t = time.perf_counter(); r = orig_piq(self); st["inp"] += time.perf_counter() - t
        return r

    ECP._process_input_queue = piq
    return True


def install_consolidate_log():
    """MOSHI_CONS_LOG=1 (API-server process): log what MultimodalPayload.consolidate_tensors concatenates."""
    import logging
    from vllm_omni.outputs import mm_outputs as mo
    MP = mo.MultimodalPayload
    if getattr(MP, "_moshi_cl", False):
        return False
    MP._moshi_cl = True
    orig = MP.consolidate_tensors
    log = logging.getLogger("moshi.cons")
    st = dict(n=0, t=0.0, bytes=0, maxlen=0, maxbytes=0, t0=time.perf_counter())

    def cons(self, strategy):
        nb, ln = 0, 0
        for v in self.tensors.values():
            if isinstance(v, list):
                ln = max(ln, len(v))
                nb += sum(int(x.numel()) * x.element_size() for x in v if hasattr(x, "numel"))
        t = time.perf_counter()
        r = orig(self, strategy)
        st["t"] += time.perf_counter() - t
        st["n"] += 1; st["k"] = st.get("k", 0) + 1; st["bytes"] += nb; st["maxlen"] = max(st["maxlen"], ln); st["maxbytes"] = max(st["maxbytes"], nb)
        if st["n"] % 2000 == 0 or (st["n"] % 200 == 0 and os.environ.get("MOSHI_CONS_PRINT") == "1"):
            span = time.perf_counter() - st["t0"]
            import sys
            print("[MoshiCons] %d calls: %.3f ms/call, %.1f%% of wall, avg %.0f KB/call, max list %d, max %.0f KB" % (
                        st["n"], 1e3 * st["t"] / max(1, st["k"]), 100 * st["t"] / span, st["bytes"] / max(1, st["k"]) / 1024, st["maxlen"],
                        st["maxbytes"] / 1024), file=sys.stderr, flush=True)
            st.update(t=0.0, bytes=0, maxlen=0, maxbytes=0, k=0, t0=time.perf_counter())
        return r

    MP.consolidate_tensors = cons
    return True


def install_consolidate_fix():
    """MOSHI_CONS_FIX=1 (API-server process): amortized O(1) append for the cumulative 1-D audio that vLLM-Omni
    re-concatenates every frame (MultimodalPayload.consolidate_tensors -> torch.cat([all so far, new]), O(session
    length) per frame). The accumulated tensor becomes a prefix view of a capacity-doubling buffer; new chunks are
    written after it, so earlier views handed to the data plane stay valid."""
    import torch
    from vllm_omni.outputs import mm_outputs as mo
    if getattr(mo, "_moshi_cf", False):
        return False
    mo._moshi_cf = True
    orig = mo._consolidate_tensor_list
    REPLACE = getattr(mo.TensorAccumulationStrategy, "REPLACE", None)

    def cons(key, tensor_list, strategy):
        if (strategy == REPLACE or len(tensor_list) < 2
                or not all(isinstance(t, torch.Tensor) and t.dim() == 1 for t in tensor_list)
                or len({t.dtype for t in tensor_list}) != 1):
            return orig(key, tensor_list, strategy)
        base = tensor_list[0]
        buf = getattr(base, "_moshi_buf", None)
        n0 = base.numel()
        total = n0 + sum(t.numel() for t in tensor_list[1:])
        if buf is None or buf.data_ptr() != base.data_ptr() or buf.numel() < total:
            nb = torch.empty(max(2 * total, 1 << 16), dtype=base.dtype)
            nb[:n0] = base
            buf = nb
        off = n0
        for t in tensor_list[1:]:
            buf[off:off + t.numel()] = t
            off += t.numel()
        out = buf[:total]
        out._moshi_buf = buf
        return out

    mo._consolidate_tensor_list = cons
    return True


def install_fast_append():
    """MOSHI_FAST_APPEND=1 (API-server process), 0.29 analog of vllm-omni PR #9491 (submit without thread offload).

    Stock 0.29 `AsyncOmniEngine.append_duplex_input_async` runs every 80 ms frame of every session through
    `loop.run_in_executor` -> `DuplexControlClient.execute` -> janus sync put -> blocking wait on a per-call
    queue.Queue filled by the RPC result router thread (two thread handoffs per frame, GIL ping-pong with the
    orchestrator thread). Here the event-loop thread enqueues directly (non-blocking janus put) and awaits an
    asyncio future that the router thread resolves via call_soon_threadsafe. Same message, same post-processing
    (DuplexControlClient over a pre-resolved transport), same errors."""
    import asyncio
    import uuid
    from vllm_omni.engine import async_omni_engine as aoe
    from vllm_omni.engine.duplex.control_client import DuplexControlClient
    from vllm_omni.engine.duplex.messages import AppendDuplexInputMessage
    E = aoe.AsyncOmniEngine
    if getattr(E, "_moshi_fa", False):
        return False
    E._moshi_fa = True
    orig = E.append_duplex_input_async

    class _FutWaiter:
        __slots__ = ("loop", "fut")

        def __init__(self, loop, fut):
            self.loop, self.fut = loop, fut

        def put_nowait(self, m):
            def _set():
                if not self.fut.done():
                    self.fut.set_result(m)
            self.loop.call_soon_threadsafe(_set)

    class _Pre:
        def __init__(self, result):
            self.result = result

        def execute(self, key, message, *, timeout=None, timeout_message="", block_on_submit=False):
            return self.result

    async def append_async(self, session_id, *, mode, payload, operation_id=None, final=False, expected_epoch=None,
                           fence, timeout=10.0):
        rpc = getattr(self, "_correlated_rpc_client", None)
        router = getattr(rpc, "_router", None)
        if router is None or (expected_epoch is not None and expected_epoch != fence.epoch):
            return await orig(self, session_id, mode=mode, payload=payload, operation_id=operation_id, final=final,
                              expected_epoch=expected_epoch, fence=fence, timeout=timeout)
        cid = uuid.uuid4().hex
        msg = AppendDuplexInputMessage(control_id=cid, operation_id=operation_id, fence=fence, session_id=session_id,
                                       expected_epoch=expected_epoch, mode=mode, payload=payload, final=final)
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        key = ("duplex", cid)
        w = _FutWaiter(loop, fut)
        with router._lock:
            if router._terminal_error is not None:
                raise RuntimeError(router._terminal_error.error)
            router._pending[key] = w
        try:
            try:
                self.request_queue.sync_q.put_nowait(msg)
            except Exception:  # queue full: fall back to the blocking path off the loop
                with router._lock:
                    router._pending.pop(key, None)
                return await orig(self, session_id, mode=mode, payload=payload, operation_id=operation_id,
                                  final=final, expected_epoch=expected_epoch, fence=fence, timeout=timeout)
            try:
                result = await asyncio.wait_for(fut, timeout)
            except asyncio.TimeoutError as exc:
                raise TimeoutError(f"duplex control timed out after {timeout} seconds") from exc
        finally:
            with router._lock:
                if router._pending.get(key) is w:
                    router._pending.pop(key, None)
        from vllm_omni.engine.messages import ErrorMessage
        if isinstance(result, ErrorMessage):
            raise RuntimeError(result.error)
        # reuse the stock post-processing (fence check, result dict, error raising) with the same control id
        return DuplexControlClient(_Pre(result), control_id_factory=lambda: cid).append(
            session_id, mode=mode, payload=payload, operation_id=operation_id, final=final,
            expected_epoch=expected_epoch, fence=fence, timeout=timeout)

    E.append_duplex_input_async = append_async
    return True
