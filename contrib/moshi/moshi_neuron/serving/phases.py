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


def install_realtime_marks_fix():
    """MOSHI_MARKS_FIX=1 (API-server process): O(1)-per-frame response bookkeeping in vLLM-Omni's realtime output
    projector (0.29 entrypoints/duplex/realtime_output.py). Stock code does, for EVERY audio delta of a session:
      - _remember_response_audio_metadata: list(all marks) + extend + dict-dedupe + full sort  -> O(n log n)
      - _refresh_in_progress_response_item (up to twice): rebuilds the in-progress conversation item with a dict copy
        of every mark and "".join(all transcript parts)                                       -> O(n)
    i.e. quadratic in session length on the single API event-loop thread (60 -> 75% CPU over a 60 s session).
    Fix: marks are appended in order when monotonic (the normal case: each frame's marks end later than the last);
    the full merge/sort runs only when a mark arrives out of order. The in-progress item refresh is deferred: the
    item is marked dirty and rebuilt only when something reads it (conversation.item.retrieve / truncate / done)."""
    from vllm_omni.entrypoints.duplex import realtime_output as ro
    P = ro.RealtimeOutputProjector
    if getattr(P, "_moshi_mf", False):
        return False
    P._moshi_mf = True
    orig_remember = P._remember_response_audio_metadata
    orig_refresh = P._refresh_in_progress_response_item

    def remember(self, response_id, event):
        state = self._response_state(response_id)
        if state is None:
            return
        marks = event.get("audio_text_marks")
        if not isinstance(marks, list) or not marks:
            return orig_remember(self, response_id, event)
        clean = []
        for m in marks:
            if not isinstance(m, dict):
                continue
            tc = m.get("text_chars")
            ae = m.get("audio_end_ms", m.get("audio_ms"))
            if not isinstance(tc, int | float) or not isinstance(ae, int | float):
                continue
            clean.append({"text_chars": max(0, int(tc)), "audio_end_ms": max(0, int(ae))})
        cur = state.audio_text_marks
        last = (cur[-1]["audio_end_ms"], cur[-1]["text_chars"]) if cur else (-1, -1)
        keys = [(c["audio_end_ms"], c["text_chars"]) for c in clean]
        if clean and all(k > last for k in keys) and keys == sorted(set(keys)):
            # fast path: strictly after the last stored mark, sorted, unique -> same result as merge+dedupe+sort
            duration = event.get("audio_duration_ms")
            playback = event.get("playback")
            if not isinstance(duration, int | float) and isinstance(playback, dict):
                duration = playback.get("sent_ms") or playback.get("generated_ms")
            if isinstance(duration, int | float):
                state.audio_duration_ms = max(state.audio_duration_ms or 0, int(duration))
            cur.extend(clean)
            return
        return orig_remember(self, response_id, event)

    def refresh(self, response_id):
        d = self.__dict__.setdefault("_moshi_dirty", set())
        d.add(response_id)

    def _flush(self):
        d = self.__dict__.get("_moshi_dirty")
        if d:
            for rid in list(d):
                orig_refresh(self, rid)
            d.clear()

    P._remember_response_audio_metadata = remember
    P._refresh_in_progress_response_item = refresh
    P._moshi_flush = _flush
    # Every projected event except an audio delta sees an up-to-date item (done / item events / transcript etc.).
    orig_from = P._from_duplex_event

    _cheap = {"response.output_audio.delta", "response.text.delta", "playback.acknowledged",
              "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped"}

    def from_event(self, event):
        if not (isinstance(event, dict) and event.get("type") in _cheap):
            _flush(self)
        return orig_from(self, event)
    P._from_duplex_event = from_event
    # Client-side reads of conversation items (retrieve / truncate) go through the input translator, which is a
    # mixin of the same session object in 0.29: flush before handling any client realtime event.
    from vllm_omni.entrypoints.duplex import realtime_input as ri
    T = ri.RealtimeInputTranslator
    for name in ["_to_duplex_event"]:
        fn = getattr(T, name)

        def mk(fn):
            # only events that read or mutate conversation items need the deferred refresh applied first
            # (the per-frame input_audio_buffer.append must stay O(1))
            def _needs(a, k):
                ev = a[0] if a and isinstance(a[0], dict) else k.get("event")
                t = ev.get("type") if isinstance(ev, dict) else None
                # per-frame / per-event client traffic that never reads conversation items
                return not (isinstance(t, str) and (t.startswith("input_audio_buffer.") or t in (
                    "session.event_ack", "playback.ack", "session.heartbeat")))
            if __import__("inspect").iscoroutinefunction(fn):
                async def w(self, *a, **k):
                    f = getattr(self, "_moshi_flush", None)
                    if f and _needs(a, k):
                        f()
                    return await fn(self, *a, **k)
            else:
                def w(self, *a, **k):
                    f = getattr(self, "_moshi_flush", None)
                    if f and _needs(a, k):
                        f()
                    return fn(self, *a, **k)
            return w
        setattr(T, name, mk(fn))
    return True


def install_snapshot_fix():
    """MOSHI_SNAP_FIX=1 (stage-0 engine-core process): vllm-omni's chunk adapter snapshots the stage-0 Request for
    every 80 ms frame before handing it to the custom stage processor (`_snapshot_processor_request`), copying
    `_all_token_ids` / `_output_token_ids` (one token per frame for the whole session -> O(session length) per
    frame, quadratic per session) and deep-copying `additional_information`. Our PCM pass-through processor
    (MOSHI_S0_DECODE=1, processors.pcm_passthrough_async_chunk) reads only `additional_information` (this frame's
    PCM + text token, replaced every step) and `resumable`, so the snapshot keeps those and shares the token lists
    read-only."""
    if os.environ.get("MOSHI_S0_DECODE", "0") != "1":
        return False
    from vllm_omni.distributed.omni_connectors.transfer_adapter import chunk_transfer_adapter as cta
    A = cta.OmniChunkTransferAdapter
    if getattr(A, "_moshi_sf", False):
        return False
    A._moshi_sf = True
    import copy as _copy

    def snap(request):
        s = _copy.copy(request)
        ai = getattr(request, "additional_information", None)
        if isinstance(ai, dict):
            s.additional_information = dict(ai)
        return s

    A._snapshot_processor_request = staticmethod(snap)
    return True


def install_stage_events_cap():
    """MOSHI_STAGE_EVENTS_CAP=N (API-server process): vllm-omni's OrchestratorAggregator appends one
    StageRequestStats (+ StageStats) per 80 ms frame to stage_events[request_id] for the whole duplex session and
    never trims it (the resumable duplex request never finishes) -- ~24k objects/30 s at 32 sessions, the API server's
    RSS grows ~4 MB/s. All per-frame readers read `stage_events[rid][cursor:]` where cursor was taken at the start
    of the current append (DuplexRequestClient.append -> stage_event_cursor), i.e. only the newest few events. Keep
    the last N per request; the cursor is rebased by remembering how many were dropped."""
    from vllm_omni.metrics import stats as st
    A = st.OrchestratorAggregator
    if getattr(A, "_moshi_cap", False):
        return False
    A._moshi_cap = True
    cap = int(os.environ.get("MOSHI_STAGE_EVENTS_CAP", "64"))

    class _Capped(list):
        """list whose logical indices keep growing while only the tail is stored: len() and slices [c:] are in
        logical coordinates, so `cursor = len(x)` then `x[cursor:]` works as with the full list."""
        __slots__ = ("dropped",)

        def __init__(self, *a):
            super().__init__(*a)
            self.dropped = 0

        def append(self, v):
            super().append(v)
            n = super().__len__()
            if n > 2 * cap:
                del self[:n - cap]  # list.__delitem__ (physical)
                self.dropped += n - cap

        def __len__(self):
            return super().__len__() + self.dropped

        def __getitem__(self, k):
            if isinstance(k, slice) and k.step is None and (k.start is None or k.start >= 0) and k.stop is None:
                s = (k.start or 0) - self.dropped
                return list.__getitem__(self, slice(max(0, s), None))
            return list.__getitem__(self, k)

        def __iter__(self):
            return list.__iter__(self)

        def __bool__(self):
            return list.__len__(self) > 0

    orig_init = A.__init__

    import weakref
    live = weakref.WeakSet()

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        live.add(self)

        class _D(dict):
            def setdefault(self, key, default=None):
                if key not in self:
                    dict.__setitem__(self, key, _Capped(default or []))
                return dict.__getitem__(self, key)
        self.stage_events = _D(self.stage_events)

    A.__init__ = init
    if os.environ.get("MOSHI_STAGE_EVENTS_LOG"):
        import sys
        import threading

        def _log():
            while True:
                time.sleep(float(os.environ["MOSHI_STAGE_EVENTS_LOG"]))
                aggs = list(live)
                stored = sum(sum(list.__len__(v) if isinstance(v, list) else 0 for v in a.stage_events.values())
                             for a in aggs)
                logical = sum(sum(len(v) for v in a.stage_events.values()) for a in aggs)
                other = sum(len(getattr(a, "transfer_events", {})) + len(getattr(a, "e2e_events", []))
                            + sum(len(v) for v in getattr(a, "accumulated_gen_time_ms", {}).values())
                            for a in aggs)
                sys.stderr.write(f"[MoshiSE] aggregators {len(aggs)} stage_events stored {stored} logical {logical} "
                                 f"other {other}\n")
        threading.Thread(target=_log, daemon=True, name="moshi-se").start()
    return True


def install_mark_coalesce():
    """MOSHI_MARK_COALESCE=1 (API-server process): `DuplexSession.mark_audio_sent` appends ~2 audio/text marks per
    80 ms frame for the whole response (kept for barge-in truncation: audio position -> transcript chars). All
    consumers interpolate chars piecewise-linearly between consecutive marks (protocol._text_chars_for_audio_ms,
    realtime_input._text_chars_for_audio_ms_from_marks). Marks are appended in non-decreasing audio_end_ms; for a run
    of marks with the SAME text_chars, the interior marks change nothing except where the segment starts, so
    keeping the first and the last of each run gives identical results at every audio position. Appended marks
    that only extend the last run replace its last element instead of growing the list."""
    from vllm_omni.entrypoints.duplex import protocol as pr
    S = pr.DuplexSession
    if getattr(S, "_moshi_mc", False):
        return False
    S._moshi_mc = True
    orig = S.mark_audio_sent

    def mark_audio_sent(self, duration_ms=None, *, text_chars=None, audio_text_marks=None):
        lst = self._response.assistant_audio_text_marks
        n0 = len(lst)
        orig(self, duration_ms, text_chars=text_chars, audio_text_marks=audio_text_marks)
        # coalesce the newly appended tail into the existing runs
        i = max(0, n0)
        while i < len(lst):
            m = lst[i]
            if (i >= 2 and lst[i - 1].text_chars == m.text_chars == lst[i - 2].text_chars
                    and lst[i - 1].audio_end_ms <= m.audio_end_ms):
                # [.., a(c), b(c), m(c)] -> [.., a(c), m(c)]: b is interior to a flat run
                del lst[i - 1]
                continue
            if i >= 1 and lst[i - 1].text_chars == m.text_chars and lst[i - 1].audio_end_ms == m.audio_end_ms:
                del lst[i]  # exact duplicate
                continue
            i += 1

    S.mark_audio_sent = mark_audio_sent
    return True


def install_drain_batch():
    """MOSHI_DRAIN_BATCH=1 (API-server process): DuplexRequestClient.collect_outputs returns after the FIRST output
    message, so each session's frontend drain loop forwards one 80 ms frame per iteration (one projection + one
    websocket send + several awaits). Under load a session that falls behind can never catch up (it can only
    forward one frame per iteration, at the rate frames arrive). Patched: keep collecting while more messages are
    already queued (no extra waiting), so a backlog is forwarded in one iteration."""
    import inspect
    import textwrap
    from vllm_omni.entrypoints import duplex_request_client as m
    C = m.DuplexRequestClient
    if getattr(C, "_moshi_db", False):
        return False
    src = textwrap.dedent(inspect.getsource(C.collect_outputs))
    old = "            if message.finished or outputs:\n                break"
    if old.replace("            ", "    ", 1) not in src and old not in src:
        old_d = "        if message.finished or outputs:\n            break"
    else:
        old_d = old.replace("            ", "    ", 1) if old.replace("            ", "    ", 1) in src else old
    new_d = old_d.replace("message.finished or outputs:", "message.finished or (outputs and request_state.queue.empty()):")
    if old_d not in src:
        raise RuntimeError("collect_outputs source changed; drain-batch patch not applied")
    ns = {}
    exec(compile(src.replace(old_d, new_d), m.__file__, "exec"), m.__dict__, ns)
    C.collect_outputs = ns["collect_outputs"]
    C._moshi_db = True
    return True


def install_output_queue_probe():
    """MOSHI_OQ_PROBE=<s>: every <s> s log the depth of the orchestrator -> frontend output queue and the per-request
    queues (where the 44-session output lag accumulates)."""
    import sys
    import threading
    from vllm_omni.engine import async_omni_engine as aoe
    E = aoe.AsyncOmniEngine
    if getattr(E, "_moshi_oq", False):
        return False
    E._moshi_oq = True
    orig_init = E.__init__
    period = float(os.environ.get("MOSHI_OQ_PROBE", "5"))

    def init(self, *a, **k):
        orig_init(self, *a, **k)
        eng = self

        def run():
            while True:
                time.sleep(period)
                try:
                    q = eng.output_queue.sync_q.qsize()
                    rq = eng.request_queue.sync_q.qsize()
                    sys.stderr.write(f"[MoshiOQ] output_queue {q} request_queue {rq}\n")
                except Exception:  # noqa: BLE001
                    pass
        threading.Thread(target=run, daemon=True, name="moshi-oq").start()
    E.__init__ = init
    return True
