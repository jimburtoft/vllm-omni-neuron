# Loaded by every Python process when scripts/fs_site is on PYTHONPATH: installs the Moshi frame-sync gate hook and
# optional GC instrumentation / tuning.
#   MOSHI_GC_LOG=<ms>   log every GC pass longer than <ms> (pid, generation, duration) to stderr
#   MOSHI_GC_FULL=0     no automatic full (gen-2) collections: per-session state grows over a session, and a gen-2
#                       pass over a large heap stalls the serving thread for 100+ ms (gens 0/1 still run)
#   MOSHI_GC_FREEZE=1   gc.freeze() once startup objects exist (after model load / warmup): moves them to the
#                       permanent generation so later gen-2 passes do not traverse the whole startup heap
import os

try:
    from moshi_neuron.serving.framesync import install_import_hook
    install_import_hook()
except Exception as _e:  # noqa: BLE001
    import sys
    sys.stderr.write(f"[MoshiFS] hook not installed: {_e}\n")

if os.environ.get("MOSHI_GC_LOG") or os.environ.get("MOSHI_GC_FULL") == "0" or os.environ.get("MOSHI_GC_FREEZE") == "1":
    import gc
    import sys
    import time
    if os.environ.get("MOSHI_GC_FULL") == "0":
        # a threshold2 change did not stop gen-2 passes (CPython also gates gen 2 on long_lived_pending/total);
        # disable automatic GC and collect generations 0/1 from a timer thread instead (gen 2 never runs).
        import threading
        gc.disable()

        def _young():
            while True:
                time.sleep(float(os.environ.get("MOSHI_GC_PERIOD_S", "0.5")))
                gc.collect(1)
        threading.Thread(target=_young, daemon=True, name="moshi-gc-young").start()
    if os.environ.get("MOSHI_GC_FREEZE") == "1":
        import threading

        def _freeze():
            time.sleep(float(os.environ.get("MOSHI_GC_FREEZE_AFTER_S", "120")))
            gc.collect()
            gc.freeze()
            sys.stderr.write(f"[MoshiGC] pid {os.getpid()} froze {gc.get_freeze_count()} objects\n")
        threading.Thread(target=_freeze, daemon=True, name="moshi-gc-freeze").start()
    _lim = float(os.environ.get("MOSHI_GC_LOG", "1e9")) / 1e3
    _st = {}

    def _gc_cb(phase, info):
        if phase == "start":
            _st["t"] = time.perf_counter()
        else:
            dt = time.perf_counter() - _st.get("t", time.perf_counter())
            if dt > _lim:
                sys.stderr.write(f"[MoshiGC] pid {os.getpid()} gen{info.get('generation')} {dt * 1e3:.0f} ms "
                                 f"collected {info.get('collected')}\n")

    gc.callbacks.append(_gc_cb)

if os.environ.get("MOSHI_HEAP_LOG") and "serve_launcher" in open("/proc/self/cmdline", "rb").read().decode(errors="ignore"):
    # API-server process only: a gc.get_objects() scan in a worker during torch.compile tracing broke Dynamo
    # ("tupleobject.c: bad argument to internal function")
    # MOSHI_HEAP_LOG=<s>: every <s> seconds log the object count by type growth (top 8) for this process
    import gc as _gc
    import sys as _sys
    import threading as _th
    import time as _time
    from collections import Counter as _C

    def _heap():
        prev = None
        while True:
            _time.sleep(float(os.environ["MOSHI_HEAP_LOG"]))
            c = _C(type(o).__name__ for o in _gc.get_objects())
            if prev is not None:
                d = sorted(((c[k] - prev.get(k, 0), k) for k in c), reverse=True)[:8]
                _sys.stderr.write(f"[MoshiHeap] pid {os.getpid()} total {sum(c.values())} grow " +
                                  " ".join(f"{k}:+{v}" for v, k in d if v > 0) + "\n")
            prev = c
    _th.Thread(target=_heap, daemon=True, name="moshi-heap").start()

def _cmdline():
    try:
        return open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return ""


if os.environ.get("MOSHI_TRACEMALLOC") and "serve_launcher" in _cmdline():
    # MOSHI_TRACEMALLOC=<s>: API-server only; every <s> seconds log the top-10 allocation sites by size growth
    import sys as _s2
    import threading as _t2
    import time as _tm2
    import tracemalloc as _tr
    _tr.start(8)

    def _tm():
        prev = None
        while True:
            _tm2.sleep(float(os.environ["MOSHI_TRACEMALLOC"]))
            snap = _tr.take_snapshot().filter_traces([_tr.Filter(False, _tr.__file__)])
            if prev is not None:
                top = snap.compare_to(prev, "traceback")[:6]
                for st in top:
                    fr = " <- ".join(f"{f.filename.split('site-packages/')[-1]}:{f.lineno}" for f in st.traceback[:5])
                    _s2.stderr.write(f"[MoshiTM] +{st.size_diff / 1e6:.1f} MB ({st.count_diff:+d}) {fr}\n")
            prev = snap
    _t2.Thread(target=_tm, daemon=True, name="moshi-tm").start()

if os.environ.get("MOSHI_SWITCH_INTERVAL"):
    # MOSHI_SWITCH_INTERVAL=<seconds>: CPython GIL switch interval (default 0.005). In the API process the frontend
    # event loop and the orchestrator thread hand the GIL back and forth per frame; a waiting thread can sit up to
    # one interval before it gets the GIL. Applied to every process (cheap); matters where threads contend.
    import sys as _s3
    _s3.setswitchinterval(float(os.environ["MOSHI_SWITCH_INTERVAL"]))

if os.environ.get("MOSHI_API_CPUS") or os.environ.get("MOSHI_EC_CPUS"):
    # MOSHI_API_CPUS / MOSHI_EC_CPUS: pin the API-server process / stage engine-core processes to CPU lists so they
    # do not land on the TP workers' cores (and their hyperthread siblings)
    try:
        _cl = open("/proc/self/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="ignore")

        def _cpus(spec):
            out = set()
            for part in spec.split(","):
                a, _, b = part.partition("-")
                out |= set(range(int(a), int(b or a) + 1))
            return out
        if "serve_launcher" in _cl and os.environ.get("MOSHI_API_CPUS"):
            os.sched_setaffinity(0, _cpus(os.environ["MOSHI_API_CPUS"]))
    except Exception:  # noqa: BLE001
        pass

if os.environ.get("MOSHI_SHM_BUSY_S"):
    # MOSHI_SHM_BUSY_S=<seconds>: vLLM's shm MessageQueue readers (TP workers waiting for the scheduler broadcast, the
    # engine core waiting for worker results) busy-spin with sched_yield for busy_loop_s (default 1 s) after the last
    # read before blocking on the zmq notify socket. With a step every ~60 ms they never block -> each reader burns a
    # core. A few ms keeps the fast path for back-to-back messages and blocks between steps.
    try:
        import importlib.abc as _iab2
        import importlib.util as _iu2
        import sys as _s4
        _busy = float(os.environ["MOSHI_SHM_BUSY_S"])

        def _patch_sc(mod):
            SC = mod.SpinCondition
            if getattr(SC, "_moshi_busy", False):
                return
            _oi = SC.__init__

            def _init(self, *a, **k):
                _oi(self, *a, **k)
                if getattr(self, "is_reader", False):
                    self.busy_loop_s = _busy      # set after init (the reader branch assigns the default itself)
            SC.__init__ = _init
            SC._moshi_busy = True

        _NAME = "vllm.distributed.device_communicators.shm_broadcast"
        if _NAME in _s4.modules:
            _patch_sc(_s4.modules[_NAME])
        else:
            class _F2(_iab2.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname != _NAME:
                        return None
                    _s4.meta_path.remove(self)
                    spec = _iu2.find_spec(fullname)
                    if spec is None or spec.loader is None:
                        return spec
                    _oe = spec.loader.exec_module

                    def _ex(module):
                        _oe(module)
                        _patch_sc(module)
                    spec.loader.exec_module = _ex
                    return spec
            _s4.meta_path.insert(0, _F2())
    except Exception as _e:  # noqa: BLE001
        __import__("sys").stderr.write(f"[MoshiShm] patch failed: {_e}\n")


if os.environ.get("MOSHI_RESP_SLEEP_MS"):
    # MOSHI_RESP_SLEEP_MS=<ms>: the stage-0 engine core waits for the TP workers' step result by spinning on the shm
    # response queue for the whole device step (~50 ms, ~65% of a core). Sleep <ms> per empty poll instead of
    # sched_yield (results arrive at most <ms> later). Only the engine core's response queues (reader of
    # worker_response_mq) are affected: TP workers read the broadcast queue, which stays a fast spin.
    try:
        import importlib.abc as _iab3
        import importlib.util as _iu3
        import sys as _s5
        import time as _t5
        _rs = float(os.environ["MOSHI_RESP_SLEEP_MS"]) / 1e3
        _NAME3 = "vllm.v1.executor.multiproc_executor"

        def _patch_mp(mod):
            ME = mod.MultiprocExecutor
            if getattr(ME, "_moshi_rs", False):
                return
            ME._moshi_rs = True
            _orig_init_ex = ME._init_executor

            def _ie(self, *a, **k):
                r = _orig_init_ex(self, *a, **k)
                for mq in getattr(self, "response_mqs", []) or []:
                    sc = getattr(mq, "_spin_condition", None)
                    if sc is None or not getattr(sc, "is_reader", False):
                        continue
                    _ow = sc.wait

                    def _w(timeout_ms=None, _ow=_ow, _sc=sc):
                        if _t5.monotonic() <= _sc.last_read + _sc.busy_loop_s:
                            _t5.sleep(_rs)
                        else:
                            _ow(timeout_ms)
                    sc.wait = _w
                    _s5.stderr.write(f"[MoshiResp] response queue wait -> sleep {_rs * 1e3:.2f} ms (pid {os.getpid()})\n")
                return r
            ME._init_executor = _ie

        if _NAME3 in _s5.modules:
            _patch_mp(_s5.modules[_NAME3])
        else:
            class _F3(_iab3.MetaPathFinder):
                def find_spec(self, fullname, path, target=None):
                    if fullname != _NAME3:
                        return None
                    _s5.meta_path.remove(self)
                    spec = _iu3.find_spec(fullname)
                    if spec is None or spec.loader is None:
                        return spec
                    _oe = spec.loader.exec_module

                    def _ex(module):
                        _oe(module)
                        _patch_mp(module)
                    spec.loader.exec_module = _ex
                    return spec
            _s5.meta_path.insert(0, _F3())
    except Exception as _e:  # noqa: BLE001
        __import__("sys").stderr.write(f"[MoshiResp] patch failed: {_e}\n")
