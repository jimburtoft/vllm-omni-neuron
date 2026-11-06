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
