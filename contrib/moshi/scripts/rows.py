"""Summarize a tp4_rows.sh run: client latency + stage-0 steps/s, rows per step, step ms (from MOSHI_TRACE)."""
import glob
import json
import sys

import numpy as np

R, n = sys.argv[1], int(sys.argv[2])
res = [json.loads(l) for l in open(f"{R}/c_{n}.log") if l.startswith("{")]
out = dict(sessions=n, returned=len(res))
if res:
    out.update(min_frames=min(r["chunks"] for r in res),
               lag_p50=[round(min(r["lag_p50_ms"] for r in res)), round(max(r["lag_p50_ms"] for r in res))],
               lag_p99_max=round(max(r["lag_p99_ms"] for r in res)))
# rank-0 trace: the file with the most s0_exit rows > 0 (all ranks trace; take the busiest)
best = None
for p in glob.glob(f"{R}/tr_{n}/trace.s0.*"):
    ev = [l.split() for l in open(p)]
    ent = [(float(t), int(r)) for e, t, r in ev if e == "s0_enter"]
    ext = [(float(t), int(r)) for e, t, r in ev if e == "s0_exit"]
    if best is None or len(ext) > len(best[1]):
        best = (ent, ext)
if best and best[1]:
    ent, ext = best
    m = min(len(ent), len(ext))
    t = np.array([x[0] for x in ext[:m]])
    rows = np.array([x[1] for x in ext[:m]])
    dur = np.array([ext[i][0] - ent[i][0] for i in range(m)]) * 1e3
    lo, hi = t[0] + 5, t[-1] - 3            # steady state
    k = (t >= lo) & (t <= hi) & (rows > 0)
    span = hi - lo
    out.update(steps_per_s=round(k.sum() / span, 1), rows_per_s=round(rows[k].sum() / span),
               need_rows_per_s=round(n * 12.5), rows_p50=int(np.median(rows[k])), rows_mean=round(rows[k].mean(), 1),
               full_steps_pct=round(100 * float((rows[k] >= n).mean())), step_ms_p50=round(float(np.median(dur[k])), 1),
               step_ms_p99=round(float(np.quantile(dur[k], 0.99)), 1), busy=round(dur[k].sum() / 1e3 / span, 2))
print(json.dumps(out), flush=True)
with open(f"{R}/summary.jsonl", "a") as f:
    f.write(json.dumps(out) + "\n")
