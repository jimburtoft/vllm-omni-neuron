"""Split the LM frame timeline into NKI attention-kernel windows vs compiler-generated code (MLP, heads, depformer)."""
import sys

import duckdb
import numpy as np

name = sys.argv[1] if len(sys.argv) > 1 else "lm-b4"
d = f"/home/ubuntu/parquet/profiles/global/{name}@latest"
con = duckdb.connect()
con.execute(f"create view I as select * from '{d}/Instruction.parquet'")
con.execute(f"create view D as select * from '{d}/DmaPacketAggregated.parquet'")
q = lambda s: con.execute(s).fetchdf()

T0, T1 = q("select min(start_ts), max(end_ts) from I").iloc[0]
print(f"{name}: span {(T1 - T0) / 1e6:.2f} ms")
print(q("""select (nki_call_site_id is not null) is_nki, count(*) n, round(sum(duration_ns)/1e6,2) sum_ms
           from I group by is_nki""").to_string(index=False))
print(q("""select kernel_instruction_name is not null k, nki_source_location is not null s, count(*) n
           from I group by k, s""").to_string(index=False))

# merge NKI instruction intervals (pcore 0) into windows
iv = q("""select start_ts s, end_ts e from I where nki_call_site_id is not null and pcore_idx=0 order by s""")
wins = []
for s, e in zip(iv.s.values, iv.e.values):
    if wins and s - wins[-1][1] < 20000:  # merge gaps < 20 us
        wins[-1][1] = max(wins[-1][1], e)
    else:
        wins.append([s, e])
wins = np.array(wins)
dur = (wins[:, 1] - wins[:, 0]) / 1e6
print(f"NKI windows (pcore0): {len(wins)}, total {dur.sum():.2f} ms, median {np.median(dur):.3f} ms, "
      f"min {dur.min():.3f} max {dur.max():.3f}")
gaps = (wins[1:, 0] - wins[:-1, 1]) / 1e6
print(f"between windows: total {gaps.sum():.2f} ms, median {np.median(gaps):.3f} ms; before first "
      f"{(wins[0, 0] - T0) / 1e6:.2f} ms, after last {(T1 - wins[-1, 1]) / 1e6:.2f} ms")

# DMA bytes inside vs outside NKI windows
dm = q("select start_ts s, end_ts e, transfer_bytes b, queue_type from D where pcore_idx=0")
mid = (dm.s.values + dm.e.values) // 2
idx = np.searchsorted(wins[:, 0], mid, side="right") - 1
inside = (idx >= 0) & (mid <= wins[np.clip(idx, 0, None), 1])
print(f"DMA bytes pcore0: inside NKI {dm.b[inside].sum() / 1e9:.2f} GB, outside {dm.b[~inside].sum() / 1e9:.2f} GB")
for nm, m in (("inside", inside), ("outside", ~inside)):
    t = dur.sum() if nm == "inside" else gaps.sum() + (wins[0, 0] - T0) / 1e6 + (T1 - wins[-1, 1]) / 1e6
    print(f"  {nm}: {dm.b[m].sum() / 1e9:.2f} GB in {t:.2f} ms -> {dm.b[m].sum() / 1e9 / (t / 1e3):.0f} GB/s (pcore0)")

# per-engine busy inside NKI windows
for eng in ("Tensor", "Vector", "Scalar", "GpSimd", "Sync"):
    r = q(f"""select sum(duration_ns) t from I where engine='{eng}' and pcore_idx=0 and nki_call_site_id is not null""").iloc[0].t
    r2 = q(f"""select sum(duration_ns) t from I where engine='{eng}' and pcore_idx=0 and nki_call_site_id is null""").iloc[0].t
    print(f"  {eng}: sum inside NKI {(r or 0) / 1e6:.2f} ms, outside {(r2 or 0) / 1e6:.2f} ms")

print(q("""select opcode, engine, count(*) n, round(sum(duration_ns)/1e6,2) ms from I
           where nki_call_site_id is not null and pcore_idx=0 group by opcode, engine order by ms desc limit 12""").to_string(index=False))
print(q("""select nki_source_location loc, count(*) n, round(sum(duration_ns)/1e6,2) ms from I
           where nki_call_site_id is not null and pcore_idx=0 and nki_source_location is not null
           group by loc order by ms desc limit 15""").to_string(index=False))
