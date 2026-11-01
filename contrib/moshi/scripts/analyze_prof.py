"""Analyze the Moshi LM frame profile (neuron-explorer parquet): time per phase, engines, DMA efficiency."""
import sys

import duckdb

name = sys.argv[1] if len(sys.argv) > 1 else "lm-b4"
d = f"/home/ubuntu/parquet/profiles/global/{name}@latest"
con = duckdb.connect()
for t in ("Instruction", "DmaPacketAggregated", "ActiveTime", "HloInstruction", "Summary"):
    con.execute(f"create view {t} as select * from '{d}/{t}.parquet'")
q = lambda s: con.execute(s).fetchdf()

span = q("select min(start_ts) t0, max(end_ts) t1 from Instruction").iloc[0]
T0, T1 = int(span.t0), int(span.t1)
print(f"== {name}: span {(T1 - T0) / 1e6:.2f} ms")

print("\n-- engine busy (merged, ms) per pcore")
print(q(f"""select engine, pcore_idx, round(sum(duration_ns)/1e6,2) busy_ms, count(*) n from ActiveTime
            group by engine, pcore_idx order by engine, pcore_idx""").to_string(index=False))

print("\n-- top opcodes by summed duration (all engines)")
print(q("""select engine, opcode, count(*) n, round(sum(duration_ns)/1e6,2) sum_ms,
                  round(sum(evt_wait_time_ns)/1e6,2) wait_ms
           from Instruction group by engine, opcode order by sum_ms desc limit 25""").to_string(index=False))

print("\n-- DMA by queue type / op")
print(q("""select queue_type, op, count(*) n, round(sum(transfer_bytes)/1e9,3) GB,
                  round(sum(duration_ns)/1e6,2) sum_ms,
                  round(sum(transfer_bytes)/nullif(sum(duration_ns),0),2) GBps_per_pkt
           from DmaPacketAggregated group by queue_type, op order by GB desc limit 15""").to_string(index=False))

print("\n-- DMA size histogram")
print(q("""select case when transfer_bytes < 2048 then 'a <2KB' when transfer_bytes < 65536 then 'b 2-64KB'
                       when transfer_bytes < 1048576 then 'c 64KB-1MB' else 'd >=1MB' end bucket,
                  count(*) n, round(sum(transfer_bytes)/1e9,3) GB, round(sum(duration_ns)/1e6,2) sum_ms
           from DmaPacketAggregated group by bucket order by bucket""").to_string(index=False))

# time-slice view: split the span into 100 bins, show which HLO family dominates each 5% of time
print("\n-- HLO instructions by duration (top 30)")
h = q("""select hlo_name, round((end_ts-start_ts)/1e6,3) ms, round(hbm_read_bytes/1e6,1) read_MB,
                round(load_weight_bytes/1e6,1) lw_MB, start_ts
         from HloInstruction order by (end_ts-start_ts) desc limit 30""")
print(h.drop(columns=["start_ts"]).to_string(index=False))

print("\n-- HLO by name family (prefix before '.'), summed span ms (overlapping)")
print(q("""select regexp_replace(hlo_name, '[._][0-9]+$', '') fam, count(*) n,
                  round(sum(end_ts-start_ts)/1e6,2) sum_ms, round(sum(hbm_read_bytes)/1e9,3) read_GB
           from HloInstruction group by fam order by sum_ms desc limit 30""").to_string(index=False))

# gaps where nothing is active on a pcore
print("\n-- idle time per pcore (no engine active, no DMA)")
for pc in (0, 1):
    iv = q(f"""select start_ts s, end_ts e from ActiveTime where pcore_idx={pc}
               union all select start_ts s, end_ts e from DmaPacketAggregated where pcore_idx={pc} order by s""")
    cur_e, idle, nidle = T0, 0, 0
    for s, e in zip(iv.s, iv.e):
        if s > cur_e:
            if s - cur_e > 2000:
                nidle += 1
            idle += s - cur_e
        cur_e = max(cur_e, e)
    idle += max(0, T1 - cur_e)
    print(f"  pcore {pc}: idle {idle / 1e6:.2f} ms ({100 * idle / (T1 - T0):.1f}%), gaps>2us: {nidle}")
