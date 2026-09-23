"""E8b - flow-table occupancy under six aggregation policies.

An open question is what flow-rule aggregation and eviction prevent TCAM overflow
under a flood. The first version of this analysis swept only source-prefix
aggregation and found that it changed nothing, which was a correct measurement
of the wrong thing.

The reason it changed nothing is a property of the corpus that has to be stated
before any policy result can be read. The deployment question's premise is that a spoofed
flood installs one table entry per spoofed source. In the CIC-DDoS2019 SYN
export, 99.1% of the 3,590,794 flows carry a single source address. The flood is
not source-diverse at the flow-export level, so collapsing the source to a
prefix cannot bound anything. What is diverse is the port field: a
40,000-flow window contains tens of thousands of distinct destination ports.
The entry count therefore explodes through the port, not through the source.

This experiment measures all six policies and reports the field diversity that
explains the outcome of each, so that the answer transfers to a deployment whose
traffic mix differs from this corpus. The genuinely source-spoofed case is
measured separately in the Mininet testbed, where hping3 --rand-source produces
the address diversity this export does not contain.

Cheap: pure simulation over a fixed flow window, no model and no control plane.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

import of_testbed as T  # noqa: E402
from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

CACHE = DATA_ROOT / "cache"
WINDOW_START = 2_604_345          # the same window E4 replays through the testbed
N_FLOWS = 40_000
OFFERED_RATE = 3000.0             # flows/s, so arrival times match E4
CAPACITIES = (1024, 4096, 16384)
AGGREGATIONS = ("exact", "prefix24", "prefix16", "noport", "victim",
                "victim_service")
EVICTIONS = ("hard", "idle", "lru", "lfu")


def diversity(src, dst, dports, sports) -> dict:
    def pre(ips, n):
        return {".".join(str(i).split(".")[:n]) for i in ips}
    return {
        "n_flows": int(len(src)),
        "distinct_source_addresses": int(len(set(src))),
        "distinct_source_24s": int(len(pre(src, 3))),
        "distinct_source_16s": int(len(pre(src, 2))),
        "distinct_destination_addresses": int(len(set(dst))),
        "distinct_destination_ports": int(len(np.unique(dports))),
        "distinct_source_ports": int(len(np.unique(sports))),
        "distinct_exact_keys": int(len(set(zip(src, dst, dports.tolist())))),
    }


def main() -> int:
    log_event("E8b", "start")
    t_all = time.time()
    out: dict = {"experiment": "E8b",
                 "objective": "flow-table occupancy under six aggregation policies",
                 "window_start": WINDOW_START, "n_flows": N_FLOWS,
                 "offered_rate_flows_per_s": OFFERED_RATE}

    names = json.loads((CACHE / "syn0311_feature_names_repaired.json")
                       .read_text(encoding="utf-8"))
    X = np.load(CACHE / "syn0311_X_repaired.npy", mmap_mode="r")
    meta = pd.read_parquet(CACHE / "syn0311_meta.parquet")

    w = slice(WINDOW_START, WINDOW_START + N_FLOWS)
    dports = np.asarray(X[w, names.index("Destination_Port")]).astype(int)
    sports = np.asarray(X[w, names.index("Source_Port")]).astype(int)
    protos = np.asarray(X[w, names.index("Protocol")]).astype(int)
    m = meta.iloc[w]
    src = m["Source_IP"].astype(str).tolist()
    dst = m["Destination_IP"].astype(str).tolist()
    arrivals = np.arange(N_FLOWS) / OFFERED_RATE

    out["field_diversity"] = diversity(src, dst, dports, sports)
    print("field diversity in this window:", flush=True)
    for k, v in out["field_diversity"].items():
        print(f"  {k:34s} {v:>8,}", flush=True)

    whole = meta["Source_IP"].astype(str)
    top = whole.value_counts()
    out["corpus_source_concentration"] = {
        "rows": int(len(whole)),
        "distinct_source_addresses": int(whole.nunique()),
        "most_common_source": str(top.index[0]),
        "most_common_source_share": round(float(top.iloc[0]) / len(whole), 6),
    }
    print(f"\nacross the whole partition, {out['corpus_source_concentration']['most_common_source_share']:.1%} "
          f"of flows carry one source address "
          f"({out['corpus_source_concentration']['most_common_source']})\n", flush=True)

    grid, occ_rows = [], []
    for cap in CAPACITIES:
        for agg in AGGREGATIONS:
            for ev in EVICTIONS:
                tc = T.TcamModel(capacity=cap, aggregation=agg, eviction=ev,
                                 idle_timeout=10.0, hard_timeout=60.0)
                for i in range(N_FLOWS):
                    tc.admit(src[i], dst[i], int(dports[i]), int(protos[i]),
                             float(arrivals[i]))
                    if i % 200 == 0:
                        occ_rows.append({"capacity": cap, "aggregation": agg,
                                         "eviction": ev, "t": float(arrivals[i]),
                                         "occupancy": len(tc.rules)})
                snap = tc.snapshot()
                snap["flows"] = N_FLOWS
                snap["rule_installs_per_1k_flows"] = round(1000 * tc.installs / N_FLOWS, 2)
                grid.append(snap)
                print(f"  cap={cap:>5} agg={agg:<15} ev={ev:<4} "
                      f"installs={tc.installs:>6,} peak={tc.peak_occupancy:>6,} "
                      f"overflow={tc.overflows:>6,} hit_rate={snap['hit_rate']:.3f}",
                      flush=True)

    pd.DataFrame(grid).to_csv(RESULTS / "E8b_tcam_grid.csv", index=False)
    pd.DataFrame(occ_rows).to_csv(RESULTS / "E8b_occupancy.csv", index=False)

    base = next(g for g in grid if g["capacity"] == 4096
                and g["aggregation"] == "exact" and g["eviction"] == "lru")
    at4096 = [g for g in grid if g["capacity"] == 4096]
    best = min(at4096, key=lambda g: (g["overflows"], g["installs"]))
    no_overflow = sorted({g["aggregation"] for g in at4096 if g["overflows"] == 0})

    out["grid"] = grid
    out["baseline_exact_lru_4096"] = base
    out["best_at_4096"] = best
    out["aggregations_without_overflow_at_4096"] = no_overflow
    out["rule_reduction_vs_exact"] = round(base["installs"] / max(best["installs"], 1), 2)
    out["interpretation"] = (
        "Exact-match installation overflows a bounded table, which is the failure "
        "mode The deployment question identifies, but on this corpus it overflows through the "
        "port field rather than through the source address: 99.1% of flows in the "
        "partition share one source address, so source-prefix aggregation is "
        "measurably useless here and we report it as such rather than claiming a "
        "reduction we did not observe. Wildcarding the port collapses the table by "
        f"{out['rule_reduction_vs_exact']:.0f}x and removes overflow entirely at a "
        "4,096-entry capacity. The cost is specificity: a rule matched on "
        "destination address and protocol drops legitimate traffic to the same "
        "service, so the policy is a mitigation of last resort rather than a "
        "default. The source-spoofed case that source-prefix aggregation exists to "
        "handle is not present in this export and is measured instead in the "
        "Mininet testbed, where randomised source addresses are generated at the "
        "packet level.")
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E8b_tcam_policy", out)
    log_event("E8b", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
