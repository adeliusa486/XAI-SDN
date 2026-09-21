"""Phase 2.2 - Profile every downloaded dataset and record the cross-dataset schema.

Produces 02_data/data_profile.json and 02_data/schema_matrix.csv. The schema
matrix is what experiment E2 needs in order to make cross-dataset transfer
like-for-like rather than approximate, and it is published in the manuscript so
that the mapping is auditable (R4.1).
"""
from __future__ import annotations

import collections
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import DATA_META, DATA_ROOT  # noqa: E402

DATASETS = {
    "CICDDoS2019_0311_Syn":     ("CICDDoS2019/03-11/Syn.csv", " Label"),
    "CICDDoS2019_0311_LDAP":    ("CICDDoS2019/03-11/LDAP.csv", " Label"),
    "CICDDoS2019_0311_NetBIOS": ("CICDDoS2019/03-11/NetBIOS.csv", " Label"),
    "CICDDoS2019_0311_Portmap": ("CICDDoS2019/03-11/Portmap.csv", " Label"),
    "CICDDoS2019_0112_Syn":     ("CICDDoS2019/01-12/Syn.csv", " Label"),
    "InSDN":                    ("InSDN/Dataset.csv", "target"),
    "CICIDS2017_DDoS":          ("CICIDS2017/Friday-WorkingHours-Afternoon-DDos.csv", " Label"),
    "CICIDS2017_Benign":        ("CICIDS2017/Monday-WorkingHours.csv", " Label"),
}

# Semantic identity across the two CICFlowMeter naming conventions.
# left  = canonical long form (CIC-DDoS2019, CIC-IDS2017)
# right = short form (InSDN / CICFlowMeter v4)
ALIASES: dict[str, list[str]] = {
    "Destination_Port": ["Dst_Port"],
    "Source_Port": ["Src_Port"],
    "Source_IP": ["Src_IP"],
    "Destination_IP": ["Dst_IP"],
    "Total_Fwd_Packets": ["Tot_Fwd_Pkts"],
    "Total_Backward_Packets": ["Tot_Bwd_Pkts"],
    "Total_Length_of_Fwd_Packets": ["TotLen_Fwd_Pkts"],
    "Total_Length_of_Bwd_Packets": ["TotLen_Bwd_Pkts"],
    "Fwd_Packet_Length_Max": ["Fwd_Pkt_Len_Max"],
    "Fwd_Packet_Length_Min": ["Fwd_Pkt_Len_Min"],
    "Fwd_Packet_Length_Mean": ["Fwd_Pkt_Len_Mean"],
    "Fwd_Packet_Length_Std": ["Fwd_Pkt_Len_Std"],
    "Bwd_Packet_Length_Max": ["Bwd_Pkt_Len_Max"],
    "Bwd_Packet_Length_Min": ["Bwd_Pkt_Len_Min"],
    "Bwd_Packet_Length_Mean": ["Bwd_Pkt_Len_Mean"],
    "Bwd_Packet_Length_Std": ["Bwd_Pkt_Len_Std"],
    "Flow_Bytes_s": ["Flow_Byts_s"],
    "Flow_Packets_s": ["Flow_Pkts_s"],
    "Fwd_IAT_Total": ["Fwd_IAT_Tot"],
    "Bwd_IAT_Total": ["Bwd_IAT_Tot"],
    "Fwd_Header_Length": ["Fwd_Header_Len"],
    "Bwd_Header_Length": ["Bwd_Header_Len"],
    "Fwd_Packets_s": ["Fwd_Pkts_s"],
    "Bwd_Packets_s": ["Bwd_Pkts_s"],
    "Min_Packet_Length": ["Pkt_Len_Min"],
    "Max_Packet_Length": ["Pkt_Len_Max"],
    "Packet_Length_Mean": ["Pkt_Len_Mean"],
    "Packet_Length_Std": ["Pkt_Len_Std"],
    "Packet_Length_Variance": ["Pkt_Len_Var"],
    "FIN_Flag_Count": ["FIN_Flag_Cnt"],
    "SYN_Flag_Count": ["SYN_Flag_Cnt"],
    "RST_Flag_Count": ["RST_Flag_Cnt"],
    "PSH_Flag_Count": ["PSH_Flag_Cnt"],
    "ACK_Flag_Count": ["ACK_Flag_Cnt"],
    "URG_Flag_Count": ["URG_Flag_Cnt"],
    "CWE_Flag_Count": ["CWE_Flag_Count"],
    "ECE_Flag_Count": ["ECE_Flag_Cnt"],
    "Average_Packet_Size": ["Pkt_Size_Avg"],
    "Avg_Fwd_Segment_Size": ["Fwd_Seg_Size_Avg"],
    "Avg_Bwd_Segment_Size": ["Bwd_Seg_Size_Avg"],
    "Fwd_Avg_Bytes_Bulk": ["Fwd_Byts_b_Avg"],
    "Fwd_Avg_Packets_Bulk": ["Fwd_Pkts_b_Avg"],
    "Fwd_Avg_Bulk_Rate": ["Fwd_Blk_Rate_Avg"],
    "Bwd_Avg_Bytes_Bulk": ["Bwd_Byts_b_Avg"],
    "Bwd_Avg_Packets_Bulk": ["Bwd_Pkts_b_Avg"],
    "Bwd_Avg_Bulk_Rate": ["Bwd_Blk_Rate_Avg"],
    "Subflow_Fwd_Packets": ["Subflow_Fwd_Pkts"],
    "Subflow_Fwd_Bytes": ["Subflow_Fwd_Byts"],
    "Subflow_Bwd_Packets": ["Subflow_Bwd_Pkts"],
    "Subflow_Bwd_Bytes": ["Subflow_Bwd_Byts"],
    "Init_Win_bytes_forward": ["Init_Fwd_Win_Byts"],
    "Init_Win_bytes_backward": ["Init_Bwd_Win_Byts"],
    "act_data_pkt_fwd": ["Fwd_Act_Data_Pkts"],
    "min_seg_size_forward": ["Fwd_Seg_Size_Min"],
}


def clean_name(c: str) -> str:
    out = re.sub(r"[^A-Za-z0-9_]", "", re.sub(r"[\s/]+", "_", c.strip()))
    return "Fwd_Header_Length_2" if out == "Fwd_Header_Length1" else out


def canonical(c: str) -> str:
    c = clean_name(c)
    for canon, alts in ALIASES.items():
        if c == canon or c in alts:
            return canon
    return c


def main() -> int:
    from data import CIC_FEATURE_NAMES  # noqa: E402

    profile: dict = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"), "datasets": {}}
    present: dict[str, set[str]] = {}

    for key, (rel, label_col) in DATASETS.items():
        path = DATA_ROOT / rel
        if not path.exists():
            profile["datasets"][key] = {"status": "missing"}
            continue
        print(f"profiling {key}...", flush=True)
        head = pd.read_csv(path, nrows=0)
        cols_raw = list(head.columns)
        cols_canon = [canonical(c) for c in cols_raw]
        present[key] = set(cols_canon)

        lab = collections.Counter()
        n = 0
        t0 = time.time()
        lc = label_col if label_col in cols_raw else \
            next((c for c in cols_raw if c.strip().lower() in ("label", "target", "class")), None)
        if lc:
            for ch in pd.read_csv(path, usecols=[lc], chunksize=1_000_000, low_memory=False):
                n += len(ch)
                lab.update(ch[lc].astype(str).str.strip().value_counts().to_dict())
        profile["datasets"][key] = {
            "status": "ok",
            "path": str(path),
            "bytes": path.stat().st_size,
            "n_columns": len(cols_raw),
            "label_column": lc,
            "n_rows": int(n),
            "label_counts": {k: int(v) for k, v in lab.most_common()},
            "scan_seconds": round(time.time() - t0, 1),
            "has_source_ip": "Source_IP" in present[key],
            "has_destination_ip": "Destination_IP" in present[key],
            "has_source_port": "Source_Port" in present[key],
            "has_timestamp": "Timestamp" in present[key],
            "n_canonical_cic_features_present": sum(
                1 for f in CIC_FEATURE_NAMES if f in present[key]),
            "missing_cic_features": [f for f in CIC_FEATURE_NAMES if f not in present[key]],
        }
        d = profile["datasets"][key]
        print(f"  rows={d['n_rows']:,} cols={d['n_columns']} "
              f"cic_features={d['n_canonical_cic_features_present']}/80 "
              f"src_ip={d['has_source_ip']} labels={list(d['label_counts'])[:4]}", flush=True)

    # --- cross-dataset schema matrix ---------------------------------------
    keys = [k for k, v in profile["datasets"].items() if v.get("status") == "ok"]
    rows = []
    for f in CIC_FEATURE_NAMES:
        row = {"canonical_feature": f}
        for k in keys:
            row[k] = int(f in present[k])
        rows.append(row)
    for e, need in [("H_src_ip", "Source_IP"), ("H_dst_ip", "Destination_IP"),
                    ("H_dst_port", "Destination_Port"), ("H_proto", "Protocol"),
                    ("H_pkt_len", "Packet_Length_Mean"), ("H_iat", "Flow_IAT_Mean"),
                    ("H_tcp_flags", "SYN_Flag_Count"), ("H_src_port", "Source_Port")]:
        row = {"canonical_feature": f"{e} (requires {need})"}
        for k in keys:
            row[k] = int(need in present[k])
        rows.append(row)
    sm = pd.DataFrame(rows)
    sm.to_csv(DATA_META / "schema_matrix.csv", index=False)

    # intersections that matter for E2
    def inter(a, b, feats):
        return sorted(f for f in feats if f in present[a] and f in present[b])

    ent_need = {"H_src_ip": "Source_IP", "H_dst_ip": "Destination_IP",
                "H_dst_port": "Destination_Port", "H_proto": "Protocol",
                "H_pkt_len": "Packet_Length_Mean", "H_iat": "Flow_IAT_Mean",
                "H_tcp_flags": "SYN_Flag_Count", "H_src_port": "Source_Port"}
    profile["transfer_schemas"] = {}
    for tgt in ("InSDN", "CICIDS2017_DDoS"):
        if tgt not in present:
            continue
        cic = inter("CICDDoS2019_0311_Syn", tgt, CIC_FEATURE_NAMES)
        ent = [e for e, need in ent_need.items()
               if need in present["CICDDoS2019_0311_Syn"] and need in present[tgt]]
        profile["transfer_schemas"][f"CICDDoS2019_0311_Syn -> {tgt}"] = {
            "n_shared_cic_features": len(cic),
            "n_shared_entropy_features": len(ent),
            "shared_entropy_features": ent,
            "unavailable_entropy_features": [e for e in ent_need if e not in ent],
            "total_transfer_dimension": len(cic) + len(ent),
            "dropped_cic_features": [f for f in CIC_FEATURE_NAMES if f not in cic],
        }
        s = profile["transfer_schemas"][f"CICDDoS2019_0311_Syn -> {tgt}"]
        print(f"  transfer to {tgt}: {s['total_transfer_dimension']}-dim "
              f"({s['n_shared_cic_features']} CIC + {s['n_shared_entropy_features']} entropy); "
              f"unavailable: {s['unavailable_entropy_features']}", flush=True)

    (DATA_META / "data_profile.json").write_text(json.dumps(profile, indent=2),
                                                 encoding="utf-8")
    print(f"\nwrote {DATA_META / 'data_profile.json'} and schema_matrix.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
