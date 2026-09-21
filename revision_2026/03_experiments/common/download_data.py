"""Phase 2.1 - Resume-capable dataset download with SHA-256 manifest.

Datasets are stored OUTSIDE the OneDrive tree (see DATA_ROOT) so that multi-GB
CSVs are not synced. Every file is hashed and recorded in MANIFEST.json.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import requests

DATA_ROOT = Path(os.environ.get("XAISDN_DATA", r"C:/Users/adeel/xaisdn_revision/data"))
HF = "https://huggingface.co/datasets/{repo}/resolve/main/{path}"

# (key, hf_repo, hf_path, local_relative_path, priority)
FILES = [
    # --- primary partition used by the submitted paper (exact 1.87 GB file) ---
    ("ddos2019_0311_syn", "baalajimaestro/CICDDoS2019", "03-11/Syn.csv",
     "CICDDoS2019/03-11/Syn.csv", 0),
    # --- cross-vector generalization (R6.1, R7.4) ---
    ("ddos2019_0311_portmap", "baalajimaestro/CICDDoS2019", "03-11/Portmap.csv",
     "CICDDoS2019/03-11/Portmap.csv", 1),
    ("ddos2019_0311_ldap", "baalajimaestro/CICDDoS2019", "03-11/LDAP.csv",
     "CICDDoS2019/03-11/LDAP.csv", 1),
    ("ddos2019_0311_netbios", "baalajimaestro/CICDDoS2019", "03-11/NetBIOS.csv",
     "CICDDoS2019/03-11/NetBIOS.csv", 1),
    ("ddos2019_0311_udp", "baalajimaestro/CICDDoS2019", "03-11/UDP.csv",
     "CICDDoS2019/03-11/UDP.csv", 2),
    ("ddos2019_0311_mssql", "baalajimaestro/CICDDoS2019", "03-11/MSSQL.csv",
     "CICDDoS2019/03-11/MSSQL.csv", 2),
    # --- cross-day generalization (different capture day) ---
    ("ddos2019_0112_syn", "baalajimaestro/CICDDoS2019", "01-12/Syn.csv",
     "CICDDoS2019/01-12/Syn.csv", 1),
    ("ddos2019_0112_udplag", "baalajimaestro/CICDDoS2019", "01-12/UDPLag.csv",
     "CICDDoS2019/01-12/UDPLag.csv", 2),
    # --- cross-dataset validation (R4.1) ---
    ("insdn", "Sharukesh/INSDN", "Dataset.csv", "InSDN/Dataset.csv", 1),
    ("ids2017_ddos", "c01dsnap/CIC-IDS2017",
     "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
     "CICIDS2017/Friday-WorkingHours-Afternoon-DDos.csv", 1),
    ("ids2017_benign_mon", "c01dsnap/CIC-IDS2017", "Monday-WorkingHours.pcap_ISCX.csv",
     "CICIDS2017/Monday-WorkingHours.csv", 1),
]

CHUNK = 1 << 22  # 4 MiB


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for blk in iter(lambda: fh.read(CHUNK), b""):
            h.update(blk)
    return h.hexdigest()


def remote_size(url: str) -> int | None:
    r = requests.head(url, allow_redirects=True, timeout=60)
    n = r.headers.get("content-length") or r.headers.get("x-linked-size")
    return int(n) if n else None


def fetch(url: str, dest: Path) -> dict:
    dest.parent.mkdir(parents=True, exist_ok=True)
    total = remote_size(url)
    have = dest.stat().st_size if dest.exists() else 0
    if total and have == total:
        return {"status": "already-complete", "bytes": have}
    headers = {"Range": f"bytes={have}-"} if have else {}
    mode = "ab" if have else "wb"
    t0 = time.time()
    with requests.get(url, headers=headers, stream=True, timeout=(60, 600)) as r:
        r.raise_for_status()
        with dest.open(mode) as fh:
            for blk in r.iter_content(CHUNK):
                fh.write(blk)
    dt = time.time() - t0
    got = dest.stat().st_size
    return {"status": "ok" if (total is None or got == total) else "size-mismatch",
            "bytes": got, "expected": total, "seconds": round(dt, 1),
            "MBps": round(got / 1e6 / dt, 2) if dt else None}


def main(max_priority: int = 9) -> int:
    manifest_path = DATA_ROOT / "MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for key, repo, path, rel, prio in FILES:
        if prio > max_priority:
            continue
        if manifest.get(key, {}).get("status") == "verified":
            print(f"[skip] {key} already verified")
            continue
        url = HF.format(repo=repo, path=path)
        dest = DATA_ROOT / rel
        print(f"[get ] {key} -> {rel}", flush=True)
        try:
            res = fetch(url, dest)
        except Exception as exc:
            print(f"[FAIL] {key}: {exc}", flush=True)
            manifest[key] = {"status": "failed", "error": str(exc), "url": url}
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            continue
        res.update({"url": url, "local": str(dest), "repo": repo, "remote_path": path})
        print(f"[hash] {key}", flush=True)
        res["sha256"] = sha256(dest)
        res["status"] = "verified" if res["status"] in ("ok", "already-complete") else res["status"]
        manifest[key] = res
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        print(f"[done] {key}  {res['bytes']:,} B  sha256={res['sha256'][:16]}...", flush=True)
    print("MANIFEST:", manifest_path)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 9))
