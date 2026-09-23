"""Shared paths and run-stamping for every experiment in this repository.

Large corpora and the virtual environment live outside the repository tree, so
multi-gigabyte CSVs are never committed. Results, logs and code stay in the
repository so that every reported number is versioned alongside the manuscript.

Set XAISDN_DATA to point at the directory holding the downloaded corpora.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[2]          # research/ or REVISION_2026/
REPO = PROJECT.parent                                   # repository root

# The same tree is laid out two ways: the numbered working tree the revision was
# produced in, and the flat layout published in the repository. Detect which one
# this copy is sitting in rather than assuming, because a script that resolves to
# a directory that does not exist fails at the point of use, not at import, and
# the mkdir loop below would otherwise scatter empty directories into the tree.
_NUMBERED = (PROJECT / "04_results").is_dir()

if _NUMBERED:
    DATA_META = PROJECT / "02_data"
    EXP = PROJECT / "03_experiments"
    RESULTS = PROJECT / "04_results"
    MANUSCRIPT = PROJECT / "05_manuscript"
    FIGURES = PROJECT / "06_figures"
    LOGS = PROJECT / "08_logs"
    PACKAGE = PROJECT / "09_package"
    SUBMITTED = PROJECT / "00_submitted"
    MATRIX = PROJECT / "01_matrix"
    RESPONSE = PROJECT / "07_response"
else:
    DATA_META = PROJECT / "data_profiles"
    EXP = PROJECT / "experiments"
    RESULTS = PROJECT / "results"
    MANUSCRIPT = PROJECT / "paper"
    FIGURES = PROJECT / "figures"
    LOGS = PROJECT / "logs"
    PACKAGE = PROJECT / "package"
    # Peer-review process material; deliberately not published in the repository.
    SUBMITTED = PROJECT / "_submitted"
    MATRIX = PROJECT / "_matrix"
    RESPONSE = PROJECT / "_response"

DATA_ROOT = Path(os.environ.get("XAISDN_DATA", Path.home() / "xaisdn_data"))
VENV_PY = Path(os.environ.get("XAISDN_PYTHON", sys.executable))

CIC_DDOS = DATA_ROOT / "CICDDoS2019"
SYN_0311 = CIC_DDOS / "03-11" / "Syn.csv"
SYN_0112 = CIC_DDOS / "01-12" / "Syn.csv"
INSDN = DATA_ROOT / "InSDN" / "Dataset.csv"
IDS2017_DDOS = DATA_ROOT / "CICIDS2017" / "Friday-WorkingHours-Afternoon-DDos.csv"
IDS2017_BENIGN = DATA_ROOT / "CICIDS2017" / "Monday-WorkingHours.csv"

SEED = 42
ARCHIVED_SEEDS = [42, 123, 456, 789, 1024]

for _d in (RESULTS, LOGS, FIGURES, DATA_META):
    _d.mkdir(parents=True, exist_ok=True)


def hardware_stamp() -> dict:
    """Hardware and software provenance attached to every result file."""
    hw_file = LOGS / "hardware.json"
    hw = json.loads(hw_file.read_text(encoding="utf-8-sig")) if hw_file.exists() else {}
    libs = {}
    for name in ("numpy", "pandas", "sklearn", "scipy", "shap", "xgboost",
                 "lightgbm", "torch", "matplotlib", "statsmodels"):
        try:
            mod = __import__(name)
            libs[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            libs[name] = None
    return {
        "hardware": hw,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "host": socket.gethostname(),
        "libraries": libs,
        "cpu_only": True,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def log_event(experiment: str, status: str, **kw) -> None:
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "experiment": experiment,
           "status": status, **kw}
    with (LOGS / "experiment_log.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


def save_result(name: str, payload: dict) -> Path:
    """Write a result file with a provenance stamp. Never hand-edit these."""
    out = RESULTS / f"{name}.json"
    payload = {"_provenance": hardware_stamp(), **payload}
    out.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return out
