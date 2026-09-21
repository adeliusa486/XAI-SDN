"""Shared paths and run-stamping for every revision experiment.

Data and the venv live OUTSIDE the OneDrive tree so that multi-GB CSVs and
thousands of venv files are never synced. Results, logs and code stay inside
the project so they are versioned with the manuscript.
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

PROJECT = Path(__file__).resolve().parents[2]          # REVISION_2026/
REPO = PROJECT.parent                                   # xai sdn full paper/

SUBMITTED = PROJECT / "00_submitted"
MATRIX = PROJECT / "01_matrix"
DATA_META = PROJECT / "02_data"
EXP = PROJECT / "03_experiments"
RESULTS = PROJECT / "04_results"
MANUSCRIPT = PROJECT / "05_manuscript"
FIGURES = PROJECT / "06_figures"
RESPONSE = PROJECT / "07_response"
LOGS = PROJECT / "08_logs"
PACKAGE = PROJECT / "09_package"

DATA_ROOT = Path(os.environ.get("XAISDN_DATA", r"C:/Users/adeel/xaisdn_revision/data"))
VENV_PY = Path(r"C:/Users/adeel/xaisdn_revision/venv/Scripts/python.exe")

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
    """Hardware + software provenance attached to every result file (R5.3)."""
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
