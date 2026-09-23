"""Phase 0.4 - Freeze the exact software environment used for every experiment."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

LOGS = Path(__file__).resolve().parents[2] / "08_logs"
LOGS.mkdir(parents=True, exist_ok=True)

freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                        capture_output=True, text=True).stdout
(LOGS / "pip_freeze.txt").write_text(freeze, encoding="utf-8")

key = {}
for name in ("numpy", "pandas", "sklearn", "scipy", "shap", "xgboost", "lightgbm",
             "torch", "matplotlib", "seaborn", "statsmodels", "psutil", "networkx",
             "datasketch", "pyarrow"):
    try:
        mod = __import__(name)
        key[name] = getattr(mod, "__version__", "unknown")
    except Exception as exc:
        key[name] = f"UNAVAILABLE ({type(exc).__name__})"

lock = {
    "python": sys.version,
    "python_executable": sys.executable,
    "platform": platform.platform(),
    "machine": platform.machine(),
    "processor": platform.processor(),
    "key_libraries": key,
    "n_pip_packages": len([l for l in freeze.splitlines() if l.strip()]),
}
try:
    import torch
    lock["torch_cuda_available"] = torch.cuda.is_available()
    lock["torch_num_threads"] = torch.get_num_threads()
    lock["torch_build"] = torch.__config__.show().splitlines()[0]
except Exception as exc:
    lock["torch_cuda_available"] = f"UNAVAILABLE ({type(exc).__name__})"

(LOGS / "env_lock.json").write_text(json.dumps(lock, indent=2), encoding="utf-8")
print(json.dumps({k: v for k, v in lock.items() if k != "key_libraries"}, indent=2))
print("key_libraries:", json.dumps(key, indent=2))
