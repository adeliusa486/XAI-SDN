"""Sequential experiment runner.

Runs the Phase 3 programme in dependency order, one experiment at a time so that
no run contaminates another's timing measurements. Each experiment's stdout goes
to 08_logs/<ID>.log and its exit status to 08_logs/queue_status.json, so the
queue can be resumed after an interruption without repeating completed work.

Usage:
    python run_queue.py                 run every pending experiment
    python run_queue.py E5 E6           run only these
    python run_queue.py --force E5      re-run even if already complete
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))
from paths import LOGS, RESULTS  # noqa: E402

ORDER = [
    ("E5",  "e5_leakage_audit.py",              "E5_leakage_audit.json"),
    ("E5b", "e5b_duplicate_leakage.py",         "E5b_duplicate_leakage.json"),
    ("E17", "e17_entropy_contribution.py",      "E17_entropy_contribution.json"),
    ("E12", "e12_pr_threshold.py",              "E12_pr_threshold.json"),
    ("E11", "e11_entropy_numerics.py",          "E11_entropy_numerics.json"),
    ("E6",  "e6_collinearity_shap_stability.py","E6_collinearity_shap_stability.json"),
    ("E15", "e15_protocol.py",                  "E15_protocol.json"),
    ("E1",  "e1_baselines_cpu.py",              "E1_baselines_cpu.json"),
    ("E9",  "e9_deep_baselines.py",             "E9_deep_baselines.json"),
    ("E14", "e14_roc_pr_curves.py",             "E14_roc_pr_curves.json"),
    ("E13", "e13_case_studies.py",              "E13_case_studies.json"),
    ("E10", "e10_explanation_budget.py",        "E10_explanation_budget.json"),
    ("E3",  "e3_cross_vector.py",               "E3_cross_vector.json"),
    ("E2",  "e2_cross_dataset.py",              "E2_cross_dataset.json"),
    ("E4",  "e4_controller_testbed.py",         "E4_controller_testbed.json"),
    ("E8b", "e8b_tcam_policy.py",              "E8b_tcam_policy.json"),
    ("E19", "e19_shap_sample.py",              "E19_shap_sample.json"),
    ("E20", "e20_few_shot_transfer.py",       "E20_few_shot_transfer.json"),
    ("E21", "e21_entropy_under_shift.py",     "E21_entropy_under_shift.json"),
    ("E22", "e22_model_size_pareto.py",       "E22_model_size_pareto.json"),
]

STATUS = LOGS / "queue_status.json"


def load_status() -> dict:
    return json.loads(STATUS.read_text()) if STATUS.exists() else {}


def save_status(st: dict) -> None:
    STATUS.write_text(json.dumps(st, indent=2), encoding="utf-8")


def main(argv: list[str]) -> int:
    force = "--force" in argv
    wanted = [a for a in argv if not a.startswith("--")]
    st = load_status()

    for eid, script, result_name in ORDER:
        if wanted and eid not in wanted:
            continue
        # A result is only reusable if it postdates the script that produced it.
        # Editing an experiment after it ran and then skipping it on the next
        # sweep is how stale numbers reach a manuscript, so the check is on
        # modification time rather than on existence alone.
        res = RESULTS / result_name
        if not force and res.exists():
            if res.stat().st_mtime >= (HERE / script).stat().st_mtime:
                print(f"[skip] {eid}: {result_name} already present", flush=True)
                st.setdefault(eid, {})["status"] = "already-complete"
                save_status(st)
                continue
            print(f"[stale] {eid}: {script} is newer than {result_name}, re-running",
                  flush=True)

        log = LOGS / f"{eid}.log"
        print(f"\n{'='*70}\n[run ] {eid}  ->  {log}\n{'='*70}", flush=True)
        t0 = time.time()
        st[eid] = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "script": script, "log": str(log)}
        save_status(st)

        with log.open("w", encoding="utf-8") as fh:
            proc = subprocess.run([sys.executable, str(HERE / script)],
                                  stdout=fh, stderr=subprocess.STDOUT,
                                  cwd=str(HERE.parent),
                                  env={**__import__("os").environ,
                                       "PYTHONIOENCODING": "utf-8",
                                       "PYTHONUNBUFFERED": "1"})
        dt = round(time.time() - t0, 1)
        ok = proc.returncode == 0 and (RESULTS / result_name).exists()
        st[eid].update({"status": "done" if ok else "failed",
                        "returncode": proc.returncode, "runtime_s": dt,
                        "finished": time.strftime("%Y-%m-%dT%H:%M:%S")})
        save_status(st)
        print(f"[{'done' if ok else 'FAIL'}] {eid} rc={proc.returncode} in {dt}s",
              flush=True)
        if not ok:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            print("---- tail of log ----", flush=True)
            for line in tail:
                print("   ", line, flush=True)
            print("---- continuing to the next experiment ----", flush=True)

    print("\nqueue finished")
    print(json.dumps({k: v.get("status") for k, v in load_status().items()}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
