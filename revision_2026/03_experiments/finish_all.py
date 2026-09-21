"""Run everything that remains, in an order that keeps measurements clean.

Launched detached so it survives session restarts. Steps run strictly one at a
time, because several of them measure latency or controller CPU and would be
corrupted by anything else using the machine.

  1. wait for the main experiment queue to go idle
  2. re-run the queue, which picks up anything that failed or was interrupted
  3. E4b, the Mininet measurement run
  4. the Mininet demonstration video, written to the user's Downloads folder

Progress goes to 08_logs/finish_all.log.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))
from paths import LOGS, RESULTS  # noqa: E402

PY = sys.executable
MAX_QUEUE_WAIT_S = 4 * 60 * 60


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


def queue_running() -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["cmdline"]):
        try:
            if "run_queue.py" in " ".join(p.info["cmdline"] or []):
                return True
        except Exception:
            pass
    return False


def run(name: str, script: str, timeout: int) -> bool:
    log(f"start {name}")
    t0 = time.time()
    logfile = LOGS / f"{name}.log"
    try:
        with logfile.open("w", encoding="utf-8") as fh:
            rc = subprocess.run([PY, str(HERE / script)], stdout=fh,
                                stderr=subprocess.STDOUT, cwd=str(HERE.parent),
                                timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        log(f"{name} TIMED OUT after {timeout}s")
        return False
    log(f"{name} finished rc={rc} in {time.time()-t0:.0f}s")
    return rc == 0


def main() -> int:
    log("waiting for the main experiment queue to go idle")
    t0 = time.time()
    while queue_running() and time.time() - t0 < MAX_QUEUE_WAIT_S:
        time.sleep(30)
    log("queue idle" if not queue_running() else "queue still busy, continuing anyway")

    # sweep up anything that failed or was interrupted by a restart
    log("re-running the queue to pick up failures")
    with (LOGS / "queue.log").open("a", encoding="utf-8") as fh:
        subprocess.run([PY, str(HERE / "run_queue.py")], stdout=fh,
                       stderr=subprocess.STDOUT, cwd=str(HERE.parent),
                       timeout=MAX_QUEUE_WAIT_S)
    log("queue sweep done")

    done = sorted(p.name for p in RESULTS.glob("*.json"))
    log(f"results present: {len(done)}")

    run("E4b", "e4b_mininet_testbed.py", timeout=3600)
    run("record_demo", "record_demo.py", timeout=1800)

    log("ALL DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
