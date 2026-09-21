"""Second finisher: the experiments the first sweep never saw.

run_queue reads its ORDER once at start, so E8b, E19, E20, E21 and E22, which
were added while the first sweep was already running, were never picked up. E4
and E14 also still need a run: both were stopped deliberately and both have since
been rewritten, so the modification-time check in run_queue will re-run them.

This waits for the demonstration recording to finish first, because that captures
a live Mininet session and anything else using the machine would distort it.

Progress goes to 08_logs/finish2.log.
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
MAX_WAIT_S = 45 * 60


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def running(needle: str) -> bool:
    try:
        import psutil
    except ImportError:
        return False
    for p in psutil.process_iter(["cmdline"]):
        try:
            if needle in " ".join(p.info["cmdline"] or []):
                return True
        except Exception:
            pass
    return False


def main() -> int:
    log("waiting for the demonstration recording to finish")
    t0 = time.time()
    while running("record_demo.py") and time.time() - t0 < MAX_WAIT_S:
        time.sleep(20)
    log("recording finished" if not running("record_demo.py")
        else "recording still running, continuing anyway")

    log("running the queue for the experiments the first sweep did not see")
    with (LOGS / "queue.log").open("a", encoding="utf-8") as fh:
        subprocess.run([PY, str(HERE / "run_queue.py")], stdout=fh,
                       stderr=subprocess.STDOUT, cwd=str(HERE.parent),
                       timeout=6 * 60 * 60)

    log("retrying the Mininet testbed now that its preflight is fixed")
    with (LOGS / "E4b.log").open("w", encoding="utf-8") as fh:
        rc = subprocess.run([PY, str(HERE / "e4b_mininet_testbed.py")], stdout=fh,
                            stderr=subprocess.STDOUT, cwd=str(HERE.parent),
                            timeout=3600).returncode
    log(f"E4b finished rc={rc}")

    done = sorted(p.stem for p in RESULTS.glob("*.json"))
    log(f"results present: {len(done)}")
    log("ALL DONE (finish2)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
