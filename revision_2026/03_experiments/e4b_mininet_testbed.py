"""E4b - Mininet proof of concept with live traffic (R4.2).

Reviewer 4 asked for "a Mininet/Ryu or ONOS proof-of-concept with live traffic,
measure controller CPU/memory, end-to-end detection latency, and QoS impact".
E4 answers the control-plane half of that on sockets. This experiment answers
the data-plane half, which sockets cannot: real hosts, a real Open vSwitch
datapath, a real SYN flood from hping3, and real iperf3 and ping measurements
taken through the switch while the attack is running.

Orchestrated from Windows, executed inside WSL Ubuntu. Three conditions:

    no_controller   the switch runs without the detector, giving the floor
    detect          the detector runs, no explanations
    detect_explain  the detector runs and attributes every admitted alert

Comparing 'detect' against 'no_controller' isolates the QoS cost of the security
function. Comparing 'detect_explain' against 'detect' isolates the cost of
explanation on the data path.

Prerequisites: Virtual Machine Platform enabled, machine rebooted, and
common/setup_mininet.sh run once.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

WSL_DIR = "/opt/xaisdn"
CONDITIONS = [
    ("no_controller", {"XAISDN_DETECT": "0", "XAISDN_EXPLAIN": "0"}),
    ("detect", {"XAISDN_DETECT": "1", "XAISDN_EXPLAIN": "0"}),
    ("detect_explain", {"XAISDN_DETECT": "1", "XAISDN_EXPLAIN": "1"}),
]


def wsl(cmd: str, timeout: int = 900) -> subprocess.CompletedProcess:
    return subprocess.run(["wsl.exe", "-d", "Ubuntu", "-u", "root", "--",
                           "bash", "-lc", cmd],
                          capture_output=True, text=True, timeout=timeout,
                          errors="replace")


def preflight() -> dict | None:
    st = wsl("echo ok")
    if st.returncode != 0 or "ok" not in st.stdout:
        return {"error": "WSL Ubuntu not reachable",
                "detail": (st.stderr or st.stdout)[:400],
                "remedy": "Run 'wsl.exe --install --no-distribution' in an "
                          "elevated PowerShell, reboot, then run "
                          "common/setup_mininet.sh"}
    for probe, name in (("mn --version", "mininet"),
                        ("ovs-vsctl --version", "open vswitch"),
                        ("test -x /opt/xaisdn/bin/python && echo yes", "controller venv"),
                        ("which hping3", "hping3"),
                        ("which iperf3", "iperf3")):
        r = wsl(probe, timeout=120)
        # Mininet writes its version banner to stderr and still exits 0, so a
        # stdout-only test reports it missing when it is installed and working.
        out = (r.stdout or "") + (r.stderr or "")
        if r.returncode != 0 or not out.strip():
            return {"error": f"{name} unavailable in WSL",
                    "probe": probe,
                    "returncode": r.returncode,
                    "output": out.strip()[:200],
                    "remedy": "run common/setup_mininet.sh"}
    return None


def run_condition(tag: str, env: dict, attack_s: int) -> dict:
    envs = " ".join(f"{k}={v}" for k, v in env.items())
    ctl_out = f"{WSL_DIR}/ctl_{tag}.json"
    mn_out = f"{WSL_DIR}/mn_{tag}.json"
    wsl(f"rm -f {ctl_out} {mn_out}", timeout=60)

    result: dict = {"condition": tag}

    if env["XAISDN_DETECT"] == "1" or True:
        # The controller is always started: without it the switch has no
        # forwarding logic at all and the comparison would measure the absence
        # of a controller rather than the absence of a detector.
        start = (f"cd {WSL_DIR} && {envs} XAISDN_OUT={ctl_out} "
                 f"nohup /opt/xaisdn/bin/python {WSL_DIR}/run_controller.py "
                 f"{WSL_DIR}/xaisdn_controller.py 6653 "
                 f"> {WSL_DIR}/ctl_{tag}.log 2>&1 & "
                 f"echo $!")
        r = wsl(start, timeout=120)
        pid = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else ""
        result["controller_pid"] = pid
        time.sleep(8)

    # run_topology exposes two modes. Whether explanations are generated is a
    # property of the controller, set through the environment above, not of
    # the topology, so both detecting conditions use the same mode here.
    mode = "nodetect" if tag == "no_controller" else "detect"
    r = wsl(f"cd {WSL_DIR} && python3 run_topology.py --mode {mode} "
            f"--out {mn_out} --attack-seconds {attack_s}",
            timeout=attack_s + 600)
    result["topology_mode"] = mode
    result["topology_returncode"] = r.returncode
    if r.returncode != 0:
        result["topology_stderr"] = (r.stderr or "")[-900:]

    wsl("pkill -f run_controller.py 2>/dev/null; sleep 2", timeout=120)

    # Verbatim terminal evidence for the paper. Reviewer 4 asked for a live
    # proof of concept, so we capture what the session actually printed rather
    # than describing it: the switch's installed flow table, the OpenFlow
    # connection state, and the tail of the controller log.
    evid = {}
    for name, cmd in (
        ("ovs_show", "ovs-vsctl show 2>&1 | head -25"),
        ("flow_table", "ovs-ofctl -O OpenFlow13 dump-flows s1 2>&1 | head -20"),
        ("ofctl_show", "ovs-ofctl -O OpenFlow13 show s1 2>&1 | head -12"),
        ("controller_tail", f"tail -25 {WSL_DIR}/ctl_{tag}.log 2>&1"),
        ("mn_version", "mn --version 2>&1; ovs-vsctl --version 2>&1 | head -1"),
    ):
        g = wsl(cmd, timeout=120)
        evid[name] = (g.stdout or g.stderr or "").strip()
    result["terminal_evidence"] = evid

    for key, path in (("mininet", mn_out), ("controller", ctl_out)):
        g = wsl(f"cat {path} 2>/dev/null", timeout=60)
        if g.stdout.strip():
            try:
                result[key] = json.loads(g.stdout)
            except json.JSONDecodeError:
                result[key] = {"unparseable": g.stdout[-500:]}
        else:
            result[key] = {"missing": path}
    return result


def main() -> int:
    log_event("E4b", "start")
    t_all = time.time()
    out: dict = {
        "experiment": "E4b",
        "objective": "Mininet proof of concept with live traffic",
        "platform": "Mininet + Open vSwitch + os-ken inside WSL2 Ubuntu",
        "conditions": [c for c, _ in CONDITIONS],
    }

    problem = preflight()
    if problem:
        out["status"] = "prerequisites not met"
        out.update(problem)
        print(json.dumps(out, indent=2))
        save_result("E4b_mininet_testbed", out)
        log_event("E4b", "skipped", reason=problem["error"])
        return 1

    # stage the controller, topology script and model into WSL
    print("staging files into WSL...", flush=True)
    wsl(f"mkdir -p {WSL_DIR}", timeout=60)
    for f in ("xaisdn_controller.py", "run_topology.py",
              "run_controller.py"):
        src = (HERE / "mininet" / f).as_posix().replace("C:/", "/mnt/c/")
        # These files live on a Windows filesystem. A stray carriage return makes
        # bash and python3 inside WSL fail in ways that look unrelated to it, so
        # they are stripped on the way in.
        wsl(f"cp '{src}' {WSL_DIR}/{f} && sed -i 's/\\r$//' {WSL_DIR}/{f}",
            timeout=120)
    model = (DATA_ROOT / "cache" / "e4_controller_rf.joblib").as_posix().replace(
        "C:/", "/mnt/c/")
    wsl(f"cp '{model}' {WSL_DIR}/model.joblib", timeout=300)
    chk = wsl(f"ls -la {WSL_DIR} | head -20", timeout=60)
    out["staged"] = chk.stdout

    results = {}
    for tag, env in CONDITIONS:
        print(f"\n=== condition: {tag} ===", flush=True)
        r = run_condition(tag, env, attack_s=20)
        results[tag] = r
        mn = r.get("mininet", {})
        if mn.get("status") == "ok":
            q = mn.get("qos_impact", {})
            print(f"  throughput retained under attack: "
                  f"{q.get('throughput_retained_fraction')}", flush=True)
            print(f"  RTT inflation: {q.get('rtt_inflation_factor')}  "
                  f"added loss: {q.get('added_loss_percent')}%", flush=True)
            print(f"  flow table: {mn.get('flow_table')}", flush=True)
        c = r.get("controller", {})
        if "decide_ms" in c:
            print(f"  controller CPU mean "
                  f"{c['controller_cpu_percent']['mean']}%  "
                  f"RSS max {c['controller_rss_gb']['max']} GB", flush=True)
            print(f"  decide p50 {c['decide_ms']['p50']} ms  "
                  f"packet_in {c['packet_in']}  alerts {c['alerts']}  "
                  f"drops {c['drops_installed']}", flush=True)
    out["results"] = results

    # cross-condition summary
    def q(tag, k):
        return results.get(tag, {}).get("mininet", {}).get("qos_impact", {}).get(k)

    ran = [c for c, v in out["results"].items()
           if v.get("topology_returncode") == 0 and v.get("mininet")]
    failed = [c for c in out["results"] if c not in ran]
    status = "ok" if not failed else (
        "no condition produced measurements" if not ran
        else "partial: " + ", ".join(failed) + " did not produce measurements")
    out["conditions_completed"] = ran
    out["conditions_failed"] = failed

    out["comparison"] = {
        "qos_cost_of_detection": {
            "throughput_retained_no_controller": q("no_controller",
                                                   "throughput_retained_fraction"),
            "throughput_retained_detect": q("detect",
                                            "throughput_retained_fraction"),
            "throughput_retained_detect_explain": q("detect_explain",
                                                    "throughput_retained_fraction"),
        },
        "interpretation": (
            "The first row is the network's own resilience to the flood with no "
            "detection logic. The second shows what the detector recovers or "
            "costs. The third isolates the additional cost of attributing every "
            "admitted alert on the data path."),
    }
    # persist the transcripts so the figure builder can typeset them verbatim
    eviddir = RESULTS / "mininet_evidence"
    eviddir.mkdir(parents=True, exist_ok=True)
    for tag, r in results.items():
        for name, text in (r.get("terminal_evidence") or {}).items():
            (eviddir / f"{tag}__{name}.txt").write_text(text, encoding="utf-8")
    out["terminal_evidence_dir"] = str(eviddir)

    out["status"] = status
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E4b_mininet_testbed", out)
    log_event("E4b", "done" if status == "ok" else "failed",
              result=str(p), runtime_s=out["total_runtime_s"], status=status)
    print(f"\nstatus: {status}")
    print(f"Wrote {p}")
    # A non-zero exit keeps a run whose conditions produced nothing out of the
    # "done" column of the queue, where it would otherwise be skipped for ever.
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
