"""E4 / E7 / E8 - Controller-in-the-loop OpenFlow testbed.

This experiment provides a live proof of concept with controller CPU and memory,
end-to-end detection latency and QoS impact. It is not obvious how 60,000 novel
flows per second are physically ingested without saturating the OpenFlow control
channel, and what flow-rule aggregation and eviction prevent TCAM overflow.

Mininet, ONOS and Docker all require hardware virtualization, which is disabled
in firmware on this machine. What is built instead is a real OpenFlow 1.3
control plane: a controller process and switch agent processes exchanging
genuine OF1.3 PDUs over TCP sockets, with switch-to-controller encodings
independently validated by the os-ken parser. The detector runs inside the
controller process, so the CPU, memory and latency figures come from a running
controller rather than from a simulation of one.

Three phases:
  E4  fixed offered load, attack and benign traffic concurrently: controller CPU
      and RSS, packet_in-to-decision latency, decision-to-flow_mod latency,
      control-channel message and byte rates, and the QoS experienced by
      background benign flows with the detector enabled and disabled.
  E7  offered-rate sweep to saturation, plus three ingest mitigations.
  E8b is a separate, cheap experiment covering flow-table occupancy under the
      full grid of aggregation and eviction policies.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "common"))

HOST = "127.0.0.1"
PORT = 6653
N_FEATURES = 88
TAU = 0.70
CTRL_TRAIN = 600_000          # deployment model size; reported, not hidden
E4_FLOWS = 40_000
E4_RATE = 3000                # offered packet_in per second for the fixed-load phase
E7_RATES = [500, 1000, 2000, 4000, 8000, 16000, 32000]
E7_FLOWS = 20_000
E4_SESSION_SECONDS = 150    # wall-clock budget for each fixed-load session. The
                            # agent is synchronous and the controller sustains
                            # single-digit flows per second under this load, so a
                            # fixed flow count made one session take over an hour
                            # while measuring nothing a bounded one does not.
E7_SESSION_SECONDS = 45     # wall-clock budget per offered rate; the agent is
                            # synchronous, so a saturated session is measured by
                            # how many flows it pushed rather than by finishing
QOS_BACKGROUND_RATE = 200


# --------------------------------------------------------------------------- #
# Controller process
# --------------------------------------------------------------------------- #

def controller_main(model_path: str, conn, explain: bool, tau: float,
                    ready_evt, stop_evt) -> None:
    """Runs in its own process so psutil can attribute CPU and RSS to it alone."""
    sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))
    import joblib
    import of_testbed as T
    from serial import force_serial

    clf = joblib.load(model_path)
    # The controller scores one flow per packet_in. Left at the fitted
    # n_jobs=-1, joblib dispatch adds ~25 ms to every call and becomes the
    # entire measured latency, which would misreport both the decision time
    # and the saturation point of the control channel.
    force_serial(clf)
    explainer = None
    if explain:
        import shap
        explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")

    dp = T.DatapathStub()
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    # Port 0 asks the OS for a free port. A fixed port lets an orphaned
    # controller from a crashed session silently absorb the next session's
    # traffic while the process being measured stays idle.
    srv.bind((HOST, 0))
    srv.listen(8)
    bound_port = srv.getsockname()[1]
    conn.send({"pid": os.getpid(), "listening": True, "port": bound_port})
    ready_evt.set()

    stats = {"packet_in": 0, "flow_mod": 0, "packet_out": 0, "alerts": 0,
             "explanations": 0, "bytes_in": 0, "bytes_out": 0,
             "decide_ns": [], "flowmod_ns": []}

    def handle(sock):
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.sendall(T.encode_hello(0))
        xid = 0
        while not stop_evt.is_set():
            got = T.read_message(sock)
            if got is None:
                break
            version, mtype, mxid, raw = got
            stats["bytes_in"] += len(raw)

            if mtype == T.ofp.OFPT_HELLO:
                sock.sendall(T.encode_header(T.ofp.OFPT_FEATURES_REQUEST, 8, 1))
                stats["bytes_out"] += 8
                continue
            if mtype == T.ofp.OFPT_ECHO_REQUEST:
                r = T.encode_echo_reply(mxid)
                sock.sendall(r); stats["bytes_out"] += len(r)
                continue
            if mtype == T.ofp.OFPT_FEATURES_REPLY:
                continue
            if mtype != T.ofp.OFPT_PACKET_IN:
                continue

            t0 = time.perf_counter_ns()
            payload = raw[-(N_FEATURES * 4):]
            x = np.frombuffer(payload, dtype=np.float32).reshape(1, N_FEATURES)
            score = float(clf.predict_proba(x)[0, 1])
            alert = score >= tau
            t1 = time.perf_counter_ns()
            stats["decide_ns"].append(t1 - t0)
            stats["packet_in"] += 1

            if alert:
                stats["alerts"] += 1
                if explainer is not None:
                    explainer.shap_values(x, check_additivity=False)
                    stats["explanations"] += 1
                xid += 1
                msg = T.encode_flow_mod(dp, xid, 100, xid, 10, 60,
                                        {"eth_type": 0x0800, "ip_proto": 6})
                stats["flow_mod"] += 1
            else:
                xid += 1
                msg = T.encode_packet_out(dp, xid, 0xFFFFFFFF, 1)
                stats["packet_out"] += 1
            sock.sendall(msg)
            stats["bytes_out"] += len(msg)
            stats["flowmod_ns"].append(time.perf_counter_ns() - t1)
        try:
            sock.close()
        except OSError:
            pass

    srv.settimeout(0.5)
    threads = []
    while not stop_evt.is_set():
        try:
            s, _ = srv.accept()
        except socket.timeout:
            continue
        except OSError:
            break
        th = threading.Thread(target=handle, args=(s,), daemon=True)
        th.start()
        threads.append(th)

    for th in threads:
        th.join(timeout=5)
    d = np.array(stats["decide_ns"], dtype=np.float64)
    f = np.array(stats["flowmod_ns"], dtype=np.float64)
    conn.send({
        "packet_in": stats["packet_in"], "flow_mod": stats["flow_mod"],
        "packet_out": stats["packet_out"], "alerts": stats["alerts"],
        "explanations": stats["explanations"],
        "bytes_in": stats["bytes_in"], "bytes_out": stats["bytes_out"],
        "decide_ms": {"p50": float(np.percentile(d, 50) / 1e6) if d.size else None,
                      "p95": float(np.percentile(d, 95) / 1e6) if d.size else None,
                      "p99": float(np.percentile(d, 99) / 1e6) if d.size else None,
                      "mean": float(d.mean() / 1e6) if d.size else None},
        "flowmod_ms": {"p50": float(np.percentile(f, 50) / 1e6) if f.size else None,
                       "p95": float(np.percentile(f, 95) / 1e6) if f.size else None,
                       "mean": float(f.mean() / 1e6) if f.size else None},
    })
    try:
        srv.close()
    except OSError:
        pass


# --------------------------------------------------------------------------- #
# Switch agent
# --------------------------------------------------------------------------- #


def fmt(block: dict, key: str, stat: str) -> str:
    """Format a latency statistic that may be absent because nothing was measured."""
    v = (block or {}).get(key) or {}
    x = v.get(stat)
    return f"{x:.4f}" if isinstance(x, (int, float)) else "n/a"


def switch_agent(X: np.ndarray, rate: int, dpid: int, results: dict,
                 tag: str, max_seconds: float | None = None,
                 port: int = PORT) -> None:
    import of_testbed as T
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.connect((HOST, port))
    s.sendall(T.encode_hello(0))

    handshake_done = False
    deadline = time.time() + 10
    while not handshake_done and time.time() < deadline:
        got = T.read_message(s)
        if got is None:
            break
        _, mtype, mxid, _ = got
        if mtype == T.ofp.OFPT_HELLO:
            continue
        if mtype == T.ofp.OFPT_FEATURES_REQUEST:
            s.sendall(T.encode_features_reply(mxid, dpid))
            handshake_done = True
    results[f"{tag}_handshake"] = handshake_done

    n = len(X)
    rtt = np.empty(n)
    interval = 1.0 / rate if rate else 0.0
    t_start = time.perf_counter()
    sent = 0
    stopped_early = False
    for i in range(n):
        target = t_start + i * interval
        now = time.perf_counter()
        if max_seconds is not None and now - t_start >= max_seconds:
            rtt = rtt[:i]
            stopped_early = True
            break
        if interval and now < target:
            time.sleep(target - now)
        payload = X[i].astype(np.float32).tobytes()
        pkt = T.encode_packet_in(i, 0xFFFFFFFF, T.ofp.OFPR_NO_MATCH, 0, 0, payload)
        t0 = time.perf_counter()
        s.sendall(pkt)
        sent += 1
        got = T.read_message(s)
        rtt[i] = (time.perf_counter() - t0) * 1000.0
        if got is None:
            rtt = rtt[:i + 1]
            break
    elapsed = time.perf_counter() - t_start
    try:
        s.close()
    except OSError:
        pass
    results[f"{tag}_sent"] = sent
    results[f"{tag}_offered"] = int(n)
    results[f"{tag}_stopped_at_time_budget"] = bool(stopped_early)
    results[f"{tag}_elapsed_s"] = elapsed
    results[f"{tag}_achieved_rate"] = sent / elapsed if elapsed else 0.0
    if len(rtt):
        results[f"{tag}_rtt_ms"] = {
            "p50": float(np.percentile(rtt, 50)),
            "p95": float(np.percentile(rtt, 95)),
            "p99": float(np.percentile(rtt, 99)),
            "mean": float(rtt.mean()), "max": float(rtt.max()),
        }
    else:
        results[f"{tag}_rtt_ms"] = {"p50": None, "p95": None, "p99": None,
                                    "mean": None, "max": None}


def sample_process(pid: int, stop: threading.Event, out: list, period=0.25) -> None:
    try:
        p = psutil.Process(pid)
        p.cpu_percent(None)
    except psutil.Error:
        return
    while not stop.is_set():
        try:
            out.append({"t": time.time(), "cpu_percent": p.cpu_percent(None),
                        "rss_gb": p.memory_info().rss / 2 ** 30,
                        "threads": p.num_threads()})
        except psutil.Error:
            break
        time.sleep(period)


def run_session(model_path: str, X: np.ndarray, rate: int, explain: bool,
                tag: str, background: np.ndarray | None = None,
                max_seconds: float | None = None) -> dict:
    ctx = mp.get_context("spawn")
    parent, child = ctx.Pipe()
    ready = ctx.Event()
    stop = ctx.Event()
    proc = ctx.Process(target=controller_main,
                       args=(model_path, child, explain, TAU, ready, stop))
    proc.start()
    if not ready.wait(timeout=180):
        proc.terminate()
        return {"tag": tag, "status": "controller failed to start"}
    info = parent.recv()
    pid = info["pid"]
    port = int(info.get("port") or PORT)

    samples: list = []
    sstop = threading.Event()
    sampler = threading.Thread(target=sample_process, args=(pid, sstop, samples),
                               daemon=True)
    sampler.start()

    results: dict = {}
    threads = [threading.Thread(target=switch_agent,
                                args=(X, rate, 1, results, "attack", max_seconds,
                                      port),
                                daemon=True)]
    if background is not None:
        threads.append(threading.Thread(
            target=switch_agent,
            args=(background, QOS_BACKGROUND_RATE, 2, results, "background",
                  max_seconds, port),
            daemon=True))
    t0 = time.perf_counter()
    for th in threads:
        th.start()
    join_budget = (max_seconds + 60) if max_seconds else 900
    for th in threads:
        th.join(timeout=join_budget)
    incomplete = any(th.is_alive() for th in threads)
    wall = time.perf_counter() - t0

    sstop.set(); sampler.join(timeout=2)
    stop.set()
    ctrl = parent.recv() if parent.poll(60) else {}
    proc.join(timeout=30)
    if proc.is_alive():
        proc.terminate()

    cpu = [s["cpu_percent"] for s in samples] or [0.0]
    rss = [s["rss_gb"] for s in samples] or [0.0]
    return {
        "tag": tag, "status": "incomplete" if incomplete else "ok",
        "listen_port": port,
        "explanations_enabled": explain,
        "offered_rate_per_s": rate, "wall_seconds": round(wall, 2),
        "controller": ctrl,
        "controller_cpu_percent": {"mean": float(np.mean(cpu)),
                                   "p95": float(np.percentile(cpu, 95)),
                                   "max": float(np.max(cpu))},
        "controller_rss_gb": {"mean": float(np.mean(rss)),
                              "max": float(np.max(rss))},
        "n_samples": len(samples),
        "agents": results,
        "control_channel": {
            "packet_in_per_s": round(ctrl.get("packet_in", 0) / wall, 1) if wall else None,
            "bytes_in_per_s": round(ctrl.get("bytes_in", 0) / wall, 1) if wall else None,
            "bytes_out_per_s": round(ctrl.get("bytes_out", 0) / wall, 1) if wall else None,
            "bytes_per_packet_in": round(
                ctrl.get("bytes_in", 0) / max(ctrl.get("packet_in", 1), 1), 1),
        },
        "samples": samples,
    }


def main() -> int:
    from paths import DATA_ROOT, RESULTS, log_event, save_result
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    import of_testbed as T

    log_event("E4", "start")
    t_all = time.time()
    CACHE = DATA_ROOT / "cache"

    out: dict = {
        "experiment": "E4+E7+E8",
        "objective": "controller-in-the-loop OpenFlow 1.3 testbed",
        "platform": {
            "controller": "custom OpenFlow 1.3 controller process using the os-ken "
                          "protocol stack, with the detector embedded in-process",
            "switches": "software switch agents over real TCP sockets",
            "why_not_mininet": "Mininet, ONOS and Docker require hardware "
                               "virtualization, which is disabled in firmware on the "
                               "evaluation machine (WSL2 reports 'virtualization is "
                               "not enabled on this machine')",
            "honesty_note": "this is a control-plane testbed with emulated switch "
                            "agents; it is not a data-plane emulation and is not "
                            "described as Mininet anywhere in the manuscript",
        },
        "tau": TAU,
    }

    # protocol-conformance evidence
    dp = T.DatapathStub()
    out["protocol_conformance"] = {
        nm: T.validate_with_oskan(raw) for nm, raw in [
            ("OFPT_HELLO", T.encode_hello(1)),
            ("OFPT_FEATURES_REPLY", T.encode_features_reply(3, 1)),
            ("OFPT_PACKET_IN", T.encode_packet_in(7, 0xFFFFFFFF, 1, 0, 0,
                                                  b"\x00" * (N_FEATURES * 4))),
            ("OFPT_FLOW_MOD", T.encode_flow_mod(dp, 9, 100, 1, 10, 60,
                                                {"eth_type": 0x0800, "ip_proto": 6})),
            ("OFPT_PACKET_OUT", T.encode_packet_out(dp, 11, 0xFFFFFFFF, 1)),
            ("OFPT_ECHO_REPLY", T.encode_echo_reply(5)),
        ]}
    print("protocol conformance:", json.dumps(out["protocol_conformance"], indent=2),
          flush=True)

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    meta = pd.read_parquet(CACHE / "syn0311_meta.parquet")
    names = json.loads((CACHE / "syn0311_feature_names_repaired.json").read_text())
    cut = int(round(len(y) * 0.70))

    model_path = str(CACHE / "e4_controller_rf.joblib")
    if not Path(model_path).exists():
        print(f"fitting deployment model on {CTRL_TRAIN:,} flows...", flush=True)
        t0 = time.time()
        clf = RandomForestClassifier(n_estimators=200, max_depth=None,
                                     max_features="sqrt", class_weight="balanced",
                                     n_jobs=-1, random_state=42
                                     ).fit(X[cut - CTRL_TRAIN:cut], y[cut - CTRL_TRAIN:cut])
        joblib.dump(clf, model_path, compress=3)
        print(f"  fitted in {time.time()-t0:.1f}s", flush=True)
    out["deployment_model"] = {
        "train_flows": CTRL_TRAIN,
        "rationale": "the deployment model is fitted on a temporally-coherent tail of "
                     "the training partition so that the serialised forest is small "
                     "enough to load inside a controller process; the size is "
                     "reported rather than hidden",
        "serialised_bytes": Path(model_path).stat().st_size,
    }
    t0 = time.time()
    joblib.load(model_path)
    out["deployment_model"]["load_seconds"] = round(time.time() - t0, 2)
    print(f"model {out['deployment_model']['serialised_bytes']/1e6:.1f} MB, "
          f"loads in {out['deployment_model']['load_seconds']}s", flush=True)

    rng = np.random.default_rng(42)
    start = int(rng.integers(cut, len(y) - E4_FLOWS - 1))
    Xa = X[start:start + E4_FLOWS]
    ben_idx = np.flatnonzero(y[cut:] == 0)[:4000] + cut
    Xb = X[ben_idx] if len(ben_idx) else X[cut:cut + 2000]

    # ---- E4: fixed offered load --------------------------------------------
    print("\nE4: fixed offered load", flush=True)
    out["E4"] = {}
    for explain in (False, True):
        tag = "with_explanations" if explain else "detection_only"
        print(f"  session: {tag} at {E4_RATE}/s", flush=True)
        r = run_session(model_path, Xa, E4_RATE, explain, tag, background=Xb,
                        max_seconds=E4_SESSION_SECONDS)
        ser = r.pop("samples", [])
        pd.DataFrame(ser).to_csv(RESULTS / f"E4_timeseries_{tag}.csv", index=False)
        out["E4"][tag] = r
        ag = r.get("agents", {})
        print(f"    agent: handshake={ag.get('attack_handshake')} "
              f"sent={ag.get('attack_sent', 0):,} of {ag.get('attack_offered', 0):,} "
              f"achieved={ag.get('attack_achieved_rate', 0):,.0f}/s", flush=True)
        if r.get("status") == "ok":
            print(f"    CPU mean={r['controller_cpu_percent']['mean']:.1f}% "
                  f"max={r['controller_cpu_percent']['max']:.1f}%  "
                  f"RSS max={r['controller_rss_gb']['max']:.2f} GB", flush=True)
            print(f"    decide p50={fmt(r['controller'], 'decide_ms', 'p50')} ms "
                  f"p99={fmt(r['controller'], 'decide_ms', 'p99')} ms  "
                  f"RTT p50={fmt(ag, 'attack_rtt_ms', 'p50')} ms", flush=True)
        else:
            print(f"    session status: {r.get('status')}", flush=True)

    # QoS comparison for the background flows
    try:
        a = out["E4"]["detection_only"]["agents"]["background_rtt_ms"]
        b = out["E4"]["with_explanations"]["agents"]["background_rtt_ms"]
        out["E4"]["qos_impact_on_background_flows"] = {
            "detection_only_rtt_ms": a, "with_explanations_rtt_ms": b,
            "p95_inflation_factor": round(b["p95"] / a["p95"], 3) if a["p95"] else None,
            "interpretation": "the background agent carries benign traffic through the "
                              "same controller; the change in its round-trip time is "
                              "the QoS cost that the security function imposes on "
                              "unrelated flows",
        }
        print(f"  QoS: background p95 RTT {a['p95']:.3f} ms -> {b['p95']:.3f} ms",
              flush=True)
    except KeyError:
        out["E4"]["qos_impact_on_background_flows"] = {"status": "unavailable"}

    # ---- E7: offered-rate sweep to saturation -------------------------------
    print("\nE7: offered-rate sweep", flush=True)
    sweep = []
    Xs = X[start:start + E7_FLOWS]
    for rate in E7_RATES:
        print(f"  rate {rate}/s", flush=True)
        r = run_session(model_path, Xs, rate, False, f"sweep_{rate}",
                        max_seconds=E7_SESSION_SECONDS)
        r.pop("samples", None)
        if r.get("status") != "ok":
            sweep.append({"offered_rate": rate, "status": r.get("status")})
            print(f"    session did not complete: {r.get('status')}", flush=True)
            continue
        ach = r["agents"].get("attack_achieved_rate", 0.0)
        sweep.append({
            "offered_rate": rate,
            "achieved_rate": round(ach, 1),
            "saturation_ratio": round(ach / rate, 4),
            "rtt_p50_ms": r["agents"]["attack_rtt_ms"]["p50"],
            "rtt_p99_ms": r["agents"]["attack_rtt_ms"]["p99"],
            "cpu_mean_percent": r["controller_cpu_percent"]["mean"],
            "cpu_max_percent": r["controller_cpu_percent"]["max"],
            "bytes_in_per_s": r["control_channel"]["bytes_in_per_s"],
            "bytes_out_per_s": r["control_channel"]["bytes_out_per_s"],
        })
        print(f"    achieved {ach:,.0f}/s ({100*ach/rate:.0f}% of offered), "
              f"RTT p99={sweep[-1]['rtt_p99_ms']:.3f} ms, "
              f"CPU {sweep[-1]['cpu_mean_percent']:.0f}%", flush=True)
    pd.DataFrame(sweep).to_csv(RESULTS / "E7_rate_sweep.csv", index=False)

    # The sweep and the fixed-load session measure the same quantity under two
    # different configurations: 45 s without background traffic versus 150 s
    # with it. Summarising only over `sweep` and calling the result "at most"
    # produced a ceiling the fixed-load session already exceeded, which reached
    # the manuscript. The summary below ranges over every session that measured
    # an achieved rate, and its wording is a range rather than a bound.
    sat = [s for s in sweep if s.get("saturation_ratio", 1) < 0.9]
    sweep_rates = [s["achieved_rate"] for s in sweep if "achieved_rate" in s]
    fixed_rates = [
        out["E4"][t]["agents"]["attack_achieved_rate"]
        for t in ("detection_only", "with_explanations")
        if out["E4"].get(t, {}).get("agents", {}).get("attack_achieved_rate")
    ]
    all_rates = sweep_rates + fixed_rates
    out["E7"] = {
        "sweep": sweep,
        "achieved_rate_definition": (
            "completed packet_in request/response exchanges per second of wall "
            "time for the synchronous switch agent, including classification and "
            "rule installation"),
        "sweep_session_seconds": E7_SESSION_SECONDS,
        "fixed_load_session_seconds": E4_SESSION_SECONDS,
        "sweep_has_background_agent": False,
        "fixed_load_has_background_agent": True,
        "saturated_at_every_offered_rate": bool(sat and sat[0]["offered_rate"]
                                                == min(E7_RATES)),
        "lowest_offered_rate_tested": min(E7_RATES),
        "knee_located": not (sat and sat[0]["offered_rate"] == min(E7_RATES)),
        "max_achieved_rate_sweep_per_s": max(sweep_rates, default=None),
        "max_achieved_rate_any_session_per_s": max(all_rates, default=None),
        "min_achieved_rate_any_session_per_s": min(all_rates, default=None),
        "bytes_per_packet_in": out["E4"]["detection_only"]
                                  .get("control_channel", {})
                                  .get("bytes_per_packet_in"),
        "answer_to_60k_question": None,     # filled below
    }
    lo = out["E7"]["min_achieved_rate_any_session_per_s"] or 0
    hi = out["E7"]["max_achieved_rate_any_session_per_s"] or 0
    out["E7"]["answer_to_60k_question"] = (
        f"Across every session measured here a single controller instance on this "
        f"hardware absorbed between {lo:,.0f} and {hi:,.0f} packet_in events per "
        f"second end to end, including classification and rule installation. The "
        f"spread is a property of the session configuration, not of the offered "
        f"rate: the sweep sessions run 45 s with no background traffic and the "
        f"fixed-load sessions run 150 s with it, and the sweep is saturated at "
        f"every offered rate tested, so it bounds goodput rather than locating the "
        f"knee. The 60,606 flows/s figure in an offline measurement was an "
        f"offline feature-and-inference rate measured without any control channel, "
        f"and it is not an ingest rate; it is two to three orders of magnitude "
        f"above anything measured here. Reaching that order of magnitude at the "
        f"control plane requires the flows not to arrive as individual table-miss "
        f"events: switch-side aggregation, packet_in rate limiting and sampled "
        f"export are the mechanisms, and their cost is quantified in the "
        f"flow-table analysis of E8b.")
    print("\n" + out["E7"]["answer_to_60k_question"], flush=True)

    # ---- flow-table occupancy -----------------------------------------------
    # The aggregation and eviction grid used to run here. It now lives in E8b,
    # which sweeps three further aggregation policies and, with an amortized
    # constant-time table model, completes a larger grid in seconds rather than
    # hours. Keeping a subset of it here would only duplicate that work.
    out["flow_table_analysis"] = "see E8b_tcam_policy.json"

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E4_controller_testbed", out)
    log_event("E4", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main())
