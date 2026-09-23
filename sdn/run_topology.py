"""Mininet topology, traffic generation and QoS measurement.

Runs inside WSL Ubuntu as root:

    python3 run_topology.py --mode detect --out /opt/xaisdn/mn_detect.json

Topology: one OpenFlow switch, one attacker host, one benign client, one
server. The switch points at a remote controller on 127.0.0.1:6653, which is
the XAI-SDN application.

What it measures. The benign client runs an iperf3 transfer and a ping series
against the server, first with the network idle and then while the attacker
floods. The difference between those two is the QoS impact of the attack, and
running the whole thing with the detector enabled and disabled isolates the QoS
impact of the detector itself. This is the measurement the socket-only testbed
cannot produce, because it has no data plane.
"""
from __future__ import annotations

import argparse
import json
import re
import time

from mininet.clean import cleanup
from mininet.link import TCLink
from mininet.log import setLogLevel
from mininet.net import Mininet
from mininet.node import OVSSwitch, RemoteController
from mininet.topo import Topo


class SingleSwitch(Topo):
    def build(self):
        s1 = self.addSwitch("s1", protocols="OpenFlow13")
        for h, ip in (("h1", "10.0.0.1/8"), ("h2", "10.0.0.2/8"),
                      ("h3", "10.0.0.3/8")):
            self.addLink(self.addHost(h, ip=ip), s1, cls=TCLink,
                         bw=100, delay="1ms")


def parse_ping(out: str) -> dict:
    loss = re.search(r"(\d+(?:\.\d+)?)% packet loss", out)
    rtt = re.search(r"= ([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+) ms", out)
    return {
        "loss_percent": float(loss.group(1)) if loss else None,
        "rtt_min_ms": float(rtt.group(1)) if rtt else None,
        "rtt_avg_ms": float(rtt.group(2)) if rtt else None,
        "rtt_max_ms": float(rtt.group(3)) if rtt else None,
        "rtt_mdev_ms": float(rtt.group(4)) if rtt else None,
    }


def parse_iperf(out: str) -> dict:
    try:
        d = json.loads(out)
        end = d.get("end", {})
        sent = end.get("sum_sent", {})
        recv = end.get("sum_received", {})
        return {
            "throughput_mbps": round(recv.get("bits_per_second", 0) / 1e6, 3),
            "retransmits": sent.get("retransmits"),
            "bytes": recv.get("bytes"),
        }
    except Exception:
        m = re.search(r"([\d.]+)\s+Mbits/sec", out)
        return {"throughput_mbps": float(m.group(1)) if m else None,
                "raw_tail": out[-300:]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="detect",
                    choices=["detect", "nodetect"],
                    help="label only; the controller decides whether it detects")
    ap.add_argument("--out", default="/opt/xaisdn/mn_result.json")
    ap.add_argument("--attack-seconds", type=int, default=20)
    ap.add_argument("--iperf-seconds", type=int, default=10)
    ap.add_argument("--ping-count", type=int, default=30)
    a = ap.parse_args()

    setLogLevel("warning")
    cleanup()
    net = Mininet(topo=SingleSwitch(), switch=OVSSwitch,
                  controller=lambda name: RemoteController(name, ip="127.0.0.1",
                                                           port=6653),
                  link=TCLink, autoSetMacs=True, cleanup=True)
    res: dict = {"mode": a.mode, "topology": "1 switch, 3 hosts, 100 Mbit/s, 1 ms"}
    try:
        net.start()
        h1, h2, h3 = net.get("h1"), net.get("h2"), net.get("h3")   # attacker, client, server
        time.sleep(3)

        res["reachability_before"] = net.pingAll(timeout="1")

        h3.cmd("iperf3 -s -D -1 >/dev/null 2>&1")
        time.sleep(1)

        # --- baseline, network otherwise idle -----------------------------
        res["baseline"] = {
            "ping": parse_ping(h2.cmd(f"ping -c {a.ping_count} -i 0.2 10.0.0.3")),
            "iperf": parse_iperf(h2.cmd(
                f"iperf3 -c 10.0.0.3 -t {a.iperf_seconds} -J 2>/dev/null")),
        }

        # --- under attack ---------------------------------------------------
        h3.cmd("iperf3 -s -D -1 >/dev/null 2>&1")
        time.sleep(1)
        # randomised-source SYN flood, which is the vector the detector targets
        h1.cmd(f"timeout {a.attack_seconds} hping3 --flood --rand-source "
               f"-S -p 80 10.0.0.3 >/dev/null 2>&1 &")
        time.sleep(2)
        res["under_attack"] = {
            "ping": parse_ping(h2.cmd(f"ping -c {a.ping_count} -i 0.2 10.0.0.3")),
            "iperf": parse_iperf(h2.cmd(
                f"iperf3 -c 10.0.0.3 -t {a.iperf_seconds} -J 2>/dev/null")),
        }
        time.sleep(max(0, a.attack_seconds - a.iperf_seconds - 4))
        h1.cmd("pkill hping3 2>/dev/null")

        # --- flow table occupancy left behind -------------------------------
        s1 = net.get("s1")
        dump = s1.cmd("ovs-ofctl -O OpenFlow13 dump-flows s1 2>/dev/null")
        res["flow_table"] = {
            "entries": max(0, len(dump.strip().splitlines()) - 1),
            "drop_rules": dump.count("actions=drop"),
        }

        b, u = res["baseline"], res["under_attack"]
        if b["iperf"].get("throughput_mbps") and u["iperf"].get("throughput_mbps"):
            res["qos_impact"] = {
                "throughput_retained_fraction": round(
                    u["iperf"]["throughput_mbps"] / b["iperf"]["throughput_mbps"], 4),
                "rtt_inflation_factor": round(
                    (u["ping"]["rtt_avg_ms"] or 0) / (b["ping"]["rtt_avg_ms"] or 1), 3),
                "added_loss_percent": round(
                    (u["ping"]["loss_percent"] or 0) - (b["ping"]["loss_percent"] or 0), 3),
            }
        res["status"] = "ok"
    except Exception as exc:
        res["status"] = "failed"
        res["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            net.stop()
        except Exception:
            pass
        cleanup()

    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res, indent=2))
    return 0 if res.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
