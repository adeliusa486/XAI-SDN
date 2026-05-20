"""
mininet_topo.py — Mininet Test Topology for XAI-SDN.

Creates a simple fat-tree-inspired topology:
  - 1 core switch (s1)
  - 2 aggregation switches (s2, s3)
  - 4 edge switches (s4–s7)
  - 8 hosts (h1–h8, 2 per edge switch)
  - 1 attacker host (atk)

Run:
    sudo python sdn/topology/mininet_topo.py --controller 127.0.0.1:6653
    sudo python sdn/topology/mininet_topo.py --attack udp --duration 30
"""

from __future__ import annotations

import argparse
import os
import sys
import time

MININET_AVAILABLE = False
try:
    from mininet.net import Mininet
    from mininet.node import Controller, OVSKernelSwitch, RemoteController
    from mininet.cli import CLI
    from mininet.log import setLogLevel
    from mininet.link import TCLink
    MININET_AVAILABLE = True
except ImportError:
    pass


def build_topology(controller_ip: str = "127.0.0.1", controller_port: int = 6653):
    """Build and return a Mininet network with remote Ryu controller.

    Topology:
        atk -- s1 (core)
                ├── s2 (agg)
                │   ├── s4 -- h1, h2
                │   └── s5 -- h3, h4
                └── s3 (agg)
                    ├── s6 -- h5, h6
                    └── s7 -- h7, h8
    """
    if not MININET_AVAILABLE:
        print("Mininet not available. Install: sudo apt-get install mininet")
        sys.exit(1)

    setLogLevel("info")

    net = Mininet(
        controller=RemoteController,
        switch=OVSKernelSwitch,
        link=TCLink,
        autoSetMacs=True,
    )

    # Remote Ryu controller
    c0 = net.addController(
        "c0",
        controller=RemoteController,
        ip=controller_ip,
        port=controller_port,
    )

    # Core switch
    s1 = net.addSwitch("s1", protocols="OpenFlow13")

    # Aggregation switches
    s2 = net.addSwitch("s2", protocols="OpenFlow13")
    s3 = net.addSwitch("s3", protocols="OpenFlow13")

    # Edge switches
    s4 = net.addSwitch("s4", protocols="OpenFlow13")
    s5 = net.addSwitch("s5", protocols="OpenFlow13")
    s6 = net.addSwitch("s6", protocols="OpenFlow13")
    s7 = net.addSwitch("s7", protocols="OpenFlow13")

    # Legitimate hosts
    hosts = []
    for i in range(1, 9):
        h = net.addHost(f"h{i}", ip=f"10.0.0.{i}/24")
        hosts.append(h)

    # Attacker host
    atk = net.addHost("atk", ip="10.0.1.100/24")

    # Core links (1 Gbps)
    net.addLink(s1, s2, bw=1000)
    net.addLink(s1, s3, bw=1000)

    # Aggregation links (1 Gbps)
    net.addLink(s2, s4, bw=1000)
    net.addLink(s2, s5, bw=1000)
    net.addLink(s3, s6, bw=1000)
    net.addLink(s3, s7, bw=1000)

    # Host links (100 Mbps)
    net.addLink(s4, hosts[0], bw=100)
    net.addLink(s4, hosts[1], bw=100)
    net.addLink(s5, hosts[2], bw=100)
    net.addLink(s5, hosts[3], bw=100)
    net.addLink(s6, hosts[4], bw=100)
    net.addLink(s6, hosts[5], bw=100)
    net.addLink(s7, hosts[6], bw=100)
    net.addLink(s7, hosts[7], bw=100)

    # Attacker link to core (simulates botnet traffic)
    net.addLink(s1, atk, bw=1000)

    return net, c0, hosts, atk


def run_attack(
    atk,
    target_ip: str = "10.0.0.1",
    attack_type: str = "udp",
    duration: int = 30,
) -> None:
    """Launch a DDoS attack from the attacker host.

    Args:
        atk: Mininet attacker host object.
        target_ip: Target IP address.
        attack_type: One of: udp, tcp, icmp, http, slowloris.
        duration: Attack duration in seconds.
    """
    print(f"\n[ATTACK] Starting {attack_type.upper()} flood → {target_ip} for {duration}s")
    print("[ATTACK] This is a SIMULATION for research purposes only.\n")

    if attack_type == "udp":
        # hping3 UDP flood
        cmd = f"hping3 --udp -p 53 --flood --rand-source {target_ip} &"
    elif attack_type == "tcp":
        # SYN flood
        cmd = f"hping3 -S --flood --rand-source -p 80 {target_ip} &"
    elif attack_type == "icmp":
        # ICMP flood
        cmd = f"hping3 --icmp --flood --rand-source {target_ip} &"
    elif attack_type == "http":
        # HTTP GET flood using wrk (requires wrk installed)
        cmd = f"wrk -t4 -c100 -d{duration}s http://{target_ip}/ &"
    elif attack_type == "slowloris":
        # Slowloris using slowhttptest
        cmd = f"slowhttptest -c 1000 -H -i 10 -r 200 -t GET -u http://{target_ip}/ -x 24 &"
    else:
        print(f"Unknown attack type: {attack_type}")
        return

    atk.cmd(cmd)
    print(f"[ATTACK] Running for {duration}s...")
    time.sleep(duration)
    atk.cmd("pkill hping3; pkill wrk; pkill slowhttptest")
    print("[ATTACK] Attack stopped.")


def main():
    parser = argparse.ArgumentParser(description="XAI-SDN Mininet Test Topology")
    parser.add_argument("--controller", default="127.0.0.1:6653",
                        help="Ryu controller address (default: 127.0.0.1:6653)")
    parser.add_argument("--attack", default=None,
                        choices=["udp", "tcp", "icmp", "http", "slowloris"],
                        help="Start an attack after topology initialization")
    parser.add_argument("--target", default="10.0.0.1",
                        help="Attack target IP (default: 10.0.0.1)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Attack duration in seconds (default: 30)")
    parser.add_argument("--cli", action="store_true",
                        help="Drop into Mininet CLI after setup")
    args = parser.parse_args()

    ctrl_ip, ctrl_port = args.controller.split(":")

    print("\n" + "="*60)
    print("XAI-SDN Mininet Test Topology")
    print("="*60)
    print(f"Controller: {ctrl_ip}:{ctrl_port}")
    print(f"Attack:     {args.attack or 'none'}")
    print("="*60 + "\n")

    net, c0, hosts, atk = build_topology(ctrl_ip, int(ctrl_port))
    net.start()

    print("Topology started. Waiting 3s for switch negotiation...")
    time.sleep(3)
    net.pingAll()

    if args.attack:
        run_attack(atk, args.target, args.attack, args.duration)

    if args.cli:
        CLI(net)

    net.stop()


if __name__ == "__main__":
    main()
