"""XAI-SDN controller application for the Mininet testbed.

Runs under os-ken (the maintained Ryu fork) inside WSL. Installed as:

    /opt/xaisdn/bin/osken-manager --ofp-tcp-listen-port 6653 xaisdn_controller.py

Behaviour. The switch is driven as a learning switch so that ordinary traffic
forwards and the QoS measurement is meaningful. Every table-miss packet_in is
additionally turned into a feature vector, scored by the Random Forest, and, if
the score reaches the threshold and the explanation budget admits it, attributed
with TreeSHAP. Flows scored as attacks receive a drop rule; everything else gets
a normal forwarding rule. Per-event timings, controller resource use and rule
counts are written to a JSON file when the process is signalled to stop.

The detector is forced to serial execution. A controller scores one flow per
event, and joblib's parallel dispatch would otherwise add roughly 25 ms to every
call and become the entire measured latency.
"""
from __future__ import annotations

import json
import os
import signal
import time
from collections import defaultdict

import numpy as np
import psutil

from os_ken.base import app_manager
from os_ken.controller import ofp_event
from os_ken.controller.handler import CONFIG_DISPATCHER, MAIN_DISPATCHER, set_ev_cls
from os_ken.lib.packet import ether_types, ethernet, ipv4, packet, tcp, udp
from os_ken.ofproto import ofproto_v1_3

MODEL_PATH = os.environ.get("XAISDN_MODEL", "/opt/xaisdn/model.joblib")
OUT_PATH = os.environ.get("XAISDN_OUT", "/opt/xaisdn/controller_metrics.json")
TAU = float(os.environ.get("XAISDN_TAU", "0.70"))
EXPLAIN = os.environ.get("XAISDN_EXPLAIN", "0") == "1"
DETECT = os.environ.get("XAISDN_DETECT", "1") == "1"
N_FEATURES = 88
IDLE_TIMEOUT = int(os.environ.get("XAISDN_IDLE", "10"))
HARD_TIMEOUT = int(os.environ.get("XAISDN_HARD", "60"))


class XaiSdn(app_manager.OSKenApp):
    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mac_to_port = {}
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        self.t_start = time.time()

        self.m = {
            "packet_in": 0, "flow_mod": 0, "packet_out": 0,
            "alerts": 0, "explanations": 0, "drops_installed": 0,
            "decide_ns": [], "install_ns": [],
            "bytes_in": 0, "cpu_samples": [], "rss_samples": [],
            "detect_enabled": DETECT, "explain_enabled": EXPLAIN,
        }
        self.flow_first_seen: dict = {}
        self.flow_pkts: dict = defaultdict(int)
        self.flow_bytes: dict = defaultdict(int)

        self.clf = None
        self.explainer = None
        if DETECT:
            import joblib
            self.clf = joblib.load(MODEL_PATH)
            if hasattr(self.clf, "n_jobs"):
                self.clf.n_jobs = 1
            self.logger.info("detector loaded from %s", MODEL_PATH)
            if EXPLAIN:
                import shap
                self.explainer = shap.TreeExplainer(
                    self.clf, feature_perturbation="tree_path_dependent")
                self.logger.info("TreeSHAP explainer ready")

        signal.signal(signal.SIGTERM, self._dump_and_exit)
        signal.signal(signal.SIGINT, self._dump_and_exit)

    # ---------------------------------------------------------------- utils
    def _sample(self):
        try:
            self.m["cpu_samples"].append(self.proc.cpu_percent(None))
            self.m["rss_samples"].append(self.proc.memory_info().rss / 2 ** 30)
        except psutil.Error:
            pass

    def _dump_and_exit(self, *_):
        self._write()
        os._exit(0)

    def _write(self):
        d = np.array(self.m["decide_ns"], float)
        i = np.array(self.m["install_ns"], float)
        cpu = np.array(self.m["cpu_samples"], float)
        rss = np.array(self.m["rss_samples"], float)

        def pct(a, q):
            return float(np.percentile(a, q)) if a.size else None

        out = {k: v for k, v in self.m.items()
               if k not in ("decide_ns", "install_ns", "cpu_samples", "rss_samples")}
        out["wall_seconds"] = round(time.time() - self.t_start, 2)
        out["decide_ms"] = {"p50": pct(d, 50) and pct(d, 50) / 1e6,
                            "p95": pct(d, 95) and pct(d, 95) / 1e6,
                            "p99": pct(d, 99) and pct(d, 99) / 1e6,
                            "mean": float(d.mean() / 1e6) if d.size else None,
                            "n": int(d.size)}
        out["install_ms"] = {"p50": pct(i, 50) and pct(i, 50) / 1e6,
                             "p95": pct(i, 95) and pct(i, 95) / 1e6,
                             "mean": float(i.mean() / 1e6) if i.size else None}
        out["controller_cpu_percent"] = {"mean": float(cpu.mean()) if cpu.size else None,
                                         "p95": pct(cpu, 95),
                                         "max": float(cpu.max()) if cpu.size else None}
        out["controller_rss_gb"] = {"mean": float(rss.mean()) if rss.size else None,
                                    "max": float(rss.max()) if rss.size else None}
        out["packet_in_per_s"] = (round(self.m["packet_in"] / out["wall_seconds"], 1)
                                  if out["wall_seconds"] else None)
        with open(OUT_PATH, "w") as fh:
            json.dump(out, fh, indent=2)
        self.logger.info("metrics written to %s", OUT_PATH)

    # ------------------------------------------------------------- handlers
    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features(self, ev):
        dp = ev.msg.datapath
        ofp, psr = dp.ofproto, dp.ofproto_parser
        self._add_flow(dp, 0, psr.OFPMatch(),
                       [psr.OFPActionOutput(ofp.OFPP_CONTROLLER,
                                            ofp.OFPCML_NO_BUFFER)])
        self.logger.info("switch %s connected", dp.id)

    def _add_flow(self, dp, priority, match, actions, idle=0, hard=0, drop=False):
        ofp, psr = dp.ofproto, dp.ofproto_parser
        inst = [] if drop else [psr.OFPInstructionActions(
            ofp.OFPIT_APPLY_ACTIONS, actions)]
        mod = psr.OFPFlowMod(datapath=dp, priority=priority, match=match,
                             instructions=inst, idle_timeout=idle,
                             hard_timeout=hard)
        dp.send_msg(mod)
        self.m["flow_mod"] += 1

    def _features(self, pkt, ip, l4, key) -> np.ndarray:
        """Build the 88-dimensional vector from what a packet_in exposes.

        A packet_in carries one packet, not a completed flow record, so the
        statistics a flow exporter would provide are approximated from the
        controller's own running counters for the flow key. Fields that cannot
        be observed at this point are left at zero. This is stated in the paper:
        the Mininet measurement characterises control-plane cost faithfully and
        detection quality only approximately, which is why detection accuracy is
        reported from the offline evaluation rather than from here.
        """
        x = np.zeros((1, N_FEATURES), dtype=np.float32)
        now = time.time()
        first = self.flow_first_seen.setdefault(key, now)
        self.flow_pkts[key] += 1
        length = len(pkt.data) if hasattr(pkt, "data") else 0
        self.flow_bytes[key] += length
        dur_us = max((now - first) * 1e6, 1.0)

        x[0, 0] = getattr(l4, "dst_port", 0) or 0          # Destination_Port
        x[0, 1] = dur_us                                    # Flow_Duration
        x[0, 2] = self.flow_pkts[key]                       # Total_Fwd_Packets
        x[0, 4] = self.flow_bytes[key]                      # Total_Length_of_Fwd
        x[0, 8] = length                                    # Fwd_Packet_Length_Mean
        x[0, 14] = self.flow_bytes[key] / (dur_us / 1e6)    # Flow_Bytes_s
        x[0, 15] = self.flow_pkts[key] / (dur_us / 1e6)     # Flow_Packets_s
        x[0, 40] = length                                   # Packet_Length_Mean
        if isinstance(l4, tcp.tcp):
            x[0, 44] = 1 if (l4.bits & tcp.TCP_SYN) else 0  # SYN_Flag_Count
            x[0, 47] = 1 if (l4.bits & tcp.TCP_ACK) else 0  # ACK_Flag_Count
        x[0, 78] = ip.proto if ip else 0                    # Protocol
        x[0, 79] = getattr(l4, "src_port", 0) or 0          # Source_Port
        return x

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in(self, ev):
        msg = ev.msg
        dp = msg.datapath
        ofp, psr = dp.ofproto, dp.ofproto_parser
        in_port = msg.match["in_port"]
        self.m["packet_in"] += 1
        self.m["bytes_in"] += len(msg.data)
        if self.m["packet_in"] % 50 == 0:
            self._sample()

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None or eth.ethertype == ether_types.ETH_TYPE_LLDP:
            return

        self.mac_to_port.setdefault(dp.id, {})[eth.src] = in_port
        out_port = self.mac_to_port[dp.id].get(eth.dst, ofp.OFPP_FLOOD)
        actions = [psr.OFPActionOutput(out_port)]

        ip = pkt.get_protocol(ipv4.ipv4)
        l4 = pkt.get_protocol(tcp.tcp) or pkt.get_protocol(udp.udp)
        alert = False

        # Detection is applied to TCP only. The detector is trained on TCP flow
        # records and the threat model is TCP volumetric flooding, so scoring an
        # ICMP or ARP packet means evaluating the model on a feature vector whose
        # TCP fields are all zero, which is outside its training distribution and
        # produces a meaningless score. An earlier version of this application
        # scored every IP packet and consequently dropped benign pings.
        scoreable = self.clf is not None and ip is not None and isinstance(l4, tcp.tcp)
        if ip is not None and not scoreable:
            self.m["skipped_non_tcp"] = self.m.get("skipped_non_tcp", 0) + 1

        if scoreable:
            key = (ip.src, ip.dst, getattr(l4, "dst_port", 0), ip.proto)
            t0 = time.perf_counter_ns()
            x = self._features(pkt, ip, l4, key)
            score = float(self.clf.predict_proba(x)[0, 1])
            alert = score >= TAU
            self.m["decide_ns"].append(time.perf_counter_ns() - t0)
            if alert:
                self.m["alerts"] += 1
                if self.explainer is not None:
                    self.explainer.shap_values(x, check_additivity=False)
                    self.m["explanations"] += 1

        t1 = time.perf_counter_ns()
        if alert and ip is not None:
            # Match on the transport protocol as well as the source, so that a
            # mitigation rule blocks the flooding protocol from that source and
            # not every packet the host sends.
            match = psr.OFPMatch(eth_type=ether_types.ETH_TYPE_IP,
                                 ipv4_src=ip.src, ip_proto=ip.proto)
            self._add_flow(dp, 100, match, [], idle=IDLE_TIMEOUT,
                           hard=HARD_TIMEOUT, drop=True)
            self.m["drops_installed"] += 1
        else:
            if out_port != ofp.OFPP_FLOOD:
                match = psr.OFPMatch(in_port=in_port, eth_dst=eth.dst,
                                     eth_src=eth.src)
                self._add_flow(dp, 10, match, actions, idle=IDLE_TIMEOUT,
                               hard=HARD_TIMEOUT)
            data = msg.data if msg.buffer_id == ofp.OFP_NO_BUFFER else None
            dp.send_msg(psr.OFPPacketOut(datapath=dp, buffer_id=msg.buffer_id,
                                         in_port=in_port, actions=actions,
                                         data=data))
            self.m["packet_out"] += 1
        self.m["install_ns"].append(time.perf_counter_ns() - t1)
