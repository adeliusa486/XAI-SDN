"""
xai_sdn_app.py — Ryu SDN Controller Application for XAI-SDN.

This Ryu application acts as the real-time DDoS detection engine inside
the SDN controller. It:

  1. Installs OpenFlow 1.3 rules and polls per-flow stats every 500ms.
  2. Extracts flow features using the online feature pipeline.
  3. Classifies each flow with the trained Random Forest.
  4. Invokes TreeSHAP for any flow predicted as DDoS with p ≥ τ.
  5. Dispatches structured alerts via HTTP POST to the XAI-SDN API.
  6. Optionally installs drop rules for confirmed DDoS flows.

IMPORTANT: Ryu requires Python ≤ 3.8. Run this module in a separate
virtualenv:
    python3.8 -m venv venv-ryu
    source venv-ryu/bin/activate
    pip install ryu eventlet requests joblib scikit-learn shap numpy
    ryu-manager sdn/controller/xai_sdn_app.py

Mininet topology for testing:
    sudo python sdn/topology/mininet_topo.py
"""

from __future__ import annotations

import json
import os
import time
import threading
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import requests
from loguru import logger

# ─── Ryu imports (graceful degradation for non-Ryu environments) ──────────────
try:
    from ryu.base import app_manager
    from ryu.controller import ofp_event
    from ryu.controller.handler import (
        CONFIG_DISPATCHER,
        MAIN_DISPATCHER,
        set_ev_cls,
    )
    from ryu.lib import hub
    from ryu.lib.packet import packet, ethernet, ipv4, tcp, udp, icmp
    from ryu.ofproto import ofproto_v1_3
    RYU_AVAILABLE = True
except ImportError:
    RYU_AVAILABLE = False
    logger.warning(
        "Ryu not available. Controller module loaded in stub mode. "
        "Install Ryu in Python 3.8 environment."
    )
    # Stub base class so the module can be imported and inspected
    class app_manager:
        class RyuApp:
            pass
    CONFIG_DISPATCHER = MAIN_DISPATCHER = None
    ofproto_v1_3 = None

    def set_ev_cls(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# ─── Config ───────────────────────────────────────────────────────────────────

ARTIFACTS_DIR = Path(os.getenv("MODEL_ARTIFACTS_DIR", "model/artifacts"))
API_URL = os.getenv("API_URL", "http://localhost:8000")
DETECTION_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", "0.70"))
POLL_INTERVAL = float(os.getenv("SDN_POLL_INTERVAL_MS", "500")) / 1000.0
WINDOW_SIZE = int(os.getenv("SDN_FLOW_WINDOW_SIZE", "1000"))
INSTALL_DROP_RULES = os.getenv("SDN_INSTALL_DROP_RULES", "0") == "1"


# ─── Model Loader ─────────────────────────────────────────────────────────────

class ModelBundle:
    """Holds the loaded ML artifacts for controller-side inference."""

    def __init__(self, artifacts_dir: Path) -> None:
        self.clf = None
        self.scaler = None
        self.label_encoder = None
        self.shap_explainer = None
        self.feature_names: List[str] = []
        self._load(artifacts_dir)

    def _load(self, artifacts_dir: Path) -> None:
        if not (artifacts_dir / "rf_model.pkl").exists():
            logger.warning(
                f"Model not found at {artifacts_dir}. "
                "Inference will return BENIGN for all flows."
            )
            return

        logger.info(f"Loading model artifacts from {artifacts_dir}...")
        self.clf = joblib.load(artifacts_dir / "rf_model.pkl")
        self.scaler = joblib.load(artifacts_dir / "scaler.pkl")
        self.label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")

        fn_path = artifacts_dir / "feature_names.json"
        if fn_path.exists():
            with open(fn_path) as f:
                self.feature_names = json.load(f)

        # Load SHAP explainer
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from explainability.shap_explainer import SHAPExplainer
            self.shap_explainer = SHAPExplainer(
                model=self.clf,
                feature_names=self.feature_names,
                confidence_threshold=DETECTION_THRESHOLD,
            )
            logger.info("TreeSHAP explainer ready.")
        except Exception as e:
            logger.warning(f"SHAP init failed: {e}")

        logger.info(
            f"Model loaded: {type(self.clf).__name__}, "
            f"{len(self.feature_names)} features, "
            f"classes={list(self.label_encoder.classes_)}"
        )

    @property
    def is_loaded(self) -> bool:
        return self.clf is not None

    def predict(self, x: np.ndarray) -> Tuple[str, float, int]:
        """Predict label, confidence, class index for a feature vector."""
        if not self.is_loaded:
            return "Benign", 1.0, 0
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        class_idx = int(self.clf.predict(x_scaled)[0])
        confidence = float(self.clf.predict_proba(x_scaled)[0].max())
        label = self.label_encoder.classes_[class_idx]
        return label, confidence, class_idx, x_scaled[0]


# ─── Ryu Application ──────────────────────────────────────────────────────────


class XAISDNController(app_manager.RyuApp):
    """Ryu application implementing real-time DDoS detection with SHAP attribution.

    Architecture:
        EventOFPSwitchFeatures → install table-miss flow entry
        EventOFPPacketIn       → install L2 learning rules
        Background thread      → poll FlowStats every POLL_INTERVAL seconds
        Per-flow stats         → feature extraction → RF inference → alert dispatch
    """

    OFP_VERSIONS = [ofproto_v1_3.OFP_VERSION] if ofproto_v1_3 else []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load ML model
        self.model = ModelBundle(ARTIFACTS_DIR)

        # Online feature pipeline
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from features.cicflowmeter import (
            CIC_FEATURE_NAMES,
            extract_features_from_openflow,
            flow_record_from_openflow,
        )
        from features.entropy import EntropyFeatureExtractor, ENTROPY_FEATURE_NAMES
        self.CIC_FEATURE_NAMES = CIC_FEATURE_NAMES
        self.ENTROPY_FEATURE_NAMES = ENTROPY_FEATURE_NAMES
        self.extract_cic = extract_features_from_openflow
        self.flow_record_fn = flow_record_from_openflow
        self.entropy_extractor = EntropyFeatureExtractor(window_size=WINDOW_SIZE)

        # State
        self.mac_to_port: Dict[int, Dict[str, int]] = defaultdict(dict)
        self.datapaths: Dict[int, Any] = {}
        self._alert_count = 0
        self._flow_count = 0
        self._start_time = time.time()

        # Start stats polling thread
        if RYU_AVAILABLE:
            self.monitor_thread = hub.spawn(self._monitor_loop)
            logger.info(f"XAI-SDN Controller started (poll_interval={POLL_INTERVAL}s)")

    # ── OpenFlow event handlers ────────────────────────────────────────────

    @set_ev_cls(ofp_event.EventOFPSwitchFeatures, CONFIG_DISPATCHER)
    def switch_features_handler(self, ev):
        """Install table-miss flow entry on switch connection."""
        datapath = ev.msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser

        # Table-miss: send all unmatched packets to controller
        match = parser.OFPMatch()
        actions = [parser.OFPActionOutput(ofproto.OFPP_CONTROLLER, ofproto.OFPCML_NO_BUFFER)]
        self._add_flow(datapath, 0, match, actions)

        self.datapaths[datapath.id] = datapath
        logger.info(f"Switch connected: DPID={datapath.id:#018x}")

    @set_ev_cls(ofp_event.EventOFPPacketIn, MAIN_DISPATCHER)
    def packet_in_handler(self, ev):
        """L2 learning switch with ARP/IP forwarding."""
        msg = ev.msg
        datapath = msg.datapath
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        in_port = msg.match["in_port"]

        pkt = packet.Packet(msg.data)
        eth = pkt.get_protocol(ethernet.ethernet)
        if eth is None:
            return

        dst = eth.dst
        src = eth.src
        dpid = datapath.id

        self.mac_to_port[dpid][src] = in_port

        out_port = self.mac_to_port[dpid].get(dst, ofproto.OFPP_FLOOD)
        actions = [parser.OFPActionOutput(out_port)]

        if out_port != ofproto.OFPP_FLOOD:
            match = parser.OFPMatch(in_port=in_port, eth_dst=dst, eth_src=src)
            self._add_flow(datapath, 1, match, actions)

        data = msg.data if msg.buffer_id == ofproto.OFP_NO_BUFFER else None
        out = parser.OFPPacketOut(
            datapath=datapath,
            buffer_id=msg.buffer_id,
            in_port=in_port,
            actions=actions,
            data=data,
        )
        datapath.send_msg(out)

    @set_ev_cls(ofp_event.EventOFPFlowStatsReply, MAIN_DISPATCHER)
    def flow_stats_reply_handler(self, ev):
        """Process per-flow statistics reply from switch."""
        body = ev.msg.body
        dpid = ev.msg.datapath.id

        for stat in body:
            if stat.priority == 0:  # Skip table-miss entry
                continue

            self._flow_count += 1
            stat_dict = self._stat_to_dict(stat, dpid)

            # Extract features
            cic_features = self.extract_cic(stat_dict)
            flow_record = self.flow_record_fn(stat_dict)

            # Build 88-dim feature vector
            cic_array = np.array(
                [cic_features.get(n, 0.0) for n in self.CIC_FEATURE_NAMES],
                dtype=np.float64,
            )
            entropy_dict = self.entropy_extractor.update_and_compute(flow_record)
            entropy_array = np.array(
                [entropy_dict[n] for n in self.ENTROPY_FEATURE_NAMES],
                dtype=np.float64,
            )
            x = np.nan_to_num(
                np.concatenate([cic_array, entropy_array]),
                nan=0.0, posinf=0.0, neginf=0.0,
            )

            # Inference
            label, confidence, class_idx, x_scaled = self.model.predict(x)

            if label != "Benign" and confidence >= DETECTION_THRESHOLD:
                self._handle_detection(
                    stat_dict, label, confidence, class_idx, x_scaled, cic_features
                )

    # ── Detection handler ──────────────────────────────────────────────────

    def _handle_detection(
        self,
        stat: Dict,
        label: str,
        confidence: float,
        class_idx: int,
        x_scaled: np.ndarray,
        cic_features: Dict,
    ) -> None:
        """Handle a confirmed DDoS detection: compute SHAP + dispatch alert."""
        match = stat.get("match", {})

        # Compute SHAP attribution
        shap_data = {}
        if self.model.shap_explainer and self.model.shap_explainer.is_ready:
            try:
                shap_data = self.model.shap_explainer.format_alert_attribution(
                    x_scaled, class_idx, top_k=10
                )
            except Exception as e:
                logger.warning(f"SHAP failed for flow: {e}")

        alert_payload = {
            "flow_id": str(uuid.uuid4()),
            "src_ip": match.get("ipv4_src", "0.0.0.0"),
            "dst_ip": match.get("ipv4_dst", "0.0.0.0"),
            "src_port": int(match.get("tp_src", 0)),
            "dst_port": int(match.get("tp_dst", 0)),
            "protocol": int(match.get("ip_proto", 0)),
            "label": label,
            "confidence": confidence,
            "switch_id": f"{stat.get('dpid', 0):#018x}",
            "flow_duration_ms": float(stat.get("duration_sec", 0)) * 1000,
            "packet_count": stat.get("packet_count", 0),
            "byte_count": stat.get("byte_count", 0),
            "shap_top_features": shap_data.get("top_features"),
            "shap_full_attribution": shap_data.get("full_attribution"),
        }

        # Dispatch alert asynchronously (don't block stats handler)
        threading.Thread(
            target=self._post_alert,
            args=(alert_payload,),
            daemon=True,
        ).start()

        # Optionally install drop rule
        if INSTALL_DROP_RULES and label not in ("Benign", "DDoS-SlowLoris"):
            # Slow loris needs per-connection handling; skip blanket drop
            logger.warning(
                f"Would install drop rule for {match.get('ipv4_src')} → "
                f"{match.get('ipv4_dst')}:{match.get('tp_dst')} "
                f"[NOT IMPLEMENTED — requires datapath reference]"
            )

        self._alert_count += 1
        logger.info(
            f"ALERT #{self._alert_count} | {label} | conf={confidence:.3f} | "
            f"src={match.get('ipv4_src')} → dst={match.get('ipv4_dst')}:"
            f"{match.get('tp_dst')}"
        )

    # ── Background monitor ─────────────────────────────────────────────────

    def _monitor_loop(self) -> None:
        """Periodically request FlowStats from all connected switches."""
        while True:
            hub.sleep(POLL_INTERVAL)
            for dp in list(self.datapaths.values()):
                self._request_flow_stats(dp)

    def _request_flow_stats(self, datapath) -> None:
        """Send OFPFlowStatsRequest to a datapath."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        req = parser.OFPFlowStatsRequest(datapath)
        datapath.send_msg(req)

    # ── Utilities ──────────────────────────────────────────────────────────

    def _add_flow(self, datapath, priority, match, actions, idle_timeout=0, hard_timeout=0):
        """Install a flow rule on a switch."""
        ofproto = datapath.ofproto
        parser = datapath.ofproto_parser
        inst = [parser.OFPInstructionActions(ofproto.OFPIT_APPLY_ACTIONS, actions)]
        mod = parser.OFPFlowMod(
            datapath=datapath,
            priority=priority,
            match=match,
            instructions=inst,
            idle_timeout=idle_timeout,
            hard_timeout=hard_timeout,
        )
        datapath.send_msg(mod)

    @staticmethod
    def _stat_to_dict(stat, dpid: int) -> Dict[str, Any]:
        """Convert OFPFlowStats object to serializable dict."""
        match_dict = {}
        for key in stat.match.keys():
            try:
                match_dict[key] = stat.match[key]
            except Exception:
                pass
        return {
            "dpid": dpid,
            "packet_count": stat.packet_count,
            "byte_count": stat.byte_count,
            "duration_sec": stat.duration_sec,
            "duration_nsec": stat.duration_nsec,
            "priority": stat.priority,
            "match": match_dict,
        }

    def _post_alert(self, payload: Dict) -> None:
        """HTTP POST alert to the XAI-SDN API (runs in background thread)."""
        try:
            resp = requests.post(
                f"{API_URL}/api/v1/alerts",
                json=payload,
                timeout=2.0,
            )
            if resp.status_code not in (200, 201):
                logger.warning(f"Alert POST failed: {resp.status_code} {resp.text[:100]}")
        except requests.exceptions.ConnectionError:
            logger.debug("API not reachable (controller running without API).")
        except Exception as e:
            logger.error(f"Alert dispatch error: {e}")
