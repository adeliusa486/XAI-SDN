"""
app.py — Streamlit Security Operations Dashboard for XAI-SDN.

Provides a real-time SOC-facing view of DDoS alerts with:
  - Live alert feed with per-flow SHAP attribution waterfall charts
  - Attack distribution donut chart
  - Timeline of alerts per minute
  - Top attacking IPs and targeted ports
  - Model confidence distribution
  - Per-attack-type entropy feature heatmap

Usage:
    streamlit run dashboard/app.py
    DASHBOARD_API_URL=http://api:8000 streamlit run dashboard/app.py
"""

from __future__ import annotations

import os
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Configuration ────────────────────────────────────────────────────────────

API_URL = os.getenv("DASHBOARD_API_URL", "http://localhost:8000")
REFRESH_INTERVAL = int(os.getenv("DASHBOARD_REFRESH_INTERVAL", "5"))
MAX_ALERTS = int(os.getenv("DASHBOARD_MAX_ALERTS", "500"))

LABEL_COLORS = {
    "Benign":        "#2ecc71",
    "DDoS-UDP":      "#e74c3c",
    "DDoS-TCP":      "#e67e22",
    "DDoS-ICMP":     "#9b59b6",
    "DDoS-SlowLoris":"#1abc9c",
    "DDoS-HTTP":     "#3498db",
    "Unknown":       "#95a5a6",
}

ENTROPY_FEATURES = [
    "H_src_ip", "H_dst_ip", "H_dst_port", "H_proto",
    "H_pkt_len", "H_iat", "H_tcp_flags", "H_ttl",
]

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="XAI-SDN Security Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
.metric-card {
    background: #1e2130;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
    border: 1px solid #2d3250;
}
.alert-critical { border-left: 4px solid #e74c3c; padding-left: 8px; }
.alert-warning  { border-left: 4px solid #e67e22; padding-left: 8px; }
.alert-info     { border-left: 4px solid #3498db; padding-left: 8px; }
.stAlert        { border-radius: 6px; }
</style>
""", unsafe_allow_html=True)


# ─── API Helpers ──────────────────────────────────────────────────────────────

@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_health() -> Dict:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.json()
    except Exception:
        return {"status": "unreachable", "model_loaded": False, "shap_ready": False,
                "alert_count": 0, "uptime_seconds": 0, "database_connected": False}


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_stats() -> Dict:
    try:
        r = requests.get(f"{API_URL}/api/v1/alerts/stats", timeout=3)
        return r.json()
    except Exception:
        return {}


@st.cache_data(ttl=REFRESH_INTERVAL)
def fetch_alerts(page_size: int = MAX_ALERTS, label: Optional[str] = None,
                 min_conf: float = 0.0) -> List[Dict]:
    params = {"page_size": page_size, "min_confidence": min_conf}
    if label and label != "All":
        params["label"] = label
    try:
        r = requests.get(f"{API_URL}/api/v1/alerts", params=params, timeout=5)
        return r.json().get("alerts", [])
    except Exception:
        return []


@st.cache_data(ttl=60)
def fetch_model_info() -> Dict:
    try:
        r = requests.get(f"{API_URL}/api/v1/model/info", timeout=3)
        return r.json()
    except Exception:
        return {}


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/XAI--SDN-v0.1.0-blue", use_column_width=False)
    st.title("🛡️ XAI-SDN")
    st.caption("Explainable DDoS Detection for SDN")
    st.divider()

    # Health indicator
    health = fetch_health()
    status = health.get("status", "unknown")
    color = "🟢" if status == "healthy" else ("🟡" if status == "degraded" else "🔴")
    st.metric("API Status", f"{color} {status.capitalize()}")
    st.metric("Model Loaded", "✅" if health.get("model_loaded") else "❌")
    st.metric("SHAP Ready", "✅" if health.get("shap_ready") else "❌")
    st.metric("Uptime", f"{health.get('uptime_seconds', 0):.0f}s")
    st.divider()

    # Filters
    st.subheader("Filters")
    label_filter = st.selectbox(
        "Attack Type",
        ["All", "DDoS-UDP", "DDoS-TCP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-HTTP"],
    )
    min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.05)
    auto_refresh = st.checkbox("Auto-refresh", value=True)
    refresh_secs = st.select_slider(
        "Refresh interval (s)", options=[3, 5, 10, 30, 60], value=REFRESH_INTERVAL
    )

    st.divider()
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

    # Model info
    model_info = fetch_model_info()
    if model_info:
        st.subheader("Model")
        st.caption(f"Type: {model_info.get('model_type','?')}")
        st.caption(f"Trees: {model_info.get('n_estimators','?')}")
        st.caption(f"Features: {model_info.get('n_features','?')}")
        if model_info.get("training_macro_f1"):
            st.caption(f"Train F1: {model_info['training_macro_f1']:.4f}")


# ─── Main Content ─────────────────────────────────────────────────────────────

st.title("🛡️ XAI-SDN Security Operations Dashboard")
st.caption(f"Real-time DDoS detection with Explainable AI · Last update: {datetime.utcnow().strftime('%H:%M:%S UTC')}")

# ─── KPI Row ─────────────────────────────────────────────────────────────────

stats = fetch_stats()
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Alerts", f"{stats.get('total_alerts', 0):,}")
with col2:
    st.metric("Last Hour", f"{stats.get('alerts_last_hour', 0):,}")
with col3:
    st.metric("Last 24h", f"{stats.get('alerts_last_24h', 0):,}")
with col4:
    avg_conf = stats.get("avg_confidence", 0)
    st.metric("Avg Confidence", f"{avg_conf:.1%}")
with col5:
    by_label = stats.get("by_label", {})
    n_types = len([k for k, v in by_label.items() if k != "Benign" and v > 0])
    st.metric("Active Attack Types", n_types)

st.divider()

# ─── Charts Row ───────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns([1, 1])

with chart_col1:
    st.subheader("Attack Distribution")
    by_label = stats.get("by_label", {})
    if by_label:
        df_dist = pd.DataFrame(
            [(k, v) for k, v in by_label.items() if v > 0],
            columns=["Label", "Count"],
        )
        fig_pie = px.pie(
            df_dist, names="Label", values="Count",
            color="Label",
            color_discrete_map=LABEL_COLORS,
            hole=0.45,
        )
        fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.info("No alerts yet. Waiting for traffic...")

with chart_col2:
    st.subheader("Top Source IPs")
    top_ips = stats.get("top_src_ips", [])
    if top_ips:
        df_ips = pd.DataFrame(top_ips).head(10)
        fig_ips = px.bar(
            df_ips, x="count", y="ip", orientation="h",
            color_discrete_sequence=["#e74c3c"],
        )
        fig_ips.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            yaxis_title="",
            xaxis_title="Alert Count",
        )
        st.plotly_chart(fig_ips, use_container_width=True)
    else:
        st.info("No data yet.")

# ─── Alert Timeline ───────────────────────────────────────────────────────────

st.subheader("Alert Timeline")
alerts = fetch_alerts(page_size=MAX_ALERTS, label=label_filter if label_filter != "All" else None,
                      min_conf=min_confidence)

if alerts:
    df_alerts = pd.DataFrame(alerts)
    df_alerts["timestamp"] = pd.to_datetime(df_alerts["timestamp"], errors="coerce")
    df_alerts = df_alerts.sort_values("timestamp", ascending=False)

    # Resample to 1-minute buckets
    df_time = df_alerts.copy()
    df_time = df_time.set_index("timestamp").resample("1min")["label"].value_counts().reset_index()
    df_time.columns = ["timestamp", "label", "count"]

    if not df_time.empty:
        fig_timeline = px.area(
            df_time, x="timestamp", y="count", color="label",
            color_discrete_map=LABEL_COLORS,
            labels={"count": "Alerts/min", "timestamp": "Time"},
        )
        fig_timeline.update_layout(
            margin=dict(t=10, b=10), height=250,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
else:
    st.info("No alerts to display.")

# ─── Alert Table + SHAP Details ───────────────────────────────────────────────

st.subheader("Recent Alerts")
tab_table, tab_shap, tab_entropy = st.tabs(["📋 Alert Feed", "🔍 SHAP Attribution", "📊 Entropy Analysis"])

with tab_table:
    if alerts:
        df_display = pd.DataFrame(alerts)[
            ["id", "timestamp", "label", "confidence", "src_ip", "dst_ip", "dst_port", "protocol", "switch_id"]
        ].rename(columns={
            "id": "ID", "timestamp": "Time", "label": "Label",
            "confidence": "Conf.", "src_ip": "Src IP", "dst_ip": "Dst IP",
            "dst_port": "Port", "protocol": "Proto", "switch_id": "Switch",
        })
        df_display["Conf."] = df_display["Conf."].apply(lambda x: f"{x:.2%}")
        df_display["Time"] = pd.to_datetime(df_display["Time"]).dt.strftime("%H:%M:%S")
        st.dataframe(
            df_display.head(100),
            use_container_width=True,
            hide_index=True,
            column_config={
                "Label": st.column_config.TextColumn("Label", width="medium"),
                "Conf.": st.column_config.TextColumn("Confidence", width="small"),
            },
        )
    else:
        st.info("No alerts match the current filter.")

with tab_shap:
    st.markdown(
        "**SHAP Attribution** — shows which features drove each DDoS detection. "
        "Positive values (red) push the prediction toward DDoS; "
        "negative values (blue) push it toward Benign."
    )
    if alerts:
        alert_options = {
            f"Alert #{a['id']} — {a['label']} ({a['confidence']:.0%}) from {a['src_ip']}": a
            for a in alerts[:50]
            if a.get("shap_top_features")
        }
        if alert_options:
            selected_key = st.selectbox("Select alert for SHAP view:", list(alert_options.keys()))
            selected = alert_options[selected_key]
            shap_feats = selected.get("shap_top_features", [])
            if shap_feats:
                df_shap = pd.DataFrame(shap_feats).sort_values("abs_shap", ascending=True)
                fig_shap = go.Figure(go.Bar(
                    x=df_shap["shap_value"],
                    y=df_shap["feature"],
                    orientation="h",
                    marker_color=[
                        "#d73027" if v > 0 else "#4575b4"
                        for v in df_shap["shap_value"]
                    ],
                ))
                fig_shap.add_vline(x=0, line_width=1, line_color="white")
                fig_shap.update_layout(
                    title=f"SHAP Attribution — {selected['label']} (p={selected['confidence']:.2%})",
                    xaxis_title="SHAP Value",
                    yaxis_title="",
                    height=400,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_shap, use_container_width=True)

                st.caption("💡 Entropy features (H_*) highlighted — low entropy signals DDoS concentration.")
        else:
            st.info("No SHAP data available. Ensure SHAP is enabled and model is loaded.")
    else:
        st.info("No alerts to show SHAP for.")

with tab_entropy:
    st.markdown(
        "**Entropy Feature Comparison** — compares mean entropy values across "
        "detected attack types. Low entropy in H_src_ip indicates IP concentration (botnet). "
        "Low H_dst_port indicates single-port flood."
    )
    if alerts:
        df_ent = pd.DataFrame(alerts)
        df_ent = df_ent[df_ent["feature_vector"].notna()]

        if not df_ent.empty:
            # Extract entropy features from stored feature vectors
            for feat in ENTROPY_FEATURES:
                df_ent[feat] = df_ent["feature_vector"].apply(
                    lambda fv: fv.get(feat, 0.0) if isinstance(fv, dict) else 0.0
                )

            mean_entropy = df_ent.groupby("label")[ENTROPY_FEATURES].mean().reset_index()

            fig_heat = px.imshow(
                mean_entropy.set_index("label")[ENTROPY_FEATURES].T,
                color_continuous_scale="RdYlGn",
                aspect="auto",
                labels={"color": "Mean Entropy (bits)"},
                title="Mean Entropy per Feature by Attack Type",
            )
            fig_heat.update_layout(height=300)
            st.plotly_chart(fig_heat, use_container_width=True)
        else:
            st.info("No feature vector data in stored alerts (requires feature_vector field in alerts).")
    else:
        st.info("No alerts to analyze.")

# ─── Auto-refresh ─────────────────────────────────────────────────────────────

if auto_refresh:
    time.sleep(refresh_secs)
    st.cache_data.clear()
    st.rerun()
