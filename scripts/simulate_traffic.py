"""
simulate_traffic.py — Live Traffic Simulator for XAI-SDN.

Generates and posts a continuous stream of realistic DDoS alerts to the 
FastAPI API server (http://localhost:8000), which automatically populates 
the Streamlit Dashboard with real-time graphs, alerts, and SHAP charts.
"""

import time
import random
import requests
from datetime import datetime

API_URL = "http://localhost:8000/api/v1/alerts"

ATTACK_TYPES = ["DDoS-UDP", "DDoS-TCP", "DDoS-ICMP", "DDoS-SlowLoris", "DDoS-HTTP"]

MOCK_IPS = [
    "185.220.101.4", "192.168.10.15", "10.0.0.105", 
    "45.227.254.12", "91.241.19.88", "103.88.22.44",
    "203.0.113.88", "198.51.100.4", "192.0.2.145"
]

MOCK_PORTS = [80, 443, 53, 8080, 22]

def generate_mock_alert():
    attack_type = random.choice(ATTACK_TYPES)
    src_ip = random.choice(MOCK_IPS)
    dst_ip = "10.0.0.1"
    dst_port = random.choice(MOCK_PORTS)
    protocol = 17 if attack_type == "DDoS-UDP" else (1 if attack_type == "DDoS-ICMP" else 6)
    
    confidence = round(random.uniform(0.85, 0.99), 4)
    flow_id = f"sim-{random.randint(100000, 999999)}"
    
    # 1. Generate realistic entropy feature vector values (low entropy = DDoS indicators)
    # Different attack types concentrate on different features!
    feature_vector = {
        "H_src_ip": round(random.uniform(0.05, 0.45) if attack_type in ["DDoS-UDP", "DDoS-TCP"] else random.uniform(1.2, 2.5), 3),
        "H_dst_ip": round(random.uniform(0.01, 0.15), 3),
        "H_dst_port": round(random.uniform(0.05, 0.3) if attack_type in ["DDoS-UDP", "DDoS-HTTP"] else random.uniform(1.5, 3.0), 3),
        "H_proto": 0.0,
        "H_pkt_len": round(random.uniform(0.1, 0.6) if attack_type == "DDoS-ICMP" else random.uniform(1.8, 3.5), 3),
        "H_iat": round(random.uniform(0.2, 0.8), 3),
        "H_tcp_flags": round(random.uniform(0.1, 0.5) if attack_type == "DDoS-TCP" else random.uniform(1.5, 2.8), 3),
        "H_src_port": 0.0
    }
    
    # Add some standard flow features
    feature_vector.update({
        "Flow Duration": round(random.uniform(10.0, 500.0), 2),
        "Total Fwd Packets": random.randint(50, 1000),
        "Total Length of Fwd Packets": random.randint(2000, 150000)
    })

    # 2. Generate SHAP top features (explainability scores)
    # Positive shap values drive prediction towards DDoS, negative values push towards Benign
    shap_top_features = []
    
    # Select important features based on attack type to make the charts look authentic!
    if attack_type == "DDoS-UDP":
        shap_top_features = [
            {"feature": "H_dst_port", "shap_value": 0.38, "abs_shap": 0.38},
            {"feature": "H_src_ip", "shap_value": 0.29, "abs_shap": 0.29},
            {"feature": "Total Fwd Packets", "shap_value": 0.18, "abs_shap": 0.18},
            {"feature": "H_pkt_len", "shap_value": -0.05, "abs_shap": 0.05},
        ]
    elif attack_type == "DDoS-TCP":
        shap_top_features = [
            {"feature": "H_tcp_flags", "shap_value": 0.42, "abs_shap": 0.42},
            {"feature": "H_src_ip", "shap_value": 0.25, "abs_shap": 0.25},
            {"feature": "Flow Duration", "shap_value": 0.12, "abs_shap": 0.12},
            {"feature": "H_dst_port", "shap_value": -0.04, "abs_shap": 0.04},
        ]
    elif attack_type == "DDoS-SlowLoris":
        shap_top_features = [
            {"feature": "Flow Duration", "shap_value": 0.45, "abs_shap": 0.45},
            {"feature": "H_iat", "shap_value": 0.31, "abs_shap": 0.31},
            {"feature": "H_tcp_flags", "shap_value": 0.15, "abs_shap": 0.15},
            {"feature": "Total Length of Fwd Packets", "shap_value": -0.08, "abs_shap": 0.08},
        ]
    else:  # Generic/HTTP/ICMP
        shap_top_features = [
            {"feature": "H_src_ip", "shap_value": 0.35, "abs_shap": 0.35},
            {"feature": "H_pkt_len", "shap_value": 0.22, "abs_shap": 0.22},
            {"feature": "Total Fwd Packets", "shap_value": 0.15, "abs_shap": 0.15},
            {"feature": "Flow Duration", "shap_value": -0.06, "abs_shap": 0.06},
        ]
        
    # Append the full dictionary
    payload = {
        "flow_id": flow_id,
        "src_ip": src_ip,
        "dst_ip": dst_ip,
        "src_port": random.randint(1024, 65535),
        "dst_port": dst_port,
        "protocol": protocol,
        "label": attack_type,
        "confidence": confidence,
        "switch_id": "00:00:00:00:00:00:00:01",
        "flow_duration_ms": round(random.uniform(50, 2000), 2),
        "packet_count": random.randint(100, 5000),
        "byte_count": random.randint(4000, 1000000),
        "shap_top_features": shap_top_features,
        "feature_vector": feature_vector,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    return payload

def main():
    print("=" * 60)
    print("      XAI-SDN Real-time Traffic Simulator Starting...      ")
    print("      Press CTRL+C to stop simulation at any time.       ")
    print("=" * 60)
    
    # Wait to make sure API is up
    try:
        requests.get("http://localhost:8000/health", timeout=3)
    except requests.RequestException:
        print("[WARNING] Could not connect to API server at http://localhost:8000.")
        print("Please ensure your FastAPI backend is running in Window 2!")
        print("Waiting 5 seconds before trying anyway...\n")
        time.sleep(5)
        
    count = 0
    while True:
        try:
            alert = generate_mock_alert()
            r = requests.post(API_URL, json=alert, timeout=5)
            
            if r.status_code in [200, 201]:
                count += 1
                print(f"[{count:03d}] [SUCCESS] Alert Ingested: {alert['label']} | Src: {alert['src_ip']} -> Dst: {alert['dst_ip']}:{alert['dst_port']} (Conf: {alert['confidence']:.2%})")
            else:
                print(f"[ERROR] Failed to ingest alert. Status Code: {r.status_code} | Details: {r.text}")
                
        except Exception as e:
            print(f"[CONNECTION ERROR] Failed to send alert: {e}")
            print("Make sure your API server (Window 2) is running and active.")
            
        # Send an alert every 2 to 5 seconds randomly
        time.sleep(random.uniform(2.0, 4.0))

if __name__ == "__main__":
    main()
