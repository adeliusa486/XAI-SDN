# XAI-SDN Operator Guide

*For Security Operations Center (SOC) Analysts*

---

## Reading the Dashboard

The dashboard has four main sections:

### 1. KPI Banner
- **Total Alerts** — cumulative DDoS detections since system start
- **Last Hour / Last 24h** — recent attack volume
- **Avg Confidence** — mean model certainty (>0.85 = high confidence detections)
- **Active Attack Types** — how many distinct attack classes are active

### 2. Attack Distribution (Donut Chart)
Shows proportion of each attack type. In an active DDoS incident you'll see one
sector dominate. Mixed proportions suggest a multi-vector attack.

### 3. Alert Timeline
Alerts per minute, colour-coded by type. Look for:
- **Sharp spikes** → volumetric flood starting (UDP/TCP/ICMP)
- **Sustained plateau** → application-layer attack (HTTP/SlowLoris)
- **Gradual rise** → botnet spin-up

### 4. Alert Feed Tab
Live table of recent detections sorted newest-first.

Key columns:
| Column | What to look for |
|--------|-----------------|
| Label | Attack type |
| Conf. | >90% = high certainty; 70-80% = review manually |
| Src IP | Repeat attacker? Check Top Source IPs panel |
| Port | Port 53 = DNS amp; 80/443 = HTTP flood; 0 = ICMP |
| Switch | Which switch is closest to the attack entry point |

---

## Reading SHAP Attribution

The **SHAP Attribution** tab explains *why* the model flagged a specific flow.

### What the waterfall chart shows
- **Red bars** (positive SHAP) → features that pushed the prediction *toward DDoS*
- **Blue bars** (negative SHAP) → features that pushed it *toward Benign*
- **Bar length** → magnitude of the feature's influence

### Common patterns by attack type

**DDoS-UDP Flood**
```
H_src_ip    ████████████████  +0.41  ← Very few source IPs (botnet)
H_dst_port  ███████████       +0.30  ← All to single port (DNS:53)
Flow_Bytes_s██████████        +0.19  ← Abnormally high byte rate
H_proto     ████████          +0.15  ← Only UDP seen
```
*Action*: Block source /24 subnet at perimeter; rate-limit DNS.

**DDoS-TCP SYN Flood**
```
SYN_Flag_Count ████████████████ +0.45  ← Hundreds of SYNs, no ACK
H_tcp_flags    ████████████     +0.31  ← Only SYN flag observed
H_dst_port     ████████         +0.22  ← Single target port
Flow_IAT_Mean  ██████           +0.18  ← Very fast packet rate
```
*Action*: Enable SYN cookies on target server; consider DPID-level ACL.

**DDoS-SlowLoris**
```
Flow_Duration   ██████████████ +0.38  ← Very long-lived connections
Total_Fwd_Packets ████████████ +0.29  ← Very few packets per flow
H_iat           ████████       +0.21  ← Slow, regular send rate
Active_Mean     ██████         +0.15  ← High active time ratio
```
*Action*: Set server connection timeout; block source IPs; 
         consider ModSecurity `SecRuleEngine On`.

**DDoS-HTTP Flood**
```
H_src_ip     ████████████████ +0.40  ← Distributed sources (hard to block)
Flow_Bytes_s ████████████     +0.30  ← High request rate
H_dst_port   ████             +0.12  ← Port 80/443
```
*Action*: Enable rate limiting per IP; deploy CAPTCHA; consider Cloudflare.

---

## Alert Confidence Interpretation

| Confidence | Interpretation | Recommended Action |
|------------|---------------|-------------------|
| ≥ 0.95 | Very high | Auto-alert + immediate investigation |
| 0.85–0.94 | High | Alert SOC; review within 5 minutes |
| 0.70–0.84 | Moderate | Flag for review; check with peer |
| < 0.70 | Low (not alerted) | Not shown (below threshold τ) |

---

## False Positive Handling

The model has FPR ≈ 0.48% (1 false alert per ~208 legitimate flows).

**If you believe an alert is a false positive:**

1. Note the flow ID, src IP, dst IP, port from the Alert Feed.
2. Check the SHAP tab — does the attribution make sense?
   - If entropy features are driving it: verify the window context. A burst of
     legitimate traffic (CDN purge, backup job) can temporarily lower entropy.
3. Check adjacent flows from the same src IP in the last 60 seconds.
4. If confirmed false positive, add an exemption rule (future feature) or 
   widen the `min_confidence` filter to 0.80.

---

## Escalation Procedure

1. **Detect**: Alert fires on dashboard (red indicator, label ≠ Benign).
2. **Triage**: Open SHAP tab for the alert. Identify top-3 contributing features.
3. **Validate**: Confirm with network team that the flagged switch/port is under load.
4. **Contain** (if validated):
   - Coordinate with network team to apply ACL on ingress switch.
   - For volumetric attacks: upstream black-hole routing (RTBH).
   - For application attacks: WAF rule or rate limit.
5. **Document**: Record alert IDs, SHAP features, timestamps in incident ticket.
6. **Recover**: Monitor alert rate drop-off on Timeline chart.

---

## Key Entropy Thresholds (Reference)

| Feature | Normal Range | Attack Indicator |
|---------|-------------|-----------------|
| H_src_ip | 4.0 – 7.0 bits | < 2.0 bits |
| H_dst_ip | 3.5 – 6.0 bits | < 1.0 bit (single target) |
| H_dst_port | 3.0 – 5.5 bits | < 1.0 bit (single port) |
| H_proto | 1.0 – 2.5 bits | < 0.5 bits (one protocol) |
| H_tcp_flags | 1.5 – 3.0 bits | < 0.3 bits (SYN-only) |

These are approximate — actual thresholds depend on your network's baseline.
Entropy values alone are not used for classification; they are inputs to the RF.

---

## Limitations to Be Aware Of

1. **IP rotation evasion**: Attackers cycling source IPs rapidly can maintain
   H_src_ip ≥ 2 bits, reducing entropy signal. The RF still detects via flow
   rate/volume features, but confidence may be lower (0.70–0.80 range).

2. **SlowLoris FPR**: HTTP-based slow attacks can resemble certain legitimate
   API patterns (long-polling, SSE). If you see SlowLoris alerts for known
   internal services, whitelist those source IPs.

3. **Window warmup**: For the first 1,000 flows after controller restart,
   entropy features have limited window context. Expect slightly higher FPR
   during this period.

4. **Single dataset training**: The model was trained on CIC-DDoS2019.
   Novel attack variants not present in training data may have lower confidence
   or be misclassified. Regular retraining with new captures is recommended.
