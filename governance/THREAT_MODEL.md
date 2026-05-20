# XAI-SDN Threat Model

## System Assets

| Asset | Sensitivity | Justification |
|-------|-------------|---------------|
| Trained RF model (`rf_model.pkl`) | High | Exposes decision boundary; enables evasion |
| SHAP attribution values | High | Reveals exactly which features cause detection |
| Alert database | Medium | Contains attacker IP addresses and flow metadata |
| API endpoints | Medium | Infer endpoint exposes SHAP; alerts expose network topology |
| SDN controller | Critical | Compromise grants full network control |

---

## Threat Scenarios

### T1 — SHAP-Guided Evasion
**Attacker goal**: Craft DDoS traffic that evades the classifier.  
**Method**: Query `/api/v1/infer` repeatedly with modified feature vectors to find the
decision boundary. Use returned SHAP values to identify which features to manipulate.  
**Likelihood**: Medium (requires API access)  
**Impact**: High (complete evasion of detection)  
**Mitigations**:
- Restrict `/api/v1/infer` to internal/operator network only
- Set `SHAP_ENDPOINT_INTERNAL_ONLY=1` and enforce via nginx ACL
- Rate-limit infer endpoint to 10 req/min per IP
- Never expose raw SHAP values in public-facing alerts

### T2 — Controller Impersonation
**Attacker goal**: Inject fake DDoS alerts to trigger false positives / disrupt network.  
**Method**: Forge HTTP POST requests to `/api/v1/alerts/` from a compromised host.  
**Likelihood**: Medium (no auth in default config)  
**Impact**: High (floods alert DB, disrupts SOC)  
**Mitigations**:
- Add API authentication (bearer token or mTLS client certificates)
- Validate `switch_id` against known DPID whitelist
- Rate-limit alert ingestion per source IP

### T3 — IP Rotation Evasion
**Attacker goal**: Maintain H_src_ip above detection threshold by rotating source IPs.  
**Method**: Use a large botnet (>1000 IPs) so that source IP entropy remains high.  
**Likelihood**: Medium (common in large botnets)  
**Impact**: Medium (entropy signal degraded; RF may still detect via flow volume features)  
**Mitigations**:
- Monitor H_src_ip trend over time (sudden drop from high to moderate)
- Add rate-based features (bytes/s, packets/s) as primary indicators
- Implement adaptive threshold: if flow rate exceeds 10× baseline, lower τ

### T4 — Ryu Controller Compromise
**Attacker goal**: Gain control of the SDN controller to install malicious flow rules.  
**Method**: Exploit Ryu REST API or inject packets to trigger controller vulnerabilities.  
**Likelihood**: Low (Ryu has known CVEs in older versions)  
**Impact**: Critical (full network takeover)  
**Mitigations**:
- Run Ryu behind a firewall; only allow OpenFlow port (6653) from known switch IPs
- Enable OpenFlow TLS (mutual authentication between controller and switches)
- Keep Ryu updated; monitor CVE database
- Run controller in isolated network segment

### T5 — Training Data Poisoning
**Attacker goal**: Inject mislabelled flows into training data to degrade model accuracy.  
**Method**: If retraining is automated from live traffic, inject benign-labelled attack flows.  
**Likelihood**: Low (manual training in current implementation)  
**Impact**: High (backdoored model with hidden evasion)  
**Mitigations**:
- Hash training datasets and verify before each training run
- Validate model F1 ≥ 0.95 on a clean holdout set before deployment
- Sign model artifacts with SHA-256 and verify at load time

---

## Security Checklist (Production)

- [ ] API authentication enabled (bearer token or mTLS)
- [ ] `/api/v1/infer` restricted to internal network (nginx `allow` ACL)
- [ ] `SHAP_ENDPOINT_INTERNAL_ONLY=1` set
- [ ] Alert ingestion rate-limited per source IP
- [ ] OpenFlow TLS enabled between Ryu and all switches
- [ ] Ryu REST API disabled or firewalled
- [ ] Model artifacts signed and verified at load
- [ ] Alert DB access-controlled (not world-readable)
- [ ] All secrets in `.env` rotated from defaults
- [ ] Grafana admin password changed from `admin`
- [ ] TLS termination at nginx for all external traffic
- [ ] Log retention and access auditing configured
