# XAI-SDN — Implementation Plan to Reach 9/10 Q1 Quality

**Current honest rating: 6.5/10.** Presentation is already at Q1 standard (9/10); the gap is entirely in evaluation scope and evidence rigor. This plan closes that gap. Phases are ordered by impact-per-effort; each phase lists exact tasks, commands, expected artifacts, and how the result feeds back into the paper.

**Estimated total effort: 3–5 weeks part-time.** Phases 1–3 are mandatory for 8/10. Phases 4–5 take it to 9/10. Phase 6 is submission logistics.

---

## Phase 1 — Fix reproducibility debts (2–3 days) → removes rejection triggers

### 1.1 Re-run the 10-seed campaign the paper already claims
The prose cites 10-seed Wilcoxon p-values (0.0020/0.0078/0.2500) and stratified FPR 5.77% ± 5.23%, but `model/artifacts/` archives only 5 seeds. A reviewer who clones the repo will find this.

- [ ] Extend `scripts/run_multiseed.sh` seed list to `42 123 456 789 1024 2048 31337 555 777 999` (10 seeds).
- [ ] Run both protocols: (a) ablation configurations (entropy-only / CIC-only / SVM-full / RF-full), (b) stratified random-split FPR replication.
- [ ] Recompute Wilcoxon signed-rank tests (scipy.stats.wilcoxon, two-sided) across the 10 seeds; save to `model/artifacts/ablation_multiseed_10.json`.
- [ ] Update paper Table 7 / Fig. 7 and the significance paragraph with whatever the rerun actually gives — **whatever the numbers are, report them**.

### 1.2 Regenerate the ROC figure from real model outputs
- [ ] Write `scripts/generate_roc.py`: load each archived model (`rf_model.pkl`, baselines retrained per `model/baselines.py`), compute `sklearn.metrics.roc_curve` on the shared test split, export `model/artifacts/roc_curves.json` (FPR/TPR arrays per model).
- [ ] Replace the hand-placed pgfplots coordinates in Fig. 6 with downsampled real curve points (~20 points per curve, log-spaced in FPR).

### 1.3 Align trivial number drift
- [ ] n_train: paper says 2,514,862, `metrics.json` says 2,514,860 — regenerate or fix the text.
- [ ] Confirm author name spelling ("Hasan Razzaqi" vs "Hassan Ali Razzaqi") and DNN train time for Table 8.

**Paper impact:** deletes integrity notes 1–3 from the editorial report. Rating: 6.5 → 7.0.

---

## Phase 2 — Multi-partition evaluation on CIC-DDoS2019 (1 week) → the single biggest win

### 2.1 Data
- [ ] Download from https://www.unb.ca/cic/datasets/ddos-2019.html: `UDP.csv`, `LDAP.csv`, `MSSQL.csv`, `NetBIOS.csv` (Day-1 partitions). Store under `data/raw/` (gitignored).
- [ ] Generalize `scripts/preprocess_data.py`: parameterize input CSV (`--partition udp|ldap|mssql|netbios|syn`), same NaN/dedup/temporal-split protocol, write per-partition splits to `data/splits/<partition>/`.

### 2.2 Experiments (per partition)
- [ ] Train/evaluate the identical 88-dim RF pipeline (same hyperparameters, seed 42, temporal split) → per-partition `metrics.json`, confusion matrix, per-class metrics.
- [ ] **Cross-partition generalization matrix** (the result reviewers actually value): train on SYN → test on UDP/LDAP/MSSQL; train on UDP → test on SYN; etc. Save `model/artifacts/cross_partition.json`.
- [ ] Per-partition SHAP global importance → does the entropy-feature ranking hold across attack types? (This directly strengthens the paper's core claim.)

### 2.3 Paper integration
- [ ] New Table: per-partition accuracy / macro F1 / FPR / FNR / AUC.
- [ ] New Figure: cross-partition generalization heatmap (train-partition × test-partition macro F1).
- [ ] New Results subsection "Generalization across attack vectors"; rewrite abstract/intro/conclusion claims from "the SYN partition" to "five CIC-DDoS2019 attack vectors".
- [ ] Move the "single attack family" limitation from Limitations into resolved territory.

**Rating: 7.0 → 8.0.**

---

## Phase 3 — Second dataset (1 week) → confirms external validity

Recommended: **InSDN** (Elsayed et al., 2020 — an actual SDN testbed dataset, perfect topical fit, ~343K flows, same CICFlowMeter-style features) and/or **CIC-IDS2017** (DoS/DDoS days).

- [ ] Add loader mapping InSDN column names onto the 80-feature schema (`features/pipeline.py` — add `--schema insdn`).
- [ ] Same protocol: temporal split, RF-88, multi-seed, SHAP.
- [ ] Cross-dataset transfer: train CIC-DDoS2019 → test InSDN (expect degradation — **report it honestly**; a candid transfer analysis is itself a contribution reviewers respect).
- [ ] Paper: new table + one paragraph; update Limitations.

**Rating: 8.0 → 8.5.**

---

## Phase 4 — Adversarial robustness analysis (1 week) → the intellectual differentiator

The paper's future-work already names entropy-aware evasion. Doing even a first-order version of it lifts the paper from "solid application" to "contribution with security depth":

- [ ] Implement `attacks/entropy_evasion.py`: an adaptive SYN-flood generator that (a) rotates spoofed source IPs to raise H_src toward benign levels, (b) jitters inter-arrival times, (c) mixes benign-mimicking flows. Parameterize by "evasion budget" (extra addresses/bandwidth the attacker must spend).
- [ ] Measure detection recall vs. evasion budget → cost-to-evade curve.
- [ ] Quantify which feature families degrade (SHAP under attack) — shows CIC + entropy complementarity empirically under adversarial pressure.
- [ ] Optional hardening: adversarial re-training on evasion traffic; report before/after.
- [ ] Paper: new section "Adversarial robustness", 1–2 figures (recall vs budget; SHAP drift under attack).

**Rating: 8.5 → 9.0.** This is the phase that makes reviewers say "accept with minor revisions."

---

## Phase 5 — Live SDN validation (optional, 1 week; substitute for Phase 4 if compute-limited)

- [ ] Stand up Mininet + Ryu using the existing `sdn/topology/mininet_topo.py` and `sdn/controller/xai_sdn_app.py` (Linux VM/WSL2 required).
- [ ] Replay real attack traffic (hping3 SYN flood + iperf benign background) through the live controller path.
- [ ] Measure end-to-end detection latency at the controller (flow-stat poll → alert), controller CPU/memory under attack, and time-to-first-detection.
- [ ] Paper: "Live deployment evaluation" subsection; converts the offline-only limitation into a demonstrated system.

---

## Phase 6 — Submission logistics (parallel, ~1 day of actual work)

- [ ] **Withdraw formally from the prior conference**; obtain written confirmation of non-publication; keep the email.
- [ ] Remove or replace the three tangential citations (`toqeer2026climate`, `jan2025blockchain`, `akarma2026agents`) with genuinely relevant XAI-for-SOC / regulatory-traceability references.
- [ ] Post an arXiv preprint (free, timestamps the work; both target journals allow it).
- [ ] Choose venue and format (both already prepared in `paper/`):
  - *Computers & Security* (Elsevier, no fee) → submit `XAI-SDN-journal-review.pdf` source.
  - *IEEE TNSM* (no mandatory page charges) → submit the `XAI-SDN-journal-ieee.tex` double-column version.
- [ ] Cover letter: disclose conference acceptance/withdrawal history; highlight the honest-reporting angle (FPR split-sensitivity + realistic SHAP latency) as a stated contribution.
- [ ] Update `REPO_REPORT.md` checklist and re-tag the repo (`v1.0-journal-submission`).

---

## Experiment tracking rules (apply to every phase)

1. Every run writes a JSON artifact under `model/artifacts/` with seed, config hash, and library versions (extend `reproducibility_manifest.json` pattern).
2. Every number that enters the paper must be traceable to an artifact committed to the repo.
3. Report means ± std over seeds, never single best runs, for any claim of superiority.
4. Negative/unflattering results (transfer degradation, evasion success) go in the paper — they are what make the favourable results credible.

## Score trajectory

| Milestone | Rating |
|---|---|
| Today | 6.5/10 |
| + Phase 1 (reproducibility debts) | 7.0 |
| + Phase 2 (multi-partition) | 8.0 |
| + Phase 3 (second dataset) | 8.5 |
| + Phase 4 or 5 (adversarial or live SDN) | **9.0** |

The remaining distance from 9 to 10 is novelty of method, which this line of work does not claim — and does not need for a Q1 acceptance.
