# Phase 2 Report — Data Acquisition and Baseline Reproduction

Status: **COMPLETE**. All five exit criteria met. Three findings materially change
what the revised manuscript can claim.

---

## 2.1 Datasets acquired and verified

Eight files, resume-capable download, SHA-256 manifested at
`C:/Users/adeel/xaisdn_revision/data/MANIFEST.json`.

| Key | Bytes | SHA-256 (first 12) |
|---|---:|---|
| CIC-DDoS2019 `03-11/Syn.csv` | 1,877,372,065 | `603648e7c56e` |
| CIC-DDoS2019 `03-11/LDAP.csv` | 871,398,918 | `d1cfc7cb9252` |
| CIC-DDoS2019 `03-11/NetBIOS.csv` | 1,418,468,105 | `ddd2e8cd76c1` |
| CIC-DDoS2019 `03-11/Portmap.csv` | 78,611,080 | `d0148da21f3c` |
| CIC-DDoS2019 `01-12/Syn.csv` | 637,312,127 | `05a272a7005b` |
| InSDN `Dataset.csv` | 140,906,680 | `a93d51e1c2de` |
| CIC-IDS2017 Friday DDoS | 77,123,859 | `6ff1580f5f81` |
| CIC-IDS2017 Monday benign | 176,927,918 | `852c4beb34ed` |

`03-11/Syn.csv` is byte-for-byte the "1.87 GB `Syn.csv`" the submitted manuscript names.

## 2.2 Profiles and the cross-dataset schema

`02_data/data_profile.json`, `02_data/schema_matrix.csv`.

| Dataset | Rows | Cols | CIC features present | Labels |
|---|---:|---:|---|---|
| CIC-DDoS2019 03-11 Syn | 4,320,541 | 88 | 80/80 | Syn 4,284,751 · BENIGN 35,790 |
| CIC-DDoS2019 03-11 LDAP | 2,113,234 | 88 | 80/80 | LDAP, NetBIOS, BENIGN |
| CIC-DDoS2019 03-11 NetBIOS | 3,455,899 | 88 | 80/80 | NetBIOS, BENIGN |
| CIC-DDoS2019 03-11 Portmap | 191,694 | 88 | 80/80 | Portmap, BENIGN |
| CIC-DDoS2019 01-12 Syn | 1,582,681 | 88 | 80/80 | Syn, BENIGN |
| InSDN | 343,889 | 84 | 79/80 | attack 275,465 · benign 68,424 |
| CIC-IDS2017 Friday DDoS | 225,745 | 79 | 78/80 | DDoS 128,027 · BENIGN 97,718 |
| CIC-IDS2017 Monday | 529,918 | 79 | 78/80 | BENIGN 529,918 |

Transfer schemas computed for Reviewer 4's cross-dataset request:

- **CIC-DDoS2019 → InSDN: 87 dimensions** (79 shared CICFlowMeter features + all 8
  entropy features). A clean transfer.
- **CIC-DDoS2019 → CIC-IDS2017: 82 dimensions** (78 shared + 4 entropy).
  The ISCX export carries no Source IP, Destination IP, Source Port or Protocol column,
  so `H_src_ip`, `H_dst_ip`, `H_src_port` and `H_proto` cannot be computed there. This is a
  property of the corpus, not a choice, and it is stated in the paper rather than worked
  around.

## 2.3 The TTL question, answered (R7.2)

`features/pipeline.py` hard-codes `"ttl": 64` for every flow record. Measured on all
3,590,794 cleaned flows:

| | value |
|---|---|
| `H_ttl` maximum | 1.78 × 10⁻¹⁵ |
| `H_ttl` standard deviation | 7.22 × 10⁻¹⁸ |
| distinct values | 4 (all indistinguishable from zero) |
| **single-feature AUC** | **0.500021** |
| mutual information with the label | 0.000396 bits |

An AUC of 0.500021 is chance. One of the eight advertised entropy features contributed
nothing to any result in the submitted paper.

**Repair adopted.** TTL entropy is replaced by **source-port entropy**, which is present in
every CICFlowMeter export and is discriminative for randomised-source floods:
std 0.245, 26,420 distinct values, single-feature AUC 0.9876, and the highest mutual
information of any entropy feature (0.0602 bits). The 88-dimensional representation is
preserved. Effect of the repair on the headline metrics, temporal split:

| | original (defective) | repaired |
|---|---|---|
| Accuracy | 0.997538 | 0.997579 |
| Macro-F1 | 0.978467 | 0.978813 |
| ROC-AUC | 0.999721 | 0.999728 |
| FPR | 0.001117 | 0.000953 |
| FNR | 0.002501 | 0.002464 |

Small but uniformly positive, and the feature set is now honest. Full specification table
for the manuscript: `04_results/E16_feature_specification.csv`.

## 2.4 and 2.5 Baseline reproduction — the central finding

Cleaning: 4,320,541 → **3,590,794** rows (−283,076 NaN/Inf, −446,671 duplicates).
The manuscript reports 3,592,660. The 1,866-row gap is **0.052%** and is attributable to the
exact column subset used for duplicate detection. It does not affect any conclusion and is
disclosed.

| Protocol | n train | n test | test benign | Accuracy | Macro-F1 | ROC-AUC | FPR | FNR |
|---|---:|---:|---:|---|---|---|---|---|
| **Submitted paper** | 2,514,862 | 1,077,798 | 9,311 | 99.9987% | 99.9621% | 1.0000 | 0.0537% | 0.0008% |
| **Stratified random 70/30** | 2,513,555 | 1,077,239 | 9,304 | 99.9981% | 99.9458% | 1.000000 | 0.0000% | 0.0019% |
| **Temporal 70/30** | 2,513,556 | 1,077,238 | 30,432 | 99.7538% | 97.8467% | 0.999721 | 0.1117% | 0.2501% |

**The submitted results were produced by a stratified random split, not the temporal split
the manuscript describes.** Three independent lines of evidence:

1. Arithmetic. 9,311 and 1,068,487 are exactly 30% of 31,037 and 3,561,623, which is what a
   stratified split produces by construction and what a temporal split has no reason to produce.
2. Reproduction. The stratified split lands within 0.016 macro-F1 points of the published
   figure and within 7 flows of the published test-benign count.
3. Source. The repository's own `features/pipeline.py` calls
   `train_test_split(..., stratify=y_raw.values, random_state=42)` after sorting by
   timestamp. The sort is performed; the temporal split is not.

Section VI-A of the submitted manuscript states: *"The data is partitioned into training
(70%) and test (30%) sets using a temporal split, in which all training flows precede all
test flows chronologically; this avoids leakage of near-duplicate flows across the split."*
That sentence does not describe what was done.

Under a genuine temporal split the headline macro-F1 falls from 99.96% to **97.85%**, the
FNR rises by a factor of 130, and the test partition's benign share rises from 0.86% to
2.82%, which shows directly that benign flows are concentrated late in the capture.

This is the leakage Reviewers 4 and 6 suspected and Reviewer 7 asked about. **The revision
adopts the temporal split as the primary protocol**, reports the stratified result beside it
as a quantified demonstration of how much split methodology inflates results on this
benchmark, and presents that gap as a methodological contribution rather than a footnote.

## Additional findings carried into Phase 3

**12 of the 80 CICFlowMeter features are constant** in the SYN partition and carry no
information: `Bwd_PSH_Flags`, `Fwd_URG_Flags`, `Bwd_URG_Flags`, `FIN_Flag_Count`,
`PSH_Flag_Count`, `ECE_Flag_Count`, and all six bulk-rate fields. With `H_ttl` that makes 13
dead columns out of 88. This is direct evidence for Reviewer 6's redundancy objection (R6.6)
and is quantified further in E6.

Note that `FIN_Flag_Count` is constant here, yet the submitted paper names it among the
three most influential SHAP features and Fig. 4 shows it ranked third. A constant column
cannot influence a prediction. That inconsistency is resolved by the recomputed SHAP
analysis in E6.

**The entropy features dominate single-feature separability.** Ranked by direction-agnostic
single-feature AUC on the repaired set:

| Rank | Feature | AUC | Mutual information |
|---:|---|---|---|
| 1 | H_dst_ip | 0.9986 | 0.0445 |
| 2 | H_src_ip | 0.9983 | 0.0445 |
| 3 | H_pkt_len | 0.9954 | 0.0442 |
| 4 | H_proto | 0.9922 | 0.0441 |
| 5 | H_dst_port | 0.9902 | 0.0445 |
| 6 | H_src_port | 0.9876 | 0.0602 |
| 7 | H_iat | 0.9508 | 0.0292 |
| 8 | H_tcp_flags | 0.9471 | 0.0239 |
| 9 | ACK_Flag_Count | 0.8938 | 0.0813 |

Every entropy feature outranks every CICFlowMeter feature. That is good news for the
paper's central claim and a serious question at the same time: a single windowed entropy
separating the classes at AUC 0.9986 may be reading attack structure, or it may be reading
the clock, because window entropy over a temporally segregated capture partly encodes which
phase of the capture a flow belongs to.

E5 was extended to settle this with a direct diagnostic: the entropy features are recomputed
over a **shuffled arrival order**, which destroys temporal structure while leaving every
per-flow value untouched. A large drop in separability means the features were reading the
clock; a small drop means they were reading the traffic. This is the sharpest available test
of Reviewer 6's "dataset artifacts rather than genuine detection capability" charge, and the
answer goes into the paper whichever way it comes out.

## Infrastructure verified ahead of Phase 7

- **latexdiff** was missing its `Algorithm::Diff` Perl dependency; a local copy is installed
  and the source-level diff now runs.
- **Yellow highlighting works.** The editor asked for "the yellow highlight tool within the
  pdf file". `soul`'s `\hl` cannot survive citations and maths in this class, so the
  highlighted PDF is produced by comparing rendered word streams and adding genuine PDF
  highlight annotations. Verified end to end on a probe edit: four changed words, four
  yellow annotations, correct positions, colour `[1.0, 1.0, 0.0]`.
- **Real OpenFlow 1.3 encoding verified.** `OFPT_HELLO`, `OFPT_FEATURES_REPLY`,
  `OFPT_PACKET_IN`, `OFPT_FLOW_MOD`, `OFPT_ECHO_REPLY` all round-trip through the os-ken
  parser, so the Phase 3J testbed will exchange genuine protocol messages.

## Phase 3 launched

The 13-experiment queue is running under `03_experiments/run_queue.py`, one experiment at a
time so that no run contaminates another's timing measurements. Progress in
`08_logs/queue_status.json` and per-experiment logs in `08_logs/E*.log`.
