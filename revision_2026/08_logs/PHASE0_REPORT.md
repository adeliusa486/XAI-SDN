# Phase 0 Report — Ground Truth, Workspace, Environment Lock

Status: **COMPLETE**. All six exit criteria met.

## 0.1 Submitted artefacts frozen

`REVISION_2026/00_submitted/` now holds a read-only copy of the submitted package:
`XAI-SDN-journal.tex` (55,354 B, 575 lines), `references.bib`, `XAI-SDN-journal.bbl`,
`XAI-SDN-journal.pdf` (1,285,282 B), `entropyengine_standalone.pdf`, `ieeeaccess.cls`,
and `figures/` (8 files). This is the latexdiff "old" side for Phase 7.

## 0.6 Source provenance verified

Recompiling the frozen `.tex` with `latexmk -pdf` produced a PDF of **exactly
1,285,282 bytes, 11 pages, 53,072 extracted characters, one `??` occurrence** — identical
to the submitted PDF on every measure. `IEEE_Access/XAI-SDN-journal.tex` is therefore
confirmed to be the exact source that was reviewed, and the LaTeX toolchain reproduces it.

## 0.2 Inventory extracted

| Item | Count |
|---|---|
| Pages | 11 |
| References | 32 |
| Sections / subsections | 36 |
| Figures | 9 |
| Tables | 9 |
| Numbered equations | 7 |
| Algorithms | 1 |

### Reviewer claims independently verified against the submitted PDF

| Reviewer claim | Verification |
|---|---|
| R1.1 / R2.8 broken `Eq. (??)` | **Confirmed.** The literal string sits *inside* the graphic `entropyengine_standalone.pdf` (Fig. 2), rendered from `\node ... {Eq.~(\ref{eq:rolling}) Applied Per Feature}` in a standalone file that has no `eq:rolling` label. Not fixable in the manuscript body — the figure must be rebuilt. |
| R2.8 "Detection Threshold (au)" | **Confirmed.** `fig_sensitivity.pdf` renders `Detection Threshold (au)` and `(a) Threshold (au) Sensitivity`. A `\tau` lost its backslash in the axis label. |
| R7.6 Fig. 6 does not match its caption | **Confirmed.** The caption promises "ROC curve and FPR vs. threshold"; the panels actually show *Accuracy* vs. threshold and *Accuracy* vs. window size, with FPR on twin axes. No ROC curve appears in the figure. |
| R7.6 baseline ROC curves missing | **Confirmed.** §VI-E "ROC analysis" contains prose describing ROC curves but contains **no figure at all**. |
| R5.6 not all floats are cited | **Confirmed.** Never referenced in the body text: Fig. 7 (`fig:ablation`), Fig. 9 (`fig:radar`), Table 5 (`tab:ablation`), Eq. (4) (`eq:metrics`), Eq. (5) (`eq:srcip`). |
| R5.5 arXiv refs 23, 25, 31, **34** | **Partially confirmed.** The submitted PDF has only **32** references, so #34 does not exist. The preprint-flavoured entries are #23 `doshivelez2017rigorous`, #25 `lundberg2017shap` (arXiv URL), #31 `kumar2026meta` ("arXiv preprint"), #32 `you2026plaa` ("arXiv preprint"). Treat the request as 23, 25, 31, 32. |
| R1.2 / R2.8 / R6.2 informal English | **Confirmed.** 28 automated hits, including every phrase the reviewers quoted, each now line-located: "bring the entire network to its knees" (L53), "It looked at 60,606 flows" (L55), "to be able to see the whole network" (L69), "Some of them use something called" (L71), "a bunch of" (L209), "things like destination IP addresses" (L216), "Here are the details" (L235), "It gets 99.9987% accuracy" (L313), "lightning speeds" / "highly compelling" (L541). |
| R2.1 / R2.2 / R6.8 over-claiming | **Confirmed.** 26 automated hits: "exceptional robustness", "generalizes robustly", "This confirms", "exceptionally robust", "exceptionally negligible", "surpasses", "outperforming", "real-world SDN deployment". |

Machine-readable output: `00_submitted/inventory/{inventory.json, defects.json, uncited_floats.json, pdf_text.txt}`.

## 0.3 Workspace

`REVISION_2026/{00_submitted, 01_matrix, 02_data, 03_experiments, 04_results, 05_manuscript,
06_figures, 07_response, 08_logs, 09_package}`.

Data and the virtual environment deliberately live **outside** OneDrive at
`C:/Users/adeel/xaisdn_revision/` so that ~10 GB of CSVs and ~20k venv files are never synced.
Paths are centralised in `03_experiments/common/paths.py`.

## 0.4 Environment locked

The Microsoft Store Python had a **broken PyTorch install** (`torch.__file__ is None`,
only dunder attributes present), which would have silently failed the deep-learning
baselines Reviewer 4 asked for. A clean venv was built instead.

| Package | Version |
|---|---|
| Python | 3.11.9 |
| numpy / pandas / scipy | 2.3.5 / 2.3.3 / 1.17.1 |
| scikit-learn | 1.8.0 |
| shap | 0.51.0 |
| xgboost / lightgbm | 3.2.0 / 4.7.0 |
| **torch** | **2.14.0+cpu** (CUDA unavailable by construction, 14 threads) |
| matplotlib / seaborn / statsmodels | 3.10.9 / 0.13.2 / 0.14.6 |
| psutil / networkx / datasketch / pyarrow | 7.2.2 / 3.6.1 / 2.0.0 / 25.0.1 |
| **os-ken** | installed, OpenFlow 1.3 (`OFP_VERSION = 4`) verified importable |

Installing the **CPU-only** torch wheel is deliberate: Reviewer 1 (1.4, 1.5, 1.6) asked for
CPU-only baselines, and a build with no CUDA makes accidental GPU measurement impossible.
Full freeze: `08_logs/pip_freeze.txt` (74 packages), `08_logs/env_lock.json`.

## 0.5 Hardware recorded (answers R5.3, R1.4, R1.6)

| Field | Value |
|---|---|
| CPU | 13th Gen Intel Core i9-13900H |
| Cores / threads | 14 / 20 |
| Base clock | 2.60 GHz |
| L2 / L3 cache | 11,776 KB / 24,576 KB |
| RAM | 47.64 GB |
| OS | Windows 11 Pro, build 26120, 64-bit |
| GPUs present | Intel Iris Xe; NVIDIA GeForce RTX 4060 Laptop |
| Machine | ASUS Zenbook UX6404VV |

A GPU exists on this machine, which is exactly why the CPU-only torch build matters: every
timing number in the revision will be CPU-measured on this single machine, making Table 8's
latency and throughput columns mutually comparable for the first time.

`08_logs/hardware.json`. Every result file written from now on carries this stamp
automatically via `paths.hardware_stamp()`.

---

## Findings carried forward into Phase 1 and Phase 2

Two implementation defects were discovered while tracing the entropy pipeline. Both are
things Reviewer 7 sensed but could not see from the PDF.

**F1 — `H_ttl` is identically zero.** `features/pipeline.py:269` hard-codes `"ttl": 64` for
every flow record. Shannon entropy of a constant is 0, so one of the eight advertised
entropy features carries no information at all, in training or in inference. CICFlowMeter
CSVs contain no TTL column, so this cannot be fixed by remapping. This is the substance of
**R7.2** ("entropy calculations lack detail, particularly TTL data"). Resolution proposed for
Phase 2.3: replace TTL entropy with **source-port entropy**, which is genuinely present in
the CSVs and genuinely discriminative for SYN floods with randomised source ports. That
preserves eight entropy features and the 88-dimensional vector, makes all eight informative,
and is disclosed in the response letter as a correctness fix. To be confirmed empirically
against the real data before adoption.

**F2 — the paper's Eq. (6) is not the equation the code implements.** The manuscript states
`H' = H − φ(n_a) + φ(n_a+1) − φ(n_d) + φ(n_d−1)` with `φ(m) = −(m/N)log₂(m/N)`. The actual
offline implementation (`features/entropy.py`) maintains a running sum `S = Σ c·log₂c` and
returns `H = log₂(|W|) − S/|W|`, updating `S` by `−c log₂c + (c±1)log₂(c±1)` on each
admit/evict. The two forms agree only when `N` is fixed, and the code's form is the correct
general one. This matters directly for **R6.4** (catastrophic cancellation): the drift
analysis must be run against the `S`-form that is actually executed, and the manuscript's
equation must be corrected to match the implementation.

Neither finding changes any published number yet; both are inputs to Phase 2's baseline
reproduction, where their effect will be measured rather than assumed.

## Phase 2 prefetch already running

Dataset download started in the background (`03_experiments/common/download_data.py`,
resume-capable, SHA-256 manifested). Confirmed reachable and downloading:
CIC-DDoS2019 `03-11/Syn.csv` — **1,877,372,065 bytes, the exact "1.87 GB Syn.csv" the
submitted paper describes**.
