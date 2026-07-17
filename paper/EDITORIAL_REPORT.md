# Editorial Report — XAI-SDN Journal Manuscript

**Files:** `XAI-SDN-journal.tex` (Elsevier `elsarticle`, 5p two-column), `references.bib`, `figures/architecture.jpeg`, compiled `XAI-SDN-journal.pdf` (9 pages).
**Target venue class:** Q1 subscription journals with **no submission or publication fee** — recommended in order: *Computers & Security* (Elsevier), *Computer Networks* (Elsevier), *IEEE TNSM*. The `\journal{}` line is set to Computers & Security; change it if you pick another.

## 1. Key improvements made

**Structure (conference → journal):**
- Expanded to full journal structure: Introduction with explicit gap/requirements framing and contributions, three-subsection Related Work with a new qualitative comparison table (Table 1), a **new Threat Model & Problem Formulation section** (Eq. 1: decision rule + explanation-efficiency requirement), Framework, Mathematical Foundation, Setup, Results, Ablation/Comparative, **new Discussion section** (deployment considerations + limitations/threats to validity), Conclusion, and Elsevier declarations (CRediT, competing interests, data availability).
- Added a **new derivation** of the O(1) rolling entropy update (Eq. 3) — previously the O(1) claim was asserted but never shown. This is pure algebra on the paper's own definition, no new empirical claims.
- Added evaluation-metric definitions (Eq. 7), per-class results table, and reproducibility paragraph.

**Figures (all new figures use only real archived data):**
- Fig. 2 (new): TikZ rolling-entropy engine diagram.
- Fig. 3 (new): confusion-matrix heatmap — counts 9,306/5/9/1,068,478 from your artifacts (previously a commented-out table).
- Fig. 4 (new): log-scale latency breakdown — values from your latency table.
- Fig. 5 (upgraded): SHAP importance extended from 8 to 12 features using `model/artifacts/shap_global_importance.csv` (real values).
- Fig. 7 (new): ablation grouped bars with error bars from `model/artifacts/ablation_multiseed.json` (5 archived seeds — real).
- Fig. 8 (new): F1-vs-throughput trade-off scatter from `model/artifacts/baseline_results.json` (real).
- Fig. 6 (ROC): **kept exactly as in your accepted paper** — see action item 3.

**Tables:** added related-work comparison, per-class metrics (computed from your own confusion matrix — they reproduce your macro-F1 of 99.9621% exactly), ablation mean±std table, and train-time column in the baseline table (from `baseline_results.json` and `metrics.json`).

**Writing:** removed all duplicated/garbled sentences from the conference version (e.g., the doubled "single point of failure" sentence, "Gaspagaspar2024xair et al.", the doubled controlled-protocol sentence); consistent tense and terminology throughout; every figure/table/equation has caption, label, and in-text reference.

**Bibliography:** removed duplicate `mckeown2008openflow` entry; normalized all entries to `doi = {...}` fields; consistent title casing; all 17 entries are cited.

## 2. Corrections that change wording (not results)

- The conference version's ablation sentence was self-contradictory ("significantly outperforms entropy-only (p=0.0020) … however the incremental improvement **over entropy** is not statistically significant (p=0.2500)"). Based on your artifacts, p=0.2500 clearly refers to the **CIC-only** comparison; the sentence now reads "the increment over CIC-only features … (p=0.2500)". **Please confirm this matches your analysis.**
- The latency "corrects a prior overstatement" sentence was reworded to criticize the literature generally rather than your own prior text, which is the appropriate framing for a journal version.

## 3. Action items for authors (must do before submission)

1. **10-seed vs 5-seed inconsistency (important).** The prose claims a 10-seed Wilcoxon campaign (p=0.0020 / 0.0078 / 0.2500) and a 10-seed stratified-split FPR of 5.77% ± 5.23%, but the public repo archives only **5 seeds** (42, 123, 456, 789, 1024), whose Wilcoxon p-values are 0.0625/0.0625/0.125 (n=5 cannot reach p=0.002). Either (a) re-run and archive the full 10-seed campaign so the claims are reproducible, or (b) change the prose to the archived 5-seed numbers. The new Table 7/Fig. 7 are explicitly captioned "five archived seeds" so they are internally consistent either way.
2. **Repo/paper number drift.** `metrics.json` says n_train = 2,514,860; the paper says 2,514,862. Trivial but a reviewer running your code will notice. Regenerate or fix the text.
3. **ROC curve provenance (Fig. 6).** The baseline ROC coordinates look hand-placed rather than computed. For a Q1 journal, regenerate this figure from actual `roc_curve()` outputs of the archived models, or a reviewer request for the underlying data will be awkward.
4. **Tangential citations.** `toqeer2026climate` (climate digital twins), `jan2025blockchain`, and `akarma2026agents` are cited for claims about DDoS/XAI where they are at best loosely related. Reviewers increasingly flag this as citation padding, and Computers & Security desk-checks it. Recommend removing them or citing genuinely relevant regulatory/XAI-in-SOC literature instead.
5. **Prior conference acceptance.** Formally withdraw the paper from the conference that accepted it and get written confirmation it will not appear in their proceedings; declare the history in your cover letter. Journals treat undisclosed prior publication as dual submission.
6. **Strengthen for Q1 acceptance odds (recommended, not required to compile):** add at least one more CIC-DDoS2019 partition (UDP or LDAP) and ideally one more dataset (InSDN / CIC-IDS2017). Your repo's pipeline already supports this. The single-partition scope is the most likely rejection reason; the Limitations section currently acknowledges it honestly.
7. Fill in the DNN train time in Table 8 (currently "—") if you have it, and verify author name spelling "Hasan Razzaqi" vs "Hassan Ali Razzaqi" (both appear in your own bib entry `jan2025blockchain`).

## 4. Placeholders / invented content

**None.** Every number in the manuscript comes from your accepted conference paper or the archived artifacts in the XAI-SDN repo (`metrics.json`, `evaluation_results.json`, `baseline_results.json`, `ablation_multiseed.json`, `shap_global_importance.csv`). The per-class table was derived arithmetically from your published confusion matrix. No fabricated results, datasets, citations, or DOIs were added.

## 5. Build instructions

```
pdflatex XAI-SDN-journal
bibtex   XAI-SDN-journal
pdflatex XAI-SDN-journal
pdflatex XAI-SDN-journal
```
Compiles clean on MiKTeX with `elsarticle`; only cosmetic warnings remain.
