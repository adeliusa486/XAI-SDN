# Phase 4 Report — References

Status: **COMPLETE** for the bibliography itself. Integration of the new citations into
the manuscript body happens in Phase 5, where each one is placed in the section that
actually discusses it.

Phase 4 was run concurrently with the Phase 3 experiment queue because it depends on no
experimental result.

---

## 4.1 The four preprints Reviewer 5 flagged (R5.5)

Reviewer 5 asked us to replace arXiv references "23, 25, 31, and 34" with their published
versions. The submitted PDF contains **32** references, so item 34 does not exist; the
preprint-flavoured entries are 23, 25, 31 and 32. All four were checked against the
publisher record.

| # | Entry | Finding | Action |
|---|---|---|---|
| 23 | Doshi-Velez & Kim, *Towards a rigorous science of interpretable machine learning* | **No peer-reviewed version exists.** The work has remained arXiv:1702.08608 since 2017. | Kept, but recast from a fake `@article` with `journal = {arXiv preprint arXiv:1702.08608}` into a proper preprint entry that renders as `arXiv:1702.08608 [stat.ML], 2017, preprint`. |
| 25 | Lundberg & Lee, *A unified approach to interpreting model predictions* | **Published**: Advances in Neural Information Processing Systems 30 (NeurIPS 2017), pp. 4765–4774. The submitted entry pointed at the arXiv DOI instead. | Replaced with the proceedings record, editors and publisher included; arXiv URL removed. |
| 31 | Kumar, Ishigaki & Belman, *Enhanced multi-class DDoS attack identification using a meta-learning ensemble* | **Still a preprint**, arXiv:2607.16521, 17 July 2026. No published version located. | Kept as a properly formatted preprint with its identifier, which the submitted entry lacked entirely. |
| 32 | You *et al.*, *PLAA: Packet-level adversarial attacks in network traffic detection* | **Still a preprint**, arXiv:2606.28439, 26 June 2026. No published version located. | Same treatment. |

Two of the four had a published version and now cite it. The other two do not, and saying so
plainly is a better answer than silently leaving `journal = {arXiv preprint}` in the file.
Both now carry their arXiv identifier and are explicitly marked as preprints, which is what
the IEEE reference style requires of unpublished work.

## 4.2 Reviewer 2's requested reference (R2.9)

Verified against the publisher record:

> F. Chen, C. Ding, X. Hu, X. He, X. Yin, J. Yang, and Z. Zhao, "Tensor Poincaré plot index:
> A novel nonlinear dynamic method for extracting abnormal state information of pumped
> storage units," *Reliability Engineering & System Safety*, vol. 254, part B, art. no.
> 110607, 2025, doi: 10.1016/j.ress.2024.110607.

The work is genuinely relevant despite the different application domain. It constructs a
nonlinear dynamic index by decomposing an operational signal across time and frequency
scales and transforming the components into Poincaré plots, in order to expose abnormal
state information that scale-agnostic statistics miss. That is the same methodological move
this paper makes with windowed Shannon entropy: replace a raw per-observation statistic with
a distributional summary computed over a window, because the abnormality lives in the
distribution rather than in any single sample. It will be cited in a new §II-D on nonlinear
dynamic indices for abnormal-state detection, with the connection stated rather than
implied.

## 4.3 Reviewer 3's requested reference (R3.5)

Verified against IEEE Xplore:

> M. Swaminathan, O. W. Bhatti, Y. Guo, E. Huang, and O. Akinwande, "Bayesian learning for
> uncertainty quantification, optimization, and inverse design," *IEEE Transactions on
> Microwave Theory and Techniques*, vol. 70, no. 11, pp. 4620–4634, Nov. 2022,
> doi: 10.1109/TMTT.2022.3206455.

Placed where it does real work rather than as a courtesy citation. The revision reports a
detector that emits a calibrated score and gates both alerting and explanation on that
score; the question of how much confidence to place in a single prediction is exactly the
uncertainty-quantification problem this reference addresses. It is cited in §II-C alongside
the discussion of attribution, and again in the limitations, where Bayesian treatment of
detector confidence is named as future work.

## 4.4 New references for Reviewer 4's requested subsection (R4.3)

Reviewer 4 asked for a dedicated discussion of deep learning for SDN DDoS detection and
QoS-aware software-defined IoT, covering accuracy/latency/explainability trade-offs,
adaptation to SD-IoT constraints, lightweight and edge inference, and cross-layer
coordination between security enforcement and QoS policy. Six references were added, each
verified against the publisher record and each mapped to a specific claim in the new
subsection:

| Reference | What it supports |
|---|---|
| Bhayo *et al.*, *Eng. Appl. Artif. Intell.* **123**:106432, 2023 | ML-based DDoS detection inside an SD-IoT controller, with a testbed — the closest prior work to the SD-IoT adaptation we discuss |
| Gali & Mahamkali, *Concurrency Computat. Pract. Exper.* **37**:e70045, 2025 | Deep-learning detection combined with QoS-aware secure routing in SDN-IoT — the cross-layer security/QoS coordination Reviewer 4 names |
| Jain, Shukla & Goel, *Cluster Computing* **27**(9):13129–13164, 2024 | Recent survey anchoring the SDN DDoS detection and mitigation landscape |
| Lo *et al.*, *Proc. IEEE/IFIP NOMS*, pp. 1–9, 2022 | E-GraphSAGE, the reference GNN intrusion detector; contextualises our own GNN baseline in E9 |
| Manocchio *et al.*, *Expert Syst. Appl.* **241**:122564, 2024 | FlowTransformer; contextualises our transformer baseline in E9 |
| Misrak & Melaku, *Discover Internet of Things* **5**:57, 2025 | Quantised, compressed IDS for resource-constrained deployment — the lightweight/edge inference thread |

Two further references support the explainability discussion:

| Reference | What it supports |
|---|---|
| Al & Sağıroğlu, *Eng. Appl. Artif. Intell.* **144**:110145, 2025 | Current state of XAI in intrusion detection; updates the 2018–2020 XAI citations |
| Swaminathan *et al.* (above) | Uncertainty quantification of model predictions |

## 4.5 Recency (R5.4)

| | Submitted | Revised |
|---|---|---|
| Total references | 32 | **41** |
| Published 2023 or later | 3 (9%) | **10 (24%)** |
| Published 2020 or later | 9 (28%) | **17 (41%)** |
| Median year | 2017 | 2017 |

The median is unchanged by design: the foundational citations that should be old — Shannon
1948, Breiman 2001, Cortes & Vapnik 1995, McKeown 2008 — are old for good reason, and
churning them to raise an average would be worse scholarship, not better. What changed is
the contemporary layer: the revision now cites current work on SDN DDoS detection,
SD-IoT with QoS constraints, GNN and transformer intrusion detectors, lightweight edge
inference and the present state of XAI for IDS.

## 4.6 Validation

`references.bib` was compiled against `IEEEtran.bst`:

- **41 entries, 41 `\bibitem`s produced, zero BibTeX warnings, zero errors.**
- No duplicate keys.
- No entry retains `journal = {arXiv preprint}`.
- Non-ASCII author names brace-protected so BibTeX derives the right initials
  (`Ş. Sağıroğlu`, not `c. Sağı roğlu`).
- Every new entry inspected in its rendered IEEE form.

Entry types: 23 `@article`, 15 `@inproceedings`, 3 `@misc` (the three genuine preprints).

## Not yet done

Citing these in the manuscript text. That is Phase 5, and it is deliberately deferred so
each reference is placed where it is actually discussed rather than appended to a list.
Reviewer-suggested references that turned out not to strengthen a specific argument were
not added for the sake of adding them, in line with the Associate Editor's instruction.
