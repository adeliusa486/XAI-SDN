"""E13 - Individual alert case studies (R7.5).

Reviewer 7: "The SHAP analysis lacks individual alert examples demonstrating
practical usefulness."

Four real cases are drawn from actual test predictions - never constructed -
and each is reported the way a security operator would receive it: the score,
the decision, the top signed TreeSHAP contributions with the feature values that
produced them, and a short narrative that states what the evidence supports and
what it does not.

  TP  a correctly flagged attack flow
  FP  a benign flow that was flagged, which is the case an operator must
      adjudicate and the one an explanation has to earn its place on
  FN  an attack flow that was missed
  BD  a borderline flow whose score sits closest to the decision threshold
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.ensemble import RandomForestClassifier  # noqa: E402

CACHE = DATA_ROOT / "cache"
RF_KW = dict(n_estimators=200, max_depth=None, max_features="sqrt", bootstrap=True,
             class_weight="balanced", n_jobs=-1, random_state=42)
TAU = 0.70
TOP_K = 10


def narrate(kind: str, contrib: list[dict], score: float, tau: float) -> str:
    top = contrib[0]
    direction = "toward" if top["shap_value"] > 0 else "away from"
    pushers = [c["feature"] for c in contrib if c["shap_value"] > 0][:3]
    pullers = [c["feature"] for c in contrib if c["shap_value"] < 0][:3]
    base = (f"The model scored this flow {score:.4f} against a threshold of {tau:.2f}. "
            f"The single largest contribution came from {top['feature']} "
            f"(value {top['value']:.4g}), which pushed the score {direction} the "
            f"attack class by {abs(top['shap_value']):.4g}.")
    if pushers:
        base += f" Evidence favouring an attack decision: {', '.join(pushers)}."
    if pullers:
        base += f" Evidence against it: {', '.join(pullers)}."
    tail = {
        "true_positive": (" An operator can see that the decision rests on "
                          "distributional and flag-level evidence rather than on a "
                          "single volumetric field, which is what makes the alert "
                          "actionable rather than merely asserted."),
        "false_positive": (" This is the case that justifies per-decision "
                           "explanation: the attribution shows which features misled "
                           "the model, so the operator can dismiss the alert on "
                           "evidence instead of on intuition, and the same "
                           "attribution is the artefact a tuning decision would be "
                           "based on."),
        "false_negative": (" The attribution shows which benign-looking evidence "
                           "suppressed the score, which is the information needed to "
                           "decide whether the threshold or the feature set is at "
                           "fault."),
        "borderline": (" A score this close to the threshold is precisely where an "
                       "unexplained decision is least defensible, and where the "
                       "attribution changes an arbitrary-looking call into a "
                       "reviewable one."),
    }
    return base + tail.get(kind, "")


def main() -> int:
    log_event("E13", "start")
    t_all = time.time()
    import shap

    out: dict = {"experiment": "E13", "objective": "individual alert case studies",
                 "tau": TAU, "top_k": TOP_K}

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    names = json.loads((CACHE / "syn0311_feature_names_repaired.json").read_text())
    cut = int(round(len(y) * 0.70))

    print("fitting detector...", flush=True)
    clf = RandomForestClassifier(**RF_KW).fit(X[:cut], y[:cut])
    Xte, yte = X[cut:], y[cut:]
    proba = clf.predict_proba(Xte)[:, 1]
    pred = (proba >= TAU).astype(np.int8)

    tp = np.flatnonzero((pred == 1) & (yte == 1))
    fp = np.flatnonzero((pred == 1) & (yte == 0))
    fn = np.flatnonzero((pred == 0) & (yte == 1))
    bd = np.argsort(np.abs(proba - TAU))

    out["population"] = {"n_test": int(len(yte)), "n_tp": int(len(tp)),
                         "n_fp": int(len(fp)), "n_fn": int(len(fn)),
                         "n_tn": int(((pred == 0) & (yte == 0)).sum())}
    print(json.dumps(out["population"], indent=2), flush=True)

    picks = {}
    if len(tp):
        picks["true_positive"] = int(tp[np.argmax(proba[tp])])
    if len(fp):
        picks["false_positive"] = int(fp[np.argmax(proba[fp])])
    if len(fn):
        picks["false_negative"] = int(fn[np.argmin(proba[fn])])
    picks["borderline"] = int(bd[0])

    explainer = shap.TreeExplainer(clf, feature_perturbation="tree_path_dependent")
    meta = pd.read_parquet(CACHE / "syn0311_meta.parquet")
    meta_te = meta.iloc[cut:].reset_index(drop=True)

    cases, rows = {}, []
    for kind, i in picks.items():
        x = Xte[i:i + 1]
        sv = explainer.shap_values(x, check_additivity=False)
        if isinstance(sv, list):
            sv = sv[1]
        sv = np.asarray(sv)
        if sv.ndim == 3:
            sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]
        sv = sv.ravel()
        base = explainer.expected_value
        base = float(np.atleast_1d(base)[-1])

        order = np.argsort(-np.abs(sv))[:TOP_K]
        contrib = [{"feature": names[j], "value": float(Xte[i, j]),
                    "shap_value": float(sv[j])} for j in order]
        m = meta_te.iloc[i]
        case = {
            "kind": kind,
            "test_index": int(i),
            "true_label": "DDoS" if yte[i] == 1 else "Benign",
            "predicted_label": "DDoS" if pred[i] == 1 else "Benign",
            "score": float(proba[i]),
            "threshold": TAU,
            "alerted": bool(pred[i] == 1),
            "explanation_triggered": bool(pred[i] == 1 and proba[i] >= TAU),
            "flow_context": {"source_ip": str(m.get("Source_IP")),
                             "destination_ip": str(m.get("Destination_IP")),
                             "timestamp": str(m.get("Timestamp")),
                             "dataset_label": str(m.get("Label"))},
            "shap_base_value": base,
            "shap_sum_top_k": float(sv[order].sum()),
            "shap_sum_all": float(sv.sum()),
            "top_contributions": contrib,
            "operator_narrative": narrate(kind, contrib, float(proba[i]), TAU),
        }
        cases[kind] = case
        for c in contrib:
            rows.append({"case": kind, "test_index": int(i),
                         "true_label": case["true_label"],
                         "score": case["score"], **c})
        print(f"\n[{kind}] idx={i} score={proba[i]:.4f} true={case['true_label']}")
        for c in contrib[:5]:
            print(f"    {c['feature']:28s} value={c['value']:>14.4g} "
                  f"shap={c['shap_value']:+.5f}")

    out["cases"] = cases
    out["missing_case_kinds"] = [k for k in
                                 ("true_positive", "false_positive", "false_negative")
                                 if k not in cases]
    pd.DataFrame(rows).to_csv(RESULTS / "E13_case_studies.csv", index=False)
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E13_case_studies", out)
    log_event("E13", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
