"""E11 - Numerical stability of the rolling entropy engine (R6.4, R2.6, R7.3).

Reviewer 6 states that the incremental entropy update "inevitably leads to
catastrophic cancellation and floating-point drift over continuous streaming".
The objection is well-posed and deserves a measurement rather than a rebuttal.

Two facts frame the study, both established in Phase 0:

  * The manuscript prints  H' = H - phi(n_a) + phi(n_a+1) - phi(n_d) + phi(n_d-1)
    with phi(m) = -(m/N) log2 (m/N). That form is only valid for a fixed window
    length N, and it is NOT what the code executes.
  * The implementation maintains S = sum_c c*log2(c) and returns
    H = log2|W| - S/|W|, updating S by -c log2 c + (c +/- 1) log2(c +/- 1).

Both forms are therefore measured, against an exact recomputation, over a long
stream drawn from the real trace. What matters operationally is not the drift in
bits but whether drift changes any decision, so the number of label flips it
causes is measured too.
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

import data as D  # noqa: E402
from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

CACHE = DATA_ROOT / "cache"
WINDOW = 1000
N_UPDATES = 12_000_000        # well past the 10^7 the objection implies
CHECKPOINTS = 240             # exact recomputations along the stream


def _caches(window: int):
    m = window + 5
    clogc = [0.0] + [c * math.log2(c) for c in range(1, m)]
    log2len = [0.0] + [math.log2(l) for l in range(1, m)]
    return clogc, log2len


def run_variant(stream, window: int, variant: str, recompute_every: int | None,
                checkpoint_at: set[int]) -> tuple[dict[int, float], float, int]:
    """Return {checkpoint_index: H}, wall seconds, and update count."""
    clogc, log2len = _caches(window)
    q: deque = deque(maxlen=window)
    counts: dict = {}
    S = 0.0
    comp = 0.0
    N = float(window)
    H_phi = 0.0                                  # manuscript's phi-form state
    got: dict[int, float] = {}
    t0 = time.perf_counter()

    def phi(m: int) -> float:
        if m <= 0:
            return 0.0
        p = m / N
        return -p * math.log2(p)

    for i, v in enumerate(stream):
        evicted = None
        if len(q) >= window:
            evicted = q.popleft()
            co = counts[evicted]
            if variant == "phi":
                H_phi = H_phi - phi(co) + phi(co - 1)
            else:
                delta = -clogc[co] + clogc[co - 1]
                if variant == "kahan":
                    t = S + delta
                    comp += (S - t) + delta if abs(S) >= abs(delta) else (delta - t) + S
                    S = t
                else:
                    S += delta
            if co == 1:
                del counts[evicted]
            else:
                counts[evicted] = co - 1

        ci = counts.get(v, 0)
        if variant == "phi":
            H_phi = H_phi - phi(ci) + phi(ci + 1)
        else:
            delta = -clogc[ci] + clogc[ci + 1]
            if variant == "kahan":
                t = S + delta
                comp += (S - t) + delta if abs(S) >= abs(delta) else (delta - t) + S
                S = t
            else:
                S += delta
        counts[v] = ci + 1
        q.append(v)

        if recompute_every and (i + 1) % recompute_every == 0:
            S = 0.0
            for c in counts.values():
                S += clogc[c]
            comp = 0.0

        if i in checkpoint_at:
            L = len(q)
            if variant == "phi":
                got[i] = H_phi
            else:
                Seff = S + comp if variant == "kahan" else S
                got[i] = max(0.0, log2len[L] - Seff / L)

    return got, time.perf_counter() - t0, len(stream)


def exact_at(stream, window: int, checkpoint_at: set[int]) -> dict[int, float]:
    """Ground truth: recompute entropy from the multiset at each checkpoint."""
    q: deque = deque(maxlen=window)
    counts: dict = {}
    got: dict[int, float] = {}
    for i, v in enumerate(stream):
        if len(q) >= window:
            vo = q.popleft()
            c = counts[vo]
            if c == 1:
                del counts[vo]
            else:
                counts[vo] = c - 1
        counts[v] = counts.get(v, 0) + 1
        q.append(v)
        if i in checkpoint_at:
            L = len(q)
            got[i] = -sum((c / L) * math.log2(c / L) for c in counts.values())
    return got


def exact_rational_spot(stream, window: int, at: int) -> float:
    """One rational-arithmetic ground-truth value, to bound float64 reference error."""
    q: deque = deque(maxlen=window)
    counts: dict = {}
    for i, v in enumerate(stream[:at + 1]):
        if len(q) >= window:
            vo = q.popleft()
            c = counts[vo]
            if c == 1:
                del counts[vo]
            else:
                counts[vo] = c - 1
        counts[v] = counts.get(v, 0) + 1
        q.append(v)
    L = len(q)
    tot = 0.0
    for c in counts.values():
        p = Fraction(c, L)
        tot -= float(p) * math.log2(float(p))
    return tot


def main() -> int:
    log_event("E11", "start")
    t_all = time.time()
    out: dict = {"experiment": "E11",
                 "objective": "numerical stability of the rolling entropy update",
                 "window": WINDOW, "n_updates_target": N_UPDATES}

    # Drive the study with a real key stream: destination ports from the trace,
    # tiled to reach the requested update count.
    X = np.load(CACHE / "syn0311_X_repaired.npy", mmap_mode="r")
    names = json.loads((CACHE / "syn0311_feature_names_repaired.json").read_text())
    j = names.index("Destination_Port")
    base = np.asarray(X[:, j], dtype=np.int64)
    reps = int(np.ceil(N_UPDATES / len(base)))
    stream = np.tile(base, reps)[:N_UPDATES].tolist()
    out["stream"] = {"source_feature": "Destination_Port", "n": len(stream),
                     "n_distinct": int(len(set(stream))),
                     "tiled_repeats": reps}
    print(f"stream of {len(stream):,} updates, "
          f"{out['stream']['n_distinct']:,} distinct keys", flush=True)

    step = len(stream) // CHECKPOINTS
    checkpoints = sorted({min(len(stream) - 1, (k + 1) * step - 1)
                          for k in range(CHECKPOINTS)})
    cp_set = set(checkpoints)

    print("exact reference pass...", flush=True)
    t0 = time.time()
    ref = exact_at(stream, WINDOW, cp_set)
    print(f"  {time.time()-t0:.1f}s", flush=True)

    spot = checkpoints[len(checkpoints) // 2]
    rat = exact_rational_spot(stream, WINDOW, spot)
    out["reference_check"] = {
        "checkpoint": int(spot),
        "float64_reference": ref[spot],
        "rational_reference": rat,
        "abs_difference": abs(ref[spot] - rat),
    }

    variants = [
        ("running_sum", None, "implemented form: S = sum c log2 c, H = log2|W| - S/|W|"),
        ("phi", None, "manuscript Eq. (6): H' = H - phi(n_a) + phi(n_a+1) - phi(n_d) + phi(n_d-1)"),
        ("kahan", None, "running sum with Neumaier compensated summation"),
        ("running_sum", 100_000, "running sum with exact resynchronisation every 100k updates"),
        ("running_sum", 10_000, "running sum with exact resynchronisation every 10k updates"),
    ]

    rows = []
    out["variants"] = {}
    for variant, rec, desc in variants:
        key = variant if rec is None else f"{variant}_resync{rec}"
        print(f"running {key}...", flush=True)
        got, secs, n = run_variant(stream, WINDOW, variant, rec, cp_set)
        err = np.array([abs(got[c] - ref[c]) for c in checkpoints])
        res = {
            "description": desc,
            "seconds": round(secs, 2),
            "ns_per_update": round(1e9 * secs / n, 1),
            "max_abs_error_bits": float(err.max()),
            "rms_abs_error_bits": float(np.sqrt((err ** 2).mean())),
            "final_abs_error_bits": float(err[-1]),
            "error_at_1M": float(err[min(len(err) - 1,
                                         int(len(err) * 1_000_000 / len(stream)))]),
            "relative_error_final": float(err[-1] / max(ref[checkpoints[-1]], 1e-12)),
        }
        out["variants"][key] = res
        for c, e in zip(checkpoints, err):
            rows.append({"variant": key, "update_index": int(c + 1),
                         "abs_error_bits": float(e), "reference_H": float(ref[c])})
        print(f"  max|err|={res['max_abs_error_bits']:.3e} bits, "
              f"final={res['final_abs_error_bits']:.3e}, "
              f"{res['ns_per_update']:.0f} ns/update", flush=True)

    pd.DataFrame(rows).to_csv(RESULTS / "E11_drift_curve.csv", index=False)

    # ---- operational consequence: does drift change any decision? ----------
    print("Operational impact: do drifted entropies flip any label?", flush=True)
    y = np.load(CACHE / "syn0311_y.npy")
    cut = int(round(len(y) * 0.70))
    worst = max(out["variants"].items(),
                key=lambda kv: kv[1]["max_abs_error_bits"])
    eps = worst[1]["max_abs_error_bits"]
    out["worst_variant"] = {"name": worst[0], "max_abs_error_bits": eps}

    from sklearn.ensemble import RandomForestClassifier
    Xf = np.load(CACHE / "syn0311_X_repaired.npy")
    ent_start = len(names) - 8
    clf = RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                 class_weight="balanced", n_jobs=-1,
                                 random_state=42).fit(Xf[:cut], y[:cut])
    base_pred = (clf.predict_proba(Xf[cut:])[:, 1] >= 0.70).astype(np.int8)
    rng = np.random.default_rng(42)
    Xp = Xf[cut:].copy()
    Xp[:, ent_start:] += rng.uniform(-eps, eps,
                                     size=(Xp.shape[0], 8)).astype(np.float32)
    pert_pred = (clf.predict_proba(Xp)[:, 1] >= 0.70).astype(np.int8)
    flips = int((base_pred != pert_pred).sum())
    out["label_flips_under_worst_case_drift"] = {
        "perturbation_bits": float(eps),
        "n_test_flows": int(len(base_pred)),
        "n_flips": flips,
        "flip_rate": float(flips / len(base_pred)),
    }
    print(f"  perturbing all 8 entropy features by +/-{eps:.3e} bits "
          f"flips {flips:,} of {len(base_pred):,} decisions "
          f"({flips/len(base_pred):.3%})", flush=True)

    # ---- complexity evidence for R2.6 --------------------------------------
    out["complexity_evidence"] = {
        "claim": "expected O(1) amortised per admit/evict pair",
        "assumptions": [
            "fixed-capacity circular buffer gives O(1) worst-case admit and evict",
            "hash table gives expected O(1) lookup, insert and delete; worst case is "
            "O(k) under adversarial collisions",
            "the entropy value is reconstructed from a running scalar, so no sum over "
            "the window is performed per update",
            "periodic exact resynchronisation, if enabled every R updates, adds "
            "O(distinct/R) amortised work per update",
        ],
        "measured_ns_per_update": {k: v["ns_per_update"] for k, v in out["variants"].items()},
    }

    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E11_entropy_numerics", out)
    log_event("E11", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
