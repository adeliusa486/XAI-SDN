"""Single-sample inference helper.

scikit-learn ensembles fitted with n_jobs=-1 keep that setting at predict time,
and joblib's parallel dispatch costs roughly 25 ms per call regardless of how
much work there is to do. On a batch that overhead is amortised to nothing. On
a single sample it IS the measurement, and it produced a p50 of 27.4 ms per
flow for a Random Forest against 0.19 ms for an SVM, which inverts the true
ordering by two orders of magnitude.

Per-flow latency is therefore measured with the estimator forced to serial
execution, which is also what a controller serving one flow at a time would do.
Batch throughput continues to be measured with the fitted parallel setting,
because a controller processing a poll batch genuinely can use every core.
Both numbers are reported, and the paper says which is which.
"""
from __future__ import annotations

from contextlib import contextmanager


def _walk(est):
    """Yield the estimator and any nested estimators that expose n_jobs."""
    yield est
    for attr in ("steps", "estimators_", "named_steps"):
        sub = getattr(est, attr, None)
        if sub is None:
            continue
        items = sub.values() if hasattr(sub, "values") else sub
        for it in items:
            obj = it[1] if isinstance(it, tuple) and len(it) == 2 else it
            if hasattr(obj, "get_params") or hasattr(obj, "n_jobs"):
                yield obj


@contextmanager
def serial_inference(est):
    """Temporarily force n_jobs=1 on an estimator and anything nested in it."""
    saved = []
    for obj in _walk(est):
        if hasattr(obj, "n_jobs"):
            saved.append((obj, obj.n_jobs))
            try:
                obj.n_jobs = 1
            except Exception:
                saved.pop()
    try:
        yield est
    finally:
        for obj, val in saved:
            try:
                obj.n_jobs = val
            except Exception:
                pass


def force_serial(est):
    """Permanently force serial execution, for a model served one flow at a time."""
    for obj in _walk(est):
        if hasattr(obj, "n_jobs"):
            try:
                obj.n_jobs = 1
            except Exception:
                pass
    return est
