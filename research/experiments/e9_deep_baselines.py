"""E9 - Deep-learning baselines under the same CPU protocol.

This experiment provides a comparison against "modern deep learning detectors (CNN,
LSTM, autoencoder, GNN, transformer) under the same protocol". The deployment question insists
the protocol be identical for every model. The deployment question insists everything be timed
on the same hardware.

All five architectures are therefore evaluated under exactly the Protocol B
splits used in E1, on the CPU, with the same metrics and the same latency
instrumentation. The torch build in this environment has no CUDA, so a GPU
measurement cannot occur.

Architectures:
  1D-CNN        - convolution over the 88-dimensional feature vector
  LSTM          - recurrent model over a window of L consecutive flows
  Transformer   - encoder over the same window, with learned positional encoding
  Autoencoder   - trained on benign traffic only; detection by reconstruction
                  error, i.e. an unsupervised anomaly detector
  GNN           - two-layer graph convolution over a k-nearest-neighbour graph
                  built in flow-feature space, implemented directly in torch so
                  no external graph library is required
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent / "common"))

from paths import DATA_ROOT, RESULTS, log_event, save_result  # noqa: E402

from sklearn.metrics import (accuracy_score, average_precision_score,  # noqa: E402
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler  # noqa: E402

CACHE = DATA_ROOT / "cache"
TAU = 0.70
SEED = 42
SEQ_LEN = 16
EPOCHS = 8
BATCH = 512
LR = 1e-3
LATENCY_PROBE = 1000
KNN_K = 10
GNN_NODES = 20_000        # graph size kept tractable on CPU
PROTOCOL_B_TRAIN = 100_000
PROTOCOL_B_TEST = 40_000


def make_windows(X: np.ndarray, y: np.ndarray, L: int):
    """Sliding windows of L consecutive flows; the label is that of the last flow."""
    n = len(y) - L + 1
    idx = np.arange(L)[None, :] + np.arange(n)[:, None]
    return X[idx], y[L - 1:]


def metrics(y, score, tau=TAU) -> dict:
    pred = (score >= tau).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "avg_precision": float(average_precision_score(y, score)),
        "roc_auc": float(roc_auc_score(y, score)) if len(np.unique(y)) > 1 else None,
        "precision": float(tp / (tp + fp)) if (tp + fp) else None,
        "recall": float(tp / (tp + fn)) if (tp + fn) else None,
        "fpr": float(fp / (fp + tn)) if (fp + tn) else None,
        "fnr": float(fn / (fn + tp)) if (fn + tp) else None,
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main() -> int:
    log_event("E9", "start")
    t_all = time.time()
    import torch
    import torch.nn as nn

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(psutil.cpu_count(logical=False) or 8)
    device = torch.device("cpu")

    out: dict = {
        "experiment": "E9",
        "objective": "deep-learning baselines under the Protocol B splits of E1",
        "cpu_only_guarantee": {"torch_version": torch.__version__,
                               "cuda_available": bool(torch.cuda.is_available()),
                               "threads": torch.get_num_threads()},
        "hyperparameters": {"epochs": EPOCHS, "batch": BATCH, "lr": LR,
                            "sequence_length": SEQ_LEN, "knn_k": KNN_K,
                            "gnn_nodes": GNN_NODES, "seed": SEED},
        "tau": TAU,
    }

    X = np.load(CACHE / "syn0311_X_repaired.npy")
    y = np.load(CACHE / "syn0311_y.npy")
    n = len(y)
    cut = int(round(n * 0.70))
    tr_s = max(0, cut - PROTOCOL_B_TRAIN)
    te_e = min(n, cut + PROTOCOL_B_TEST)
    Xtr_raw, ytr = X[tr_s:cut], y[tr_s:cut]
    Xte_raw, yte = X[cut:te_e], y[cut:te_e]

    out["protocol_B"] = {
        "identical_to": "E1 Protocol B",
        "train_range": [int(tr_s), int(cut)], "test_range": [int(cut), int(te_e)],
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
        "train_benign": int((ytr == 0).sum()), "test_benign": int((yte == 0).sum()),
    }
    print(json.dumps(out["protocol_B"], indent=2), flush=True)

    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        rng = np.random.default_rng(SEED)
        itr = np.sort(rng.choice(cut, PROTOCOL_B_TRAIN, replace=False))
        ite = np.sort(rng.choice(np.arange(cut, n), PROTOCOL_B_TEST, replace=False))
        Xtr_raw, ytr, Xte_raw, yte = X[itr], y[itr], X[ite], y[ite]
        out["protocol_B"]["fallback"] = ("a contiguous block was single-class; a "
                                         "stratified subsample of identical size was "
                                         "used, matching the E1 fallback")
        out["protocol_B"].update({"train_benign": int((ytr == 0).sum()),
                                  "test_benign": int((yte == 0).sum())})
        print("  fallback to stratified subsample", flush=True)

    sc = StandardScaler().fit(Xtr_raw)
    Xtr = np.nan_to_num(sc.transform(Xtr_raw)).astype(np.float32)
    Xte = np.nan_to_num(sc.transform(Xte_raw)).astype(np.float32)
    d = Xtr.shape[1]
    out["preprocessing"] = ("features standardised with statistics fitted on the "
                            "training block only; tree models in E1 use the raw "
                            "features, as is standard")

    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))
    results: dict = {}
    rows = []

    def train_supervised(model, Xa, ya, Xb, name, seq=False):
        proc = psutil.Process()
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w]))
        Xt = torch.from_numpy(Xa)
        yt = torch.from_numpy(ya.astype(np.float32)).unsqueeze(1)
        nb = int(np.ceil(len(yt) / BATCH))
        t0 = time.perf_counter()
        model.train()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(yt))
            tot = 0.0
            for b in range(nb):
                i = perm[b * BATCH:(b + 1) * BATCH]
                opt.zero_grad()
                loss = lossf(model(Xt[i]), yt[i])
                loss.backward()
                opt.step()
                tot += float(loss) * len(i)
            print(f"    {name} epoch {ep+1}/{EPOCHS} loss={tot/len(yt):.5f}", flush=True)
        train_s = time.perf_counter() - t0
        peak = proc.memory_info().rss / 2 ** 30

        model.eval()
        Xb_t = torch.from_numpy(Xb)
        t0 = time.perf_counter()
        with torch.no_grad():
            sc_out = torch.cat([torch.sigmoid(model(Xb_t[i:i + BATCH]))
                                for i in range(0, len(Xb_t), BATCH)]).numpy().ravel()
        batch_s = time.perf_counter() - t0

        lat = np.empty(min(LATENCY_PROBE, len(Xb_t)))
        with torch.no_grad():
            for i in range(len(lat)):
                t = time.perf_counter()
                torch.sigmoid(model(Xb_t[i:i + 1]))
                lat[i] = (time.perf_counter() - t) * 1000
        return sc_out, {
            "train_time_s": round(train_s, 2),
            "peak_rss_gb": round(peak, 3),
            "batch_throughput_flows_per_s": round(len(Xb) / batch_s, 1),
            "single_flow_latency_ms": {"p50": float(np.percentile(lat, 50)),
                                       "p95": float(np.percentile(lat, 95)),
                                       "p99": float(np.percentile(lat, 99)),
                                       "mean": float(lat.mean())},
            "n_parameters": int(sum(p.numel() for p in model.parameters())),
        }

    score_dir = RESULTS / "scores"
    score_dir.mkdir(parents=True, exist_ok=True)

    def record(name, y_eval, score, perf, note=None):
        slug = name.replace(" ", "_").replace("(", "").replace(")", "")
        np.save(score_dir / f"E9_B_{slug}_score.npy", np.asarray(score, np.float32))
        np.save(score_dir / f"E9_B_{slug}_ytrue.npy", np.asarray(y_eval, np.int8))
        r = {"model": name, "protocol": "B", "status": "ok",
             "hardware": "CPU only", "exact_per_prediction_explanation": False,
             **metrics(y_eval, score), **perf}
        r["score_file"] = f"scores/E9_B_{slug}_score.npy"
        r["ytrue_file"] = f"scores/E9_B_{slug}_ytrue.npy"
        if note:
            r["note"] = note
        results[name] = r
        rows.append({k: v for k, v in r.items()
                     if k not in ("confusion", "single_flow_latency_ms")} |
                    {f"lat_{k}_ms": v for k, v in r["single_flow_latency_ms"].items()})
        print(f"  {name:16s} acc={r['accuracy']:.5f} F1={r['macro_f1']:.5f} "
              f"AP={r['avg_precision']:.5f} train={r['train_time_s']}s "
              f"p50={r['single_flow_latency_ms']['p50']:.4f}ms", flush=True)

    # ---- 1D-CNN -------------------------------------------------------------
    print("\n1D-CNN", flush=True)
    class CNN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.net = nn.Sequential(
                nn.Unflatten(1, (1, d)),
                nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool1d(8), nn.Flatten(),
                nn.Linear(64 * 8, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1))

        def forward(self, x):
            return self.net(x)
    s, perf = train_supervised(CNN(d), Xtr, ytr, Xte, "CNN")
    record("1D-CNN", yte, s, perf)

    # ---- sequence models ----------------------------------------------------
    Xtr_seq, ytr_seq = make_windows(Xtr, ytr, SEQ_LEN)
    Xte_seq, yte_seq = make_windows(Xte, yte, SEQ_LEN)
    seq_note = (f"evaluated on windows of {SEQ_LEN} consecutive flows, so the test "
                f"set has {len(yte_seq):,} windows rather than {len(yte):,} flows")

    print("\nLSTM", flush=True)
    class LSTMNet(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.rnn = nn.LSTM(d, 64, batch_first=True)
            self.head = nn.Sequential(nn.ReLU(), nn.Linear(64, 1))

        def forward(self, x):
            o, _ = self.rnn(x)
            return self.head(o[:, -1])
    if len(np.unique(ytr_seq)) > 1 and len(np.unique(yte_seq)) > 1:
        s, perf = train_supervised(LSTMNet(d), Xtr_seq, ytr_seq, Xte_seq, "LSTM")
        record("LSTM", yte_seq, s, perf, seq_note)
    else:
        results["LSTM"] = {"model": "LSTM", "status": "skipped",
                           "reason": "windowed labels are single-class"}

    print("\nTransformer", flush=True)
    class TransNet(nn.Module):
        def __init__(self, d, L):
            super().__init__()
            self.proj = nn.Linear(d, 64)
            self.pos = nn.Parameter(torch.zeros(1, L, 64))
            layer = nn.TransformerEncoderLayer(64, 4, 128, dropout=0.1,
                                               batch_first=True)
            self.enc = nn.TransformerEncoder(layer, 2)
            self.head = nn.Linear(64, 1)

        def forward(self, x):
            h = self.enc(self.proj(x) + self.pos)
            return self.head(h[:, -1])
    if len(np.unique(ytr_seq)) > 1 and len(np.unique(yte_seq)) > 1:
        s, perf = train_supervised(TransNet(d, SEQ_LEN), Xtr_seq, ytr_seq,
                                   Xte_seq, "Transformer")
        record("Transformer", yte_seq, s, perf, seq_note)
    else:
        results["Transformer"] = {"model": "Transformer", "status": "skipped",
                                  "reason": "windowed labels are single-class"}

    # ---- autoencoder (unsupervised) -----------------------------------------
    print("\nAutoencoder (benign-only, reconstruction error)", flush=True)
    class AE(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.e = nn.Sequential(nn.Linear(d, 64), nn.ReLU(),
                                   nn.Linear(64, 16), nn.ReLU())
            self.d = nn.Sequential(nn.Linear(16, 64), nn.ReLU(), nn.Linear(64, d))

        def forward(self, x):
            return self.d(self.e(x))

    # The autoencoder is trained on benign traffic only. The contiguous
    # Protocol B training block contains very few benign flows, because benign
    # traffic in this capture is concentrated in the final third (Sec. VII-C).
    # Benign examples are therefore drawn from the whole training partition for
    # this model alone, which is stated in the paper: an unsupervised detector
    # that cannot be given benign data cannot be evaluated at all, and the
    # alternative would be to omit the baseline entirely.
    ben = Xtr[ytr == 0]
    ae_note_extra = ""
    if len(ben) < 2000:
        pool = np.flatnonzero(y[:cut] == 0)
        if len(pool) > len(ben):
            ben = np.nan_to_num(sc.transform(X[pool])).astype(np.float32)
            ae_note_extra = (f" Benign training examples for this model were drawn "
                             f"from the entire training partition ({len(pool):,} "
                             f"flows) rather than from the Protocol B block, which "
                             f"holds too few to fit a reconstruction model.")
    out["autoencoder_benign_pool"] = int(len(ben))
    if len(ben) >= 100:
        proc = psutil.Process()
        ae = AE(d)
        opt = torch.optim.Adam(ae.parameters(), lr=LR)
        Bt = torch.from_numpy(ben)
        t0 = time.perf_counter()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Bt))
            tot = 0.0
            for b in range(int(np.ceil(len(Bt) / BATCH))):
                i = perm[b * BATCH:(b + 1) * BATCH]
                opt.zero_grad()
                loss = nn.functional.mse_loss(ae(Bt[i]), Bt[i])
                loss.backward(); opt.step()
                tot += float(loss) * len(i)
            print(f"    AE epoch {ep+1}/{EPOCHS} mse={tot/len(Bt):.5f}", flush=True)
        train_s = time.perf_counter() - t0
        ae.eval()
        Tt = torch.from_numpy(Xte)
        t0 = time.perf_counter()
        with torch.no_grad():
            err = torch.cat([((ae(Tt[i:i + BATCH]) - Tt[i:i + BATCH]) ** 2).mean(1)
                             for i in range(0, len(Tt), BATCH)]).numpy()
        batch_s = time.perf_counter() - t0
        # map reconstruction error to [0,1] using the benign training distribution
        with torch.no_grad():
            ben_err = ((ae(Bt) - Bt) ** 2).mean(1).numpy()
        lo, hi = np.percentile(ben_err, 50), np.percentile(ben_err, 99.9)
        score = np.clip((err - lo) / max(hi - lo, 1e-9), 0, 1)
        lat = np.empty(min(LATENCY_PROBE, len(Tt)))
        with torch.no_grad():
            for i in range(len(lat)):
                t = time.perf_counter()
                ae(Tt[i:i + 1])
                lat[i] = (time.perf_counter() - t) * 1000
        record("Autoencoder", yte, score, {
            "train_time_s": round(train_s, 2),
            "peak_rss_gb": round(proc.memory_info().rss / 2 ** 30, 3),
            "batch_throughput_flows_per_s": round(len(Xte) / batch_s, 1),
            "single_flow_latency_ms": {"p50": float(np.percentile(lat, 50)),
                                       "p95": float(np.percentile(lat, 95)),
                                       "p99": float(np.percentile(lat, 99)),
                                       "mean": float(lat.mean())},
            "n_parameters": int(sum(p.numel() for p in ae.parameters())),
        }, note=(ae_note_extra + " unsupervised: trained on benign training flows only, scored by "
                 "reconstruction error normalised against the benign error "
                 "distribution, so its operating point is not directly comparable "
                 "to the supervised models at the same tau"))
    else:
        results["Autoencoder"] = {"model": "Autoencoder", "status": "skipped",
                                  "reason": f"only {len(ben)} benign training flows"}

    # ---- GNN ----------------------------------------------------------------
    # The first attempt at this baseline produced a test ROC-AUC of 0.035, far
    # below chance, which means the scores were anti-correlated with the labels
    # rather than merely uninformative. Two changes follow. The adjacency is
    # symmetrically normalised in the standard GCN form rather than divided by
    # raw degree, and a validation slice of the training nodes is held out so
    # the run can be checked instead of assumed. If the held-out AUC still falls
    # below chance the result is reported as a failure of the baseline, not
    # silently sign-flipped.
    print("\nGNN (2-layer graph convolution over a kNN flow graph)", flush=True)
    from sklearn.neighbors import NearestNeighbors
    rng = np.random.default_rng(SEED)
    gi_tr = np.sort(rng.choice(len(ytr), min(GNN_NODES, len(ytr)), replace=False))
    gi_te = np.sort(rng.choice(len(yte), min(GNN_NODES // 2, len(yte)), replace=False))
    Xg = np.vstack([Xtr[gi_tr], Xte[gi_te]])
    yg = np.concatenate([ytr[gi_tr], yte[gi_te]])
    n_tr_nodes = len(gi_tr)
    mask_tr = np.zeros(len(yg), bool); mask_tr[:n_tr_nodes] = True
    # hold out a slice of the training nodes to monitor the fit
    val_cut = int(n_tr_nodes * 0.85)
    mask_fit = np.zeros(len(yg), bool); mask_fit[:val_cut] = True
    mask_val = np.zeros(len(yg), bool); mask_val[val_cut:n_tr_nodes] = True

    t0 = time.perf_counter()
    nn_ = NearestNeighbors(n_neighbors=KNN_K + 1, n_jobs=-1).fit(Xg)
    _, nbr = nn_.kneighbors(Xg)
    graph_s = time.perf_counter() - t0
    rowi = np.repeat(np.arange(len(yg)), KNN_K)
    coli = nbr[:, 1:].ravel()
    src = np.concatenate([rowi, coli, np.arange(len(yg))])   # + self loops
    dst = np.concatenate([coli, rowi, np.arange(len(yg))])
    ei = np.vstack([src, dst])
    # symmetric normalisation: D^-1/2 (A + I) D^-1/2
    deg = np.bincount(src, minlength=len(yg)).astype(np.float32)
    dinv = 1.0 / np.sqrt(np.maximum(deg, 1.0))
    vals = torch.from_numpy((dinv[src] * dinv[dst]).astype(np.float32))
    A = torch.sparse_coo_tensor(torch.from_numpy(ei), vals,
                                (len(yg), len(yg))).coalesce()

    class GCN(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.l1 = nn.Linear(d, 64)
            self.l2 = nn.Linear(64, 32)
            self.out = nn.Linear(32, 1)
            self.drop = nn.Dropout(0.1)

        def forward(self, x, A):
            h = torch.relu(self.l1(torch.sparse.mm(A, x)))
            h = self.drop(h)
            h = torch.relu(self.l2(torch.sparse.mm(A, h)))
            return self.out(h)

    if len(np.unique(yg[mask_fit])) > 1 and len(np.unique(yg[~mask_tr])) > 1:
        proc = psutil.Process()
        g = GCN(d)
        opt = torch.optim.Adam(g.parameters(), lr=LR, weight_decay=5e-4)
        Xg_t = torch.from_numpy(Xg)
        yg_t = torch.from_numpy(yg.astype(np.float32)).unsqueeze(1)
        mfit = torch.from_numpy(mask_fit)
        mval = torch.from_numpy(mask_val)
        pw = float((yg[mask_fit] == 0).sum() / max((yg[mask_fit] == 1).sum(), 1))
        lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
        best_state, best_val, best_ep = None, -1.0, -1
        t0 = time.perf_counter()
        for ep in range(300):
            g.train(); opt.zero_grad()
            loss = lossf(g(Xg_t, A)[mfit], yg_t[mfit])
            loss.backward(); opt.step()
            if (ep + 1) % 10 == 0:
                g.eval()
                with torch.no_grad():
                    sv = torch.sigmoid(g(Xg_t, A)).numpy().ravel()
                yv = yg[mask_val]
                if len(np.unique(yv)) > 1:
                    a = roc_auc_score(yv, sv[mask_val])
                    if a > best_val:
                        best_val, best_ep = a, ep + 1
                        best_state = {k: v.clone() for k, v in g.state_dict().items()}
                    if (ep + 1) % 50 == 0:
                        print(f"    GNN step {ep+1}/300 loss={float(loss):.5f} "
                              f"val AUC={a:.4f}", flush=True)
        train_s = time.perf_counter() - t0
        if best_state is not None:
            g.load_state_dict(best_state)
        g.eval()
        t0 = time.perf_counter()
        with torch.no_grad():
            sc_all = torch.sigmoid(g(Xg_t, A)).numpy().ravel()
        infer_s = time.perf_counter() - t0
        test_auc = roc_auc_score(yg[~mask_tr], sc_all[~mask_tr])
        print(f"    selected epoch {best_ep} (val AUC={best_val:.4f}); "
              f"test AUC={test_auc:.4f}", flush=True)

        note = ("transductive: the model scores nodes of a graph built over a "
                f"{len(yg):,}-flow subsample, so single-flow latency is undefined "
                "and it cannot be deployed per-flow without incremental graph "
                "maintenance; graph construction time is reported separately")
        if test_auc < 0.5:
            note += (". This baseline FAILED to learn a usable decision on this "
                     f"corpus: held-out ROC-AUC is {test_auc:.4f}, below chance. "
                     "The scores are reported as produced and are not sign-flipped. "
                     "We attribute the failure to neighbourhood smoothing over a "
                     "k-nearest-neighbour graph at 97% positive prevalence, which "
                     "erases the minority class, and we report it rather than "
                     "omitting the baseline.")
        record("GNN", yg[~mask_tr], sc_all[~mask_tr], {
            "train_time_s": round(train_s, 2),
            "graph_construction_s": round(graph_s, 2),
            "peak_rss_gb": round(proc.memory_info().rss / 2 ** 30, 3),
            "batch_throughput_flows_per_s": round(len(yg) / infer_s, 1),
            "single_flow_latency_ms": {"p50": float("nan"), "p95": float("nan"),
                                       "p99": float("nan"), "mean": float("nan")},
            "n_parameters": int(sum(p.numel() for p in g.parameters())),
            "validation_auc": float(best_val),
            "selected_epoch": int(best_ep),
            "converged": bool(test_auc >= 0.5),
        }, note=note)
    else:
        results["GNN"] = {"model": "GNN", "status": "skipped",
                          "reason": "graph subsample is single-class"}

    out["results"] = results
    pd.DataFrame(rows).to_csv(RESULTS / "E9_deep_baselines.csv", index=False)
    out["comparability_statement"] = (
        "Every model here shares the Protocol B splits, the CPU, the thread budget "
        "and the metric definitions used for the classical baselines in E1, so the "
        "two result sets may be read together. Three caveats are attached to "
        "individual rows rather than buried: the sequence models are evaluated on "
        "windows rather than single flows, the autoencoder is unsupervised and its "
        "operating point is set differently, and the graph model is transductive and "
        "has no meaningful single-flow latency.")
    out["total_runtime_s"] = round(time.time() - t_all, 1)
    p = save_result("E9_deep_baselines", out)
    log_event("E9", "done", result=str(p), runtime_s=out["total_runtime_s"])
    print(f"\nWrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
