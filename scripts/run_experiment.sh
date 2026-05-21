#!/usr/bin/env bash
# run_experiment.sh — Reproduce all key results from the paper.
# Requires: trained model artifacts in model/artifacts/
# Usage:    bash scripts/run_experiment.sh [--synthetic] [--data-dir data/raw]

set -euo pipefail

SYNTHETIC=0
DATA_DIR="data/raw"
ARTIFACTS_DIR="model/artifacts"

for arg in "$@"; do
  case $arg in
    --synthetic) SYNTHETIC=1 ;;
    --data-dir=*) DATA_DIR="${arg#*=}" ;;
  esac
done

echo "================================================================"
echo "XAI-SDN: Full Experiment Reproduction"
echo "================================================================"
echo "Mode:         $([ $SYNTHETIC -eq 1 ] && echo 'Synthetic' || echo 'Real data')"
echo "Data dir:     $DATA_DIR"
echo "Artifacts:    $ARTIFACTS_DIR"
echo "================================================================"

mkdir -p "$ARTIFACTS_DIR" logs data/synthetic

# Step 1: Generate / preprocess data
if [ $SYNTHETIC -eq 1 ]; then
  echo; echo "[1/6] Generating synthetic dataset..."
  python scripts/generate_synthetic_data.py --n-samples 20000
  TRAIN_FLAG="--use-synthetic"
else
  echo; echo "[1/6] Preprocessing CIC-DDoS2019 data..."
  python scripts/preprocess_data.py --data-dir "$DATA_DIR"
  TRAIN_FLAG="--data-dir $DATA_DIR"
fi

# Step 2: Train model
echo; echo "[2/6] Training Random Forest (200 trees)..."
python model/train.py --config configs/model_config.yaml $TRAIN_FLAG \
  --output-dir "$ARTIFACTS_DIR"

# Step 3: Evaluate
echo; echo "[3/6] Evaluating model (using saved test split)..."
if [ $SYNTHETIC -eq 1 ]; then
  python model/evaluate.py \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --output-dir "$ARTIFACTS_DIR"
else
  python model/evaluate.py \
    --artifacts-dir "$ARTIFACTS_DIR" \
    --data-dir "$DATA_DIR" \
    --run-shap \
    --output-dir "$ARTIFACTS_DIR"
fi

# Step 4: Baseline comparison
echo; echo "[4/6] Running baseline comparison..."
python model/baselines.py \
  $([ $SYNTHETIC -eq 1 ] && echo '--use-synthetic' || echo "--data-dir $DATA_DIR") \
  --skip-deep \
  --output "$ARTIFACTS_DIR/baseline_results.json"

# Step 5: Ablation study
echo; echo "[5/6] Running ablation study..."
python model/ablation.py \
  $([ $SYNTHETIC -eq 1 ] && echo '--use-synthetic' || echo "--data-dir $DATA_DIR") \
  --output "$ARTIFACTS_DIR/ablation_results.json"

# Step 6: Smoke test
echo; echo "[6/6] Running smoke test..."
python scripts/smoke_test.py --skip-api

echo
echo "================================================================"
echo "Experiment complete. Artifacts saved to: $ARTIFACTS_DIR/"
echo "----------------------------------------------------------------"
echo "  metrics.json                ← Table 2 (accuracy, F1, FPR, AUC)"
echo "  evaluation_results.json     ← Full evaluation metrics"
echo "  baseline_results.json       ← Table 3 (baseline comparison)"
echo "  ablation_results.json       ← Table 4 (ablation study)"
echo "  shap_global_importance.csv  ← Figure 4 (SHAP importance)"
echo "  shap_global_importance.png  ← Figure 4 (SHAP plot)"
echo "  confusion_matrix.csv        ← Figure 3 (confusion matrix)"
echo "================================================================"
