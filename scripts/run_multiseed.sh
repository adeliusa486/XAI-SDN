#!/usr/bin/env bash
# run_multiseed.sh — Multi-seed experiment runner for XAI-SDN.
# Produces mean ± std metrics and statistical significance tests.
#
# Usage:
#   bash scripts/run_multiseed.sh --synthetic
#   bash scripts/run_multiseed.sh --data-dir data/raw --seeds "42 123 456 789 1024"

set -euo pipefail

SEEDS="42 123 456 789 1024"
DATA_FLAG="--use-synthetic"
DATA_DIR="data/raw"
OUTPUT_DIR="model/artifacts/multiseed"

for arg in "$@"; do
  case $arg in
    --synthetic)       DATA_FLAG="--use-synthetic" ;;
    --data-dir=*)      DATA_DIR="${arg#*=}"; DATA_FLAG="--data-dir $DATA_DIR" ;;
    --seeds=*)         SEEDS="${arg#*=}" ;;
    --output-dir=*)    OUTPUT_DIR="${arg#*=}" ;;
  esac
done

mkdir -p "$OUTPUT_DIR"
RESULTS_FILE="$OUTPUT_DIR/all_seeds_results.jsonl"
> "$RESULTS_FILE"  # clear

echo "Seeds: $SEEDS"
echo "Data:  $DATA_FLAG"

for SEED in $SEEDS; do
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "Running seed=$SEED..."
  SEED_DIR="$OUTPUT_DIR/seed_$SEED"
  mkdir -p "$SEED_DIR"

  python model/train.py \
    --config configs/model_config.yaml \
    $DATA_FLAG \
    --random-state "$SEED" \
    --output-dir "$SEED_DIR"

  python model/evaluate.py \
    --artifacts-dir "$SEED_DIR" \
    --output-dir "$SEED_DIR"

  # Append seed result to JSONL
  python -c "
import json, sys
m = json.load(open('$SEED_DIR/metrics.json'))
e = json.load(open('$SEED_DIR/evaluation_results.json'))
result = {
    'seed': $SEED,
    'accuracy': m['accuracy'],
    'macro_f1': m['macro_f1'],
    'fpr': e.get('fpr', None),
    'auc': e.get('auc', None),
    'latency_rf_ms': e.get('latency_rf_ms', None),
}
print(json.dumps(result))
" >> "$RESULTS_FILE"

done

# Compute aggregate statistics
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Computing aggregate statistics..."

python - << PYEOF
import json, sys, math
from pathlib import Path

results_path = Path("$RESULTS_FILE")
results = [json.loads(line) for line in results_path.read_text().strip().split("\n")]

def stats(values):
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean)**2 for v in values) / (n - 1)) if n > 1 else 0
    return mean, std

for metric in ["accuracy", "macro_f1", "fpr", "auc"]:
    vals = [r[metric] for r in results if r.get(metric) is not None]
    if vals:
        mean, std = stats(vals)
        print(f"{metric:20s}: {mean:.4f} ± {std:.4f}  (n={len(vals)})")

# Save summary
summary = {}
for metric in ["accuracy", "macro_f1", "fpr", "auc", "latency_rf_ms"]:
    vals = [r[metric] for r in results if r.get(metric) is not None]
    if vals:
        mean, std = stats(vals)
        summary[metric] = {"mean": mean, "std": std, "n": len(vals), "values": vals}

summary["seeds"] = [r["seed"] for r in results]
output_path = Path("$OUTPUT_DIR/aggregate_results.json")
with open(output_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nAggregate results saved: {output_path}")
PYEOF
