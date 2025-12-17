#!/bin/bash
# Evaluate Repomate predictions
#
# Usage:
#   ./eval_infer.sh <predictions_path> <dataset_path> <output_path>
#
# Example:
#   ./eval_infer.sh outputs/output.jsonl data/dataset.csv outputs/eval_results.jsonl

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Arguments with defaults
PREDICTIONS="${1:-outputs/repomate/latest/output.jsonl}"
DATASET="${2:-python-only-below-0_1-rank-by-fp.csv}"
OUTPUT="${3:-outputs/repomate/latest/eval_results.jsonl}"

echo "========================================"
echo "Repomate Benchmark Evaluation"
echo "========================================"
echo "Predictions:  $PREDICTIONS"
echo "Dataset:      $DATASET"
echo "Output:       $OUTPUT"
echo "========================================"

# Ensure output directory exists
mkdir -p "$(dirname "$OUTPUT")"

# Run evaluation
python -m evaluation.benchmarks.repomate.eval_infer \
    --predictions "$PREDICTIONS" \
    --dataset "$DATASET" \
    --output "$OUTPUT"

echo "========================================"
echo "Evaluation complete!"
echo "Results: $OUTPUT"
echo "========================================"
