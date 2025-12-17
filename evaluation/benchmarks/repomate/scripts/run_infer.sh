#!/bin/bash
# Run Repomate inference with OpenHands agent
#
# Usage:
#   ./run_infer.sh <llm_config> <agent_class> <max_iterations> <dataset_path> [eval_note]
#
# Example:
#   ./run_infer.sh llm CodeActAgent 30 ../../../data/python-only.csv my_experiment

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCHMARK_DIR="$(dirname "$SCRIPT_DIR")"

# Arguments with defaults
LLM_CONFIG="${1:-llm}"
AGENT_CLS="${2:-CodeActAgent}"
MAX_ITERATIONS="${3:-30}"
DATASET_PATH="${4:-python-only-below-0_1-rank-by-fp.csv}"
EVAL_NOTE="${5:-}"

# Generate output directory name
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
DATASET_NAME=$(basename "$DATASET_PATH" .csv)
OUTPUT_DIR="outputs/repomate/${DATASET_NAME}/${AGENT_CLS}/${TIMESTAMP}"

echo "========================================"
echo "Repomate Benchmark Inference"
echo "========================================"
echo "LLM Config:      $LLM_CONFIG"
echo "Agent:           $AGENT_CLS"
echo "Max Iterations:  $MAX_ITERATIONS"
echo "Dataset:         $DATASET_PATH"
echo "Output Dir:      $OUTPUT_DIR"
echo "========================================"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run inference
python -m evaluation.benchmarks.repomate.run_infer \
    --llm-config "$LLM_CONFIG" \
    --agent-cls "$AGENT_CLS" \
    --max-iterations "$MAX_ITERATIONS" \
    --dataset "$DATASET_PATH" \
    --eval-output-dir "$OUTPUT_DIR" \
    ${EVAL_NOTE:+--eval-note "$EVAL_NOTE"}

echo "========================================"
echo "Inference complete!"
echo "Output: $OUTPUT_DIR/output.jsonl"
echo "========================================"
