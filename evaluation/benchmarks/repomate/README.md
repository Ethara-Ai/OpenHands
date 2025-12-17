# Repomate Benchmark for OpenHands

Multi-language test evaluation benchmark with support for 15+ testing frameworks.

## Overview

Repomate provides:
- Pre-built Docker images for reproducible test environments
- Comprehensive log parsers for multiple test frameworks
- Integration with OpenHands evaluation pipeline

## Supported Languages & Frameworks

| Language | Frameworks |
|----------|------------|
| Python | pytest, unittest, Django, Sympy, Matplotlib, Seaborn |
| JavaScript | Jest, Mocha |
| Go | go test |
| Rust | cargo test |
| C/C++ | Check, CTest, make check |
| Java | JUnit |

## Quick Start

### Prerequisites

- OpenHands installed and configured
- Docker running
- Access to Repomate Docker images

### Running Inference

```bash
# Using the shell script
./evaluation/benchmarks/repomate/scripts/run_infer.sh \
    llm \                    # LLM config name from config.toml
    CodeActAgent \           # Agent class
    30 \                     # Max iterations
    path/to/dataset.csv      # Dataset path

# Or directly with Python
python -m evaluation.benchmarks.repomate.run_infer \
    --llm-config llm \
    --agent-cls CodeActAgent \
    --max-iterations 30 \
    --dataset path/to/dataset.csv \
    --eval-output-dir outputs/repomate/
```

### Running Evaluation

```bash
# Using the shell script
./evaluation/benchmarks/repomate/scripts/eval_infer.sh \
    outputs/output.jsonl \     # Predictions from run_infer
    path/to/dataset.csv \      # Original dataset
    outputs/eval_results.jsonl # Output path

# Or directly with Python
python -m evaluation.benchmarks.repomate.eval_infer \
    --predictions outputs/output.jsonl \
    --dataset path/to/dataset.csv \
    --output outputs/eval_results.jsonl
```

## Dataset Format

The benchmark expects a CSV file with the following columns:

| Column | Description |
|--------|-------------|
| `instance_id` | Unique identifier for the instance |
| `image_storage_uri` | Docker image reference (e.g., `gcr.io/project/image:tag`) |
| `test_command` | Command to run tests (e.g., `pytest tests/`) |
| `test_output_parser` | Parser to use (e.g., `python/parse_log_pytest_v3`) |
| `problem_statement` | Description of the issue to fix |
| `FAIL_TO_PASS` | JSON list of tests that should change from FAIL to PASS |
| `PASS_TO_PASS` | JSON list of tests that should remain PASS (regression check) |

## Available Log Parsers

### Python
- `parse_log_pytest` - Standard pytest output
- `parse_log_pytest_v2` - Pytest with ANSI color codes
- `parse_log_pytest_v3` - Repomate optimized pytest parser
- `parse_log_unittest` - Python unittest framework
- `parse_log_django` - Django test runner
- `parse_log_sympy` - SymPy test format
- `parse_log_matplotlib` - Matplotlib pytest format
- `parse_log_seaborn` - Seaborn test format

### JavaScript
- `parse_log_jest` - Jest with --verbose
- `parse_log_jest_json` - Jest with --json
- `parse_log_mocha_json` - Mocha with JSON stream

### Other Languages
- `parse_log_gotest_json` - Go test with JSON output
- `parse_log_cargo_test` - Rust cargo test
- `parse_log_check_framework` - C Check framework / CTest
- `parse_log_junit` - Java JUnit XML format

## Integration Notes

This benchmark is adapted from the Repomate harness for OpenHands integration. Key differences from standard SWE-bench:

1. **Docker Images**: Uses `image_storage_uri` field directly instead of constructing image names
2. **Working Directory**: Expects repo at `/app/repo` instead of `/workspace`
3. **Environment**: Sources `/saved/ENV` for environment variables
4. **Multi-language**: Supports parsers for many languages beyond Python

## File Structure

```
repomate/
├── __init__.py           # Package exports
├── log_parsers.py        # 50+ test log parsers
├── test_spec.py          # Test specification dataclasses
├── run_infer.py          # Agent inference runner
├── eval_infer.py         # Evaluation/grading runner
├── README.md             # This file
└── scripts/
    ├── run_infer.sh      # Shell wrapper for inference
    └── eval_infer.sh     # Shell wrapper for evaluation
```

## Troubleshooting

### Docker image not found
Ensure the `image_storage_uri` in your CSV points to accessible Docker images.

### Parser not matching output
Check that `test_output_parser` in your CSV matches the actual test framework output format.

### Tests not being detected
Verify the test command produces output that the parser can recognize.
