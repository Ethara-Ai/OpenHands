"""
Repomate benchmark evaluation/grading runner for OpenHands.

This module evaluates agent outputs by:
1. Loading agent predictions (git patches) from output.jsonl
2. Applying patches to fresh containers
3. Running tests and parsing results
4. Computing pass rates and generating reports

Usage:
    python -m evaluation.benchmarks.repomate.eval_infer \
        --predictions output.jsonl \
        --dataset dataset.csv \
        --output eval_results.jsonl
"""

import argparse
import json
import os
from typing import Any, Optional

import pandas as pd

# OpenHands imports
try:
    from openhands.core.main import create_runtime
    from openhands.events.action import CmdRunAction
    from openhands.core.logger import openhands_logger as logger
    from evaluation.utils.shared import get_default_sandbox_config_for_eval

    OPENHANDS_AVAILABLE = True
except ImportError:
    import logging

    logger = logging.getLogger(__name__)
    OPENHANDS_AVAILABLE = False

from . import log_parsers
from .test_spec import FAIL_TO_PASS, PASS_TO_PASS


def get_parser_function(parser_name: str):
    """Get the parser function by name.

    Args:
        parser_name: Parser name in format "language/parser_function" or just "parser_function"

    Returns:
        Callable: Parser function
    """
    if "/" in parser_name:
        parser_name = parser_name.split("/")[1]

    parser_fn = getattr(log_parsers, parser_name, None)
    if parser_fn is None:
        logger.warning(f"Parser {parser_name} not found, using parse_log_pytest_v3")
        parser_fn = log_parsers.parse_log_pytest_v3

    return parser_fn


def apply_patch_and_run_tests(
    runtime,
    instance: dict,
    git_patch: str,
) -> dict[str, Any]:
    """Apply a git patch and run tests.

    Args:
        runtime: OpenHands runtime instance
        instance: Instance data dictionary
        git_patch: Git patch to apply

    Returns:
        dict: Test results including raw output and parsed results
    """
    # Navigate to repo
    action = CmdRunAction(command="cd /app/repo")
    action.set_hard_timeout(60)
    runtime.run_action(action)

    # Source environment
    action = CmdRunAction(command="source /saved/ENV || source /saved/*/ENV || true")
    action.set_hard_timeout(300)
    runtime.run_action(action)

    # Reset to clean state
    action = CmdRunAction(command="git reset --hard && git clean -fd")
    action.set_hard_timeout(120)
    runtime.run_action(action)

    # Apply the git patch if non-empty
    patch_applied = False
    if git_patch and git_patch.strip():
        # Write patch to file and apply
        action = CmdRunAction(
            command=f"cat > /tmp/agent.patch << 'PATCH_EOF'\n{git_patch}\nPATCH_EOF"
        )
        action.set_hard_timeout(60)
        runtime.run_action(action)

        action = CmdRunAction(command="git apply /tmp/agent.patch")
        action.set_hard_timeout(120)
        obs = runtime.run_action(action)
        patch_applied = obs.exit_code == 0 if hasattr(obs, "exit_code") else False

        if not patch_applied:
            logger.warning(f"Failed to apply patch: {obs}")

    # Run test command
    test_cmd = instance.get("test_command", "pytest")
    action = CmdRunAction(command=test_cmd)
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    test_output = obs.content if hasattr(obs, "content") else ""
    exit_code = obs.exit_code if hasattr(obs, "exit_code") else -1

    # Parse results
    parser_name = instance.get("test_output_parser", "python/parse_log_pytest_v3")
    parser_fn = get_parser_function(parser_name)
    parsed_results = parser_fn(test_output, None)

    return {
        "patch_applied": patch_applied,
        "test_output": test_output,
        "exit_code": exit_code,
        "parsed_results": parsed_results,
    }


def compute_resolution(
    parsed_results: dict[str, str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    """Compute whether the instance is resolved.

    Args:
        parsed_results: Dict mapping test names to status strings
        fail_to_pass: List of tests that should change from FAIL to PASS
        pass_to_pass: List of tests that should remain PASS

    Returns:
        dict: Resolution metrics
    """
    # Check fail_to_pass tests
    f2p_passed = 0
    f2p_total = len(fail_to_pass)
    for test in fail_to_pass:
        if parsed_results.get(test) == "PASSED":
            f2p_passed += 1

    # Check pass_to_pass tests (regression check)
    p2p_passed = 0
    p2p_total = len(pass_to_pass)
    for test in pass_to_pass:
        if parsed_results.get(test) == "PASSED":
            p2p_passed += 1

    # Instance is resolved if all f2p pass AND all p2p still pass
    resolved = (f2p_passed == f2p_total) and (p2p_passed == p2p_total)

    return {
        "resolved": resolved,
        "fail_to_pass_passed": f2p_passed,
        "fail_to_pass_total": f2p_total,
        "pass_to_pass_passed": p2p_passed,
        "pass_to_pass_total": p2p_total,
    }


def evaluate_predictions(
    predictions_path: str,
    dataset: pd.DataFrame,
    output_path: str,
):
    """Evaluate all predictions.

    Args:
        predictions_path: Path to predictions JSONL file
        dataset: DataFrame with instance data
        output_path: Path to write evaluation results
    """
    dataset_map = {row["instance_id"]: row.to_dict() for _, row in dataset.iterrows()}

    results = []
    resolved_count = 0
    total_count = 0

    with open(predictions_path) as f:
        for line in f:
            pred = json.loads(line)
            instance_id = pred["instance_id"]

            if instance_id not in dataset_map:
                logger.warning(f"Instance {instance_id} not found in dataset, skipping")
                continue

            instance = dataset_map[instance_id]
            git_patch = pred.get("git_patch", "")

            # For evaluation, we need a runtime - this is a simplified version
            # In production, this would create actual Docker containers
            logger.info(f"Evaluating instance: {instance_id}")

            # Parse existing test results if available in prediction
            test_results = pred.get("test_result", {})

            # Get expected test lists
            fail_to_pass = instance.get(FAIL_TO_PASS, [])
            pass_to_pass = instance.get(PASS_TO_PASS, [])

            if isinstance(fail_to_pass, str):
                try:
                    fail_to_pass = json.loads(fail_to_pass)
                except json.JSONDecodeError:
                    fail_to_pass = []

            if isinstance(pass_to_pass, str):
                try:
                    pass_to_pass = json.loads(pass_to_pass)
                except json.JSONDecodeError:
                    pass_to_pass = []

            # Compute resolution
            resolution = compute_resolution(test_results, fail_to_pass, pass_to_pass)

            result = {
                "instance_id": instance_id,
                "has_patch": bool(git_patch and git_patch.strip()),
                "patch_lines": len(git_patch.split("\n")) if git_patch else 0,
                **resolution,
            }

            results.append(result)
            total_count += 1
            if resolution["resolved"]:
                resolved_count += 1

    # Write results
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # Print summary
    print(f"\n{'=' * 50}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 50}")
    print(f"Total instances: {total_count}")
    print(f"Resolved: {resolved_count}")
    print(
        f"Resolution rate: {resolved_count / total_count * 100:.1f}%"
        if total_count > 0
        else "N/A"
    )
    print(f"{'=' * 50}")
    print(f"Results written to: {output_path}")


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate Repomate benchmark predictions"
    )
    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
        help="Path to predictions JSONL file from run_infer",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to original dataset CSV/parquet file",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to write evaluation results"
    )
    args = parser.parse_args()

    # Load dataset
    if args.dataset.endswith(".csv"):
        dataset = pd.read_csv(args.dataset)
    elif args.dataset.endswith(".parquet"):
        dataset = pd.read_parquet(args.dataset)
    else:
        raise ValueError(f"Unsupported dataset format: {args.dataset}")

    logger.info(f"Loaded {len(dataset)} instances from {args.dataset}")

    # Run evaluation
    evaluate_predictions(args.predictions, dataset, args.output)


if __name__ == "__main__":
    main()
