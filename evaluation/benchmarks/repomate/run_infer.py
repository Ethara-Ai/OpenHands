"""
Repomate benchmark inference runner for OpenHands.

This module provides the entry point for running agent inference on Repomate
benchmark instances. It handles:
- Loading instances from CSV dataset
- Configuring sandbox environments with Docker images
- Running the agent to fix failing tests
- Collecting agent outputs and git patches

Usage:
    python -m evaluation.benchmarks.repomate.run_infer \
        --agent-cls CodeActAgent \
        --llm-config llm \
        --max-iterations 30 \
        --dataset path/to/dataset.csv
"""

import os
import tempfile
import json
from typing import Any

import pandas as pd

# OpenHands imports - these will work when installed in OpenHands
try:
    from evaluation.utils.shared import (
        EvalMetadata,
        EvalOutput,
        make_metadata,
        prepare_dataset,
        run_evaluation,
        get_default_sandbox_config_for_eval,
        get_openhands_config_for_eval,
        reset_logger_for_multiprocessing,
        codeact_user_response,
    )
    from openhands.core.config import (
        get_evaluation_parser,
        get_llm_config_arg,
        AgentConfig,
    )
    from openhands.core.main import create_runtime, run_controller
    from openhands.events.action import CmdRunAction, MessageAction
    from openhands.events.observation import CmdOutputObservation
    from openhands.core.logger import openhands_logger as logger

    OPENHANDS_AVAILABLE = True
except ImportError:
    # Fallback for standalone testing
    import logging

    logger = logging.getLogger(__name__)
    OPENHANDS_AVAILABLE = False

from . import log_parsers
from .test_spec import make_eval_test_spec, RepomateTestSpec


def get_config(instance: pd.Series, metadata: "EvalMetadata"):
    """Configure the sandbox for a Repomate instance.

    Args:
        instance: DataFrame row containing instance data
        metadata: Evaluation metadata

    Returns:
        OpenHandsConfig: Configuration for the sandbox
    """
    sandbox_config = get_default_sandbox_config_for_eval()

    # Use the image_storage_uri from the CSV if available
    image_uri = instance.get("image_storage_uri", "")
    if image_uri:
        sandbox_config.base_container_image = image_uri
    else:
        # Fallback to constructing image name from instance_id
        instance_id = instance["instance_id"]
        sandbox_config.base_container_image = f"repomate/{instance_id}:latest"

    sandbox_config.platform = "linux/amd64"
    sandbox_config.enable_auto_lint = True
    sandbox_config.use_host_network = False

    config = get_openhands_config_for_eval(
        metadata=metadata,
        sandbox_config=sandbox_config,
    )

    agent_config = AgentConfig(
        enable_jupyter=False,
        enable_browsing=False,
        enable_mcp=False,
    )
    config.set_agent_config(agent_config)

    return config


def initialize_runtime(runtime, instance: pd.Series, metadata: "EvalMetadata"):
    """Initialize the runtime for Repomate evaluation.

    This sets up the container environment before the agent starts working.

    Args:
        runtime: OpenHands runtime instance
        instance: DataFrame row containing instance data
        metadata: Evaluation metadata
    """
    logger.info("-" * 30)
    logger.info("BEGIN Runtime Initialization")
    logger.info("-" * 30)

    # Source the saved environment
    action = CmdRunAction(command="source /saved/ENV || source /saved/*/ENV || true")
    action.set_hard_timeout(300)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})

    # Navigate to repository
    action = CmdRunAction(command="cd /app/repo")
    action.set_hard_timeout(60)
    logger.info(action, extra={"msg_type": "ACTION"})
    obs = runtime.run_action(action)
    logger.info(obs, extra={"msg_type": "OBSERVATION"})

    # Configure git
    action = CmdRunAction(
        command="git config --global --add safe.directory /app/repo && "
        'git config --global core.pager ""'
    )
    action.set_hard_timeout(60)
    runtime.run_action(action)

    # Reset to clean state
    action = CmdRunAction(command="git reset --hard && git clean -fd")
    action.set_hard_timeout(120)
    runtime.run_action(action)

    # Show current state
    action = CmdRunAction(command="git status && git log -1 --oneline")
    action.set_hard_timeout(60)
    obs = runtime.run_action(action)
    logger.info(f"Repository state: {obs.content if hasattr(obs, 'content') else obs}")

    logger.info("-" * 30)
    logger.info("END Runtime Initialization")
    logger.info("-" * 30)


def get_instruction(instance: pd.Series) -> str:
    """Build the instruction prompt for the agent.

    Args:
        instance: DataFrame row containing instance data

    Returns:
        str: Formatted instruction for the agent
    """
    problem_statement = instance.get("problem_statement", "")
    test_command = instance.get("test_command", "pytest")

    instruction = f"""You are a software engineer working on fixing a bug in a repository.

## Problem Description
{problem_statement if problem_statement else "There is a failing test that needs to be fixed."}

## Your Task
1. Understand the failing test and the bug it reveals
2. Navigate the codebase to find the source of the bug
3. Implement a fix for the bug
4. Verify your fix by running the tests

## Test Command
You can run the tests with:
```bash
{test_command}
```

## Important Notes
- Focus on fixing the bug, not modifying the tests
- Make minimal changes to fix the issue
- Ensure all tests pass after your fix
"""
    return instruction


def complete_runtime(runtime, instance: pd.Series) -> dict[str, Any]:
    """Complete the runtime and collect results.

    Args:
        runtime: OpenHands runtime instance
        instance: DataFrame row containing instance data

    Returns:
        dict: Contains git_patch and test_output
    """
    # Get the git diff (the agent's patch)
    action = CmdRunAction(command="git diff")
    action.set_hard_timeout(120)
    obs = runtime.run_action(action)
    git_patch = obs.content if hasattr(obs, "content") else ""

    # Run tests to get final status
    test_command = instance.get("test_command", "pytest")
    action = CmdRunAction(command=test_command)
    action.set_hard_timeout(600)
    obs = runtime.run_action(action)
    test_output = obs.content if hasattr(obs, "content") else ""

    return {
        "git_patch": git_patch,
        "test_output": test_output,
    }


def process_instance(
    instance: pd.Series,
    metadata: "EvalMetadata",
    reset_logger: bool = True,
) -> "EvalOutput":
    """Process a single Repomate instance.

    Args:
        instance: DataFrame row containing instance data
        metadata: Evaluation metadata
        reset_logger: Whether to reset the logger for multiprocessing

    Returns:
        EvalOutput: Results from processing this instance
    """
    if reset_logger:
        reset_logger_for_multiprocessing(logger, instance["instance_id"])

    instance_id = instance["instance_id"]
    logger.info(f"Processing instance: {instance_id}")

    config = get_config(instance, metadata)
    runtime = create_runtime(config)

    try:
        initialize_runtime(runtime, instance, metadata)

        instruction = get_instruction(instance)

        state = run_controller(
            config=config,
            initial_user_action=MessageAction(content=instruction),
            runtime=runtime,
            fake_user_response_fn=codeact_user_response,
        )

        # Complete and get results
        results = complete_runtime(runtime, instance)

        # Parse test results
        parser_name = instance.get("test_output_parser", "python/parse_log_pytest_v3")
        if "/" in parser_name:
            parser_name = parser_name.split("/")[1]
        parser_fn = getattr(log_parsers, parser_name, log_parsers.parse_log_pytest_v3)
        test_results = parser_fn(results["test_output"], None)

        return EvalOutput(
            instance_id=instance_id,
            instruction=instruction,
            metadata=metadata,
            history=state.history if hasattr(state, "history") else [],
            test_result=test_results,
            git_patch=results["git_patch"],
        )

    except Exception as e:
        logger.error(f"Error processing instance {instance_id}: {e}")
        raise
    finally:
        runtime.close()


def main():
    """Main entry point for Repomate inference."""
    if not OPENHANDS_AVAILABLE:
        print("Error: OpenHands is not installed. Please install it first.")
        print("  pip install openhands")
        return

    parser = get_evaluation_parser()
    parser.add_argument(
        "--dataset", type=str, required=True, help="Path to Repomate CSV dataset file"
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

    # Setup metadata
    llm_config = get_llm_config_arg(args.llm_config)
    metadata = make_metadata(
        llm_config,
        f"repomate-{os.path.basename(args.dataset)}",
        args.agent_cls,
        args.max_iterations,
        args.eval_note,
        args.eval_output_dir,
    )

    output_file = os.path.join(metadata.eval_output_dir, "output.jsonl")
    logger.info(f"Output file: {output_file}")

    # Prepare and run
    instances = prepare_dataset(dataset, output_file, args.eval_n_limit)

    run_evaluation(
        instances,
        metadata,
        output_file,
        args.eval_num_workers,
        process_instance,
        timeout_seconds=4 * 60 * 60,  # 4 hours per instance
        max_retries=3,
    )


if __name__ == "__main__":
    main()
