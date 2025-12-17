# -*- mode: python; mode: blacken -*-
"""
Repomate Test Specification for OpenHands evaluation.
Adapted from the original test_spec.py with OpenHands-compatible imports.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)

# Constants
FAIL_TO_PASS = "FAIL_TO_PASS"
PASS_TO_PASS = "PASS_TO_PASS"
START_TEST_OUTPUT = ">>>>> Start Test Output"
END_TEST_OUTPUT = ">>>>> End Test Output"
REPOMATE_NAMESPACE = "repomate"
DEFAULT_LOG_PARSER = "default"


@dataclass
class TestSpec:
    """Base test specification dataclass."""

    instance_id: str
    repo: str
    version: str
    arch: str = "x86_64"
    env_script_list: list = field(default_factory=list)
    repo_script_list: list = field(default_factory=list)
    eval_script_list: list = field(default_factory=list)
    FAIL_TO_PASS: list = field(default_factory=list)
    PASS_TO_PASS: list = field(default_factory=list)
    language: str = "py"
    docker_specs: dict = field(default_factory=dict)
    namespace: str = "swebench"
    vmvm_dataset_name: str = ""
    container_mem: str = "4g"
    container_memswap: str = "4g"

    @property
    def instance_image_key(self) -> str:
        return f"sweb.eval.{self.arch}.{self.instance_id}"


@dataclass
class RepomateTestSpec(TestSpec):
    """Repomate-specific test specification."""

    image_storage_uri: Optional[str] = None


def get_default_test_cmd_for_language(language: str) -> str:
    """Get the default test command for a given programming language.

    Args:
        language: Programming language code

    Returns:
        str: Default test command for the language
    """
    language_to_default_test_cmd = {
        "python": "pytest",
        "javascript": "npm test",
        "java": "mvn test",
        "c": "make check",
        "cpp": "make test",
        "go": "go test",
        "rust": "cargo test",
    }

    return language_to_default_test_cmd.get(language, "pytest")


def get_test_cmd_for_repomate_instance(
    test_cmd_parser: Optional[str], test_cmd: Optional[str]
) -> str:
    """Get the appropriate test command for a repomate instance.

    Args:
        test_cmd_parser: parser command in the format language/test_framework
        test_cmd: Test command from environment

    Returns:
        str: Test command to use
    """
    if test_cmd is not None and test_cmd != "":
        return test_cmd

    if test_cmd_parser and "/" in test_cmd_parser:
        repo_language = test_cmd_parser.split("/")[0].lower()
    else:
        repo_language = "python"
    return get_default_test_cmd_for_language(repo_language)


def make_eval_test_spec(instance: Union[dict, pd.Series]) -> RepomateTestSpec:
    """Create a RepomateTestSpec from an instance dictionary or Series.

    Args:
        instance: Dictionary or pandas Series containing instance data

    Returns:
        RepomateTestSpec: Configured test specification
    """
    if hasattr(instance, "to_dict"):
        instance = instance.to_dict()

    # Extract repo from instance_id if not provided
    if "repo" in instance:
        repo = instance["repo"]
    else:
        # Format: owner__repo-commit_hash
        instance_id = instance["instance_id"]
        repo = instance_id.rsplit("-", 1)[0].replace("__", "/")

    version = "repomate"
    test_patch = instance.get("test_patch", "")
    base_commit = instance.get("base_commit", "HEAD")
    repo_directory = "/app/repo"

    HEREDOC_DELIMITER = "EOF_114329324912"

    # Apply test patch command
    apply_test_patch_command = (
        f"git apply -v - <<'{HEREDOC_DELIMITER}'\n{test_patch}\n{HEREDOC_DELIMITER}"
        if test_patch
        else ""
    )

    test_command = instance.get("test_command", "pytest")
    logger.info(f"test_command: {test_command}")

    eval_commands = [
        f"cd {repo_directory}",
        "source /saved/ENV || source /saved/*/ENV || true",
        f"cd {repo_directory}",
        f"git config --global --add safe.directory {repo_directory}",
        f"cd {repo_directory}",
        "git status",
        "git show --stat",
        f"git -c core.fileMode=false diff {base_commit}",
    ]

    if apply_test_patch_command:
        eval_commands.append(apply_test_patch_command)

    eval_commands += [
        f": '{START_TEST_OUTPUT}'",
        test_command,
        f": '{END_TEST_OUTPUT}'",
    ]

    def _from_json_or_obj(value: Any) -> Any:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value

    return RepomateTestSpec(
        instance_id=instance["instance_id"],
        repo=repo,
        env_script_list=[],
        repo_script_list=[],
        eval_script_list=eval_commands,
        version=version,
        arch="x86_64",
        FAIL_TO_PASS=_from_json_or_obj(instance.get(FAIL_TO_PASS, [])),
        PASS_TO_PASS=_from_json_or_obj(instance.get(PASS_TO_PASS, [])),
        language="py",
        docker_specs={},
        namespace=REPOMATE_NAMESPACE,
        vmvm_dataset_name="repomate_image_activ_pytest",
        container_mem="4g",
        container_memswap="4g",
        image_storage_uri=instance.get("image_storage_uri"),
    )


def get_repomate_instance_image_key(instance_id: str) -> str:
    """Convert instance_id to Docker image key format."""
    parts = instance_id.rsplit("-", 1)
    if len(parts) == 2:
        return f"{parts[0]}:{parts[1]}"
    return instance_id
