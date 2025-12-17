import hashlib
import html
import json
import logging
import re
import xml.etree.ElementTree as ET

from typing import Any

# Type alias - CodingAgentGradingSpec is only used for type hints
CodingAgentGradingSpec = Any
from enum import Enum


class TestStatus(Enum):
    """Test status enum compatible with swebench."""
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"

logger = logging.getLogger()


def parse_log_pytest(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = TestStatus(test_case[0])
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_pytest_swesmith(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """Parser for test logs generated with PyTest framework. Default parser from swe-smith"""
    test_status_map = {}
    for line in log.split("\n"):
        for status in TestStatus:
            is_match = re.match(rf"^(\S+)(\s+){status.value}", line)
            if is_match:
                test_status_map[is_match.group(1)] = status
                continue
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_pytest_options(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework with options

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    option_pattern = re.compile(r"(.*?)\[(.*)\]")
    test_status_map = {}
    for line in log.split("\n"):
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            has_option = option_pattern.search(test_case[1])
            if has_option:
                main, option = has_option.groups()
                if (
                    option.startswith("/")
                    and not option.startswith("//")
                    and "*" not in option
                ):
                    option = "/" + option.split("/")[-1]
                test_name = f"{main}[{option}]"
            else:
                test_name = test_case[1]
            test_status_map[test_name] = TestStatus(test_case[0])
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_django(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Django tester framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    prev_test = None
    for line in lines:  # type: ignore
        line = line.strip()

        # This isn't ideal but the test output spans multiple lines
        if "--version is equivalent to version" in line:
            test_status_map["--version is equivalent to version"] = TestStatus.PASSED

        # Log it in case of error
        if " ... " in line:
            prev_test = line.split(" ... ")[0]

        pass_suffixes = (" ... ok", " ... OK", " ...  OK")
        for suffix in pass_suffixes:
            if line.endswith(suffix):
                # TODO: Temporary, exclusive fix for django__django-7188
                # The proper fix should involve somehow getting the test results to
                # print on a separate line, rather than the same line
                if line.strip().startswith(
                    "Applying sites.0002_alter_domain_unique...test_no_migrations"
                ):
                    line = line.split("...", 1)[-1].strip()
                test = line.rsplit(suffix, 1)[0]
                test_status_map[test] = TestStatus.PASSED
                break
        if " ... skipped" in line:
            test = line.split(" ... skipped")[0]
            test_status_map[test] = TestStatus.SKIPPED
        if line.endswith(" ... FAIL"):
            test = line.split(" ... FAIL")[0]
            test_status_map[test] = TestStatus.FAILED
        if line.startswith("FAIL:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.FAILED
        if line.endswith(" ... ERROR"):
            test = line.split(" ... ERROR")[0]
            test_status_map[test] = TestStatus.ERROR
        if line.startswith("ERROR:"):
            test = line.split()[1].strip()
            test_status_map[test] = TestStatus.ERROR

        if line.lstrip().startswith("ok") and prev_test is not None:
            # It means the test passed, but there's some additional output (including new lines)
            # between "..." and "ok" message
            test = prev_test
            test_status_map[test] = TestStatus.PASSED

    # TODO: This is very brittle, we should do better
    # There's a bug in the django logger, such that sometimes a test output near the end gets
    # interrupted by a particular long multiline print statement.
    # We have observed this in one of 3 forms:
    # - "{test_name} ... Testing against Django installed in {*} silenced.\nok"
    # - "{test_name} ... Internal Server Error: \/(.*)\/\nok"
    # - "{test_name} ... System check identified no issues (0 silenced).\nok"
    patterns = [
        r"^(.*?)\s\.\.\.\sTesting\ against\ Django\ installed\ in\ ((?s:.*?))\ silenced\)\.\nok$",
        r"^(.*?)\s\.\.\.\sInternal\ Server\ Error:\ \/(.*)\/\nok$",
        r"^(.*?)\s\.\.\.\sSystem check identified no issues \(0 silenced\)\nok$",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, log, re.MULTILINE):
            test_name = match.group(1)
            test_status_map[test_name] = TestStatus.PASSED
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_pytest_v2(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Later Version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        translator = str.maketrans("", "", escapes)
        line = line.translate(translator)
        if any(line.startswith(x.value) for x in TestStatus):
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[1]] = TestStatus(test_case[0])
        # Support older pytest versions by checking if the line ends with the test status
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = TestStatus(test_case[1])
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_pytest_v3(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework (Repomate version)

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    escapes = "".join([chr(char) for char in range(1, 32)])
    translator = str.maketrans("", "", escapes)

    # Precompile the regex pattern for efficiency
    status_values = "|".join(x.value for x in TestStatus)
    status_pattern = re.compile(rf"^({status_values})\s+")

    for line in log.split("\n"):
        line = re.sub(r"\[(\d+)m", "", line)
        line = line.translate(translator)
        line = line.replace(" - ", " ")

        match = status_pattern.match(line)
        if match:
            test_case = line.split()
            if len(test_case) >= 2:
                status = test_case[0]
                test_name = test_case[1]
                test_status_map[test_name] = TestStatus(status)

        # Optional: still support older pytest output where status is at the end
        elif any(line.endswith(x.value) for x in TestStatus):
            test_case = line.split()
            if len(test_case) >= 2:
                test_status_map[test_case[0]] = TestStatus(test_case[1])

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_seaborn(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    """
    Parser for test logs generated with seaborn testing framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.startswith(TestStatus.FAILED.value):
            test_case = line.split()[1]
            test_status_map[test_case] = TestStatus.FAILED
        elif f" {TestStatus.PASSED.value} " in line:
            parts = line.split()
            if parts[1] == TestStatus.PASSED.value:
                test_case = parts[0]
                test_status_map[test_case] = TestStatus.PASSED
        elif line.startswith(TestStatus.PASSED.value):
            parts = line.split()
            test_case = parts[1]
            test_status_map[test_case] = TestStatus.PASSED
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_sympy(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Sympy framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pattern = r"(_*) (.*)\.py:(.*) (_*)"
    matches = re.findall(pattern, log)
    for match in matches:
        test_case = f"{match[1]}.py:{match[2]}"
        test_status_map[test_case] = TestStatus.FAILED
    for line in log.split("\n"):
        line = line.strip()
        if line.startswith("test_"):
            if line.endswith(" E"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.ERROR
            if line.endswith(" F"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.FAILED
            if line.endswith(" ok"):
                test = line.split()[0]
                test_status_map[test] = TestStatus.PASSED
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_matplotlib(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs generated with PyTest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        line = line.replace("MouseButton.LEFT", "1")
        line = line.replace("MouseButton.RIGHT", "3")
        if any(line.startswith(x.value) for x in TestStatus):
            # Additional parsing for FAILED status
            if line.startswith(TestStatus.FAILED.value):
                line = line.replace(" - ", " ")
            test_case = line.split()
            if len(test_case) <= 1:
                continue
            test_status_map[test_case[1]] = TestStatus(test_case[0])
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_unittest(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs generated with unittest framework

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    lines = log.split("\n")

    line_num = 0
    while line_num < len(lines):
        line = lines[line_num].strip()

        line_starts_with_error = line.startswith("ERROR:")
        line_starts_with_failed = line.startswith("FAILED:")

        if line_starts_with_error or line_starts_with_failed:
            parts = line.split()
            if len(parts) >= 2:
                test_name = parts[1]
                if "(" in test_name:
                    if line_starts_with_error:
                        # Error tests already have the full name, just extract the test name
                        test_name = test_name.split("(")[0]
                    elif line_starts_with_failed:
                        class_name = test_name.split("(")[1].split(")")[0]
                        base_test_name = test_name.split("(")[0]
                        test_name = f"{class_name}.{base_test_name}"
                if line_starts_with_error:
                    test_status_map[test_name] = TestStatus.ERROR
                elif line_starts_with_failed:
                    test_status_map[test_name] = TestStatus.FAILED

        # Python unittest framework requires all tests to start with the "test_" prefix
        elif line.startswith("test_"):
            test_name = line.split()[0] if line.split() else ""

            if test_name.startswith("test_"):
                class_name = ""
                if "(" in line and ")" in line:
                    # Extract class name from pattern like "test_method (package.module.ClassName)"
                    paren_content = line.split("(")[1].split(")")[0]
                    class_name = paren_content

                test_content = line
                lookahead_line_num = line_num + 1

                stop_prefixes = (
                    "test_",
                    "======================================================================",
                    "----------------------------------------------------------------------",
                )
                while lookahead_line_num < len(lines):
                    current_line = lines[lookahead_line_num].strip()
                    if any(current_line.startswith(prefix) for prefix in stop_prefixes):
                        break
                    test_content += " " + current_line
                    lookahead_line_num += 1

                full_test_name = (
                    f"{class_name}.{test_name}" if class_name else test_name
                )
                if " ok" in test_content or test_content.endswith("ok"):
                    test_status_map[full_test_name] = TestStatus.PASSED
                elif "FAIL:" in test_content or test_content.endswith("FAIL"):
                    test_status_map[full_test_name] = TestStatus.FAILED
                elif "ERROR:" in test_content or test_content.endswith("ERROR"):
                    # error tests have full test name already
                    test_status_map[test_name] = TestStatus.ERROR
                elif "skipped" in test_content:
                    test_status_map[full_test_name] = TestStatus.SKIPPED
                else:
                    # retroactive TestStatus assignment for test results that span multiple lines
                    remaining_lines = lines[line_num:lookahead_line_num]
                    combined_content = " ".join(remaining_lines)
                    if combined_content.rstrip().endswith(
                        " ok"
                    ) or combined_content.rstrip().endswith("\nok"):
                        test_status_map[full_test_name] = TestStatus.PASSED

                line_num = lookahead_line_num - 1

        line_num += 1

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def log_parser_hips_autograd(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs from HIPS/autograd repository

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        for status in TestStatus:
            is_match = re.match(rf"^\[gw\d\]\s{status.value}\s(\S+)", line)
            if is_match:
                test_status_map[is_match.group(1)] = status
                continue
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def log_parser_paramiko_paramiko(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs from paramiko/paramiko repository

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        for status in TestStatus:
            is_match = re.match(rf"^{status.value}\s(\S+)", line)
            if is_match:
                test_status_map[is_match.group(1)] = status
                continue
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def log_parser_un33k_python_slugify(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs from un33k/python-slugify repository

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    pattern = r"^([a-zA-Z0-9_\-,\.\s\(\)']+)\s\.{3}\s"
    for line in log.split("\n"):
        is_match = re.match(f"{pattern}ok$", line)
        if is_match:
            test_status_map[is_match.group(1)] = TestStatus.PASSED
            continue
        for keyword, status in {
            "FAIL": TestStatus.FAILED,
            "ERROR": TestStatus.ERROR,
        }.items():
            is_match = re.match(f"{pattern}{keyword}$", line)
            if is_match:
                test_status_map[is_match.group(1)] = status
                continue
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def log_parser_tornadoweb_tornado(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs from tornadoweb/tornado repository

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        if line.endswith("... ok"):
            test_case = line.split(" ... ")[0]
            test_status_map[test_case] = TestStatus.PASSED
        elif " ... skipped " in line:
            test_case = line.split(" ... ")[0]
            test_status_map[test_case] = TestStatus.SKIPPED
        elif any(line.startswith(x) for x in ["ERROR:", "FAIL:"]):
            test_case = " ".join(line.split()[1:3])
            test_status_map[test_case] = TestStatus.FAILED
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def log_parser_python_mypy(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs from python/mypy repository

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}
    for line in log.split("\n"):
        for status in [
            TestStatus.PASSED,
            TestStatus.FAILED,
        ]:
            if status.value in line:
                test_case = line.split()[-1]
                test_status_map[test_case] = status
                break
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_junit(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    logger.info("Parsing junit test log")
    logger.debug(f"Log: {log}")
    logger.debug(f"Log type: {type(log)}")

    # Try to parse as JSON first
    try:
        log_lines = json.loads(log)
    except json.JSONDecodeError:
        logger.debug("JSON parsing failed, treating as newline-separated format")
        log_lines = log.split("\n")
    except Exception as e:
        logger.error("Failed to parse junit test log: %s", e)
        raise e

    result = {}
    for line in log_lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Remove outer quotes if present (from JSON string format)
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]

        # Skip lines that don't look like CSV data
        if line.count(",") != 2:
            logger.debug(f"Skipping line (not 3 columns): {line}")
            continue

        if line == "Test-class-name,Test-name,status":
            logger.debug(f"Skipping header line: {line}")
            continue

        # Parse test result lines with format: "package.class,test_name,status"
        parts = line.split(",")
        if len(parts) == 3:
            test_class, test_name, status = parts
            test_key = f'"{test_class}.{test_name}"'

            logger.debug(f"Found test result: {test_key} {status}")
            status_upper = status.upper()
            if status_upper == "PASS":
                result[test_key] = TestStatus.PASSED
            elif status_upper == "FAIL":
                result[test_key] = TestStatus.FAILED
            elif status_upper == "ERROR":
                result[test_key] = TestStatus.ERROR
            else:
                logger.warning(
                    f"Unknown test status: {status}. Setting test to SKIPPED"
                )
                result[test_key] = TestStatus.SKIPPED
        else:
            logger.warning(f"Invalid test result line: {line}")

    logger.info(f"Parsed {len(result)} test results")
    logger.debug(f"Test results: \n\t{result}")
    return {test_name: test_status.value for test_name, test_status in result.items()}


def parse_log_jest(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes --verbose flag.

    Args:
        log (str): log content
    Returns:
        dict: test case to test status mapping
    """
    test_status_map = {}

    pattern = r"^\s*(✓|✕|○)\s(.+?)(?:\s\((\d+\s*m?s)\))?$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status_symbol, test_name, _duration = match.groups()
            if status_symbol == "✓":
                test_status_map[test_name] = TestStatus.PASSED
            elif status_symbol == "✕":
                test_status_map[test_name] = TestStatus.FAILED
            elif status_symbol == "○":
                test_status_map[test_name] = TestStatus.SKIPPED
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_jest_json(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for test logs generated with Jest. Assumes the --json flag has been
    piped into JEST_JSON_JQ_TRANSFORM. Unlike --verbose, tests with the same name
    in different describe blocks print with different names.
    """
    test_status_map = {}

    pattern = r"^\[(PASSED|FAILED)\]\s(.+)$"

    for line in log.split("\n"):
        match = re.match(pattern, line.strip())
        if match:
            status, test_name = match.groups()
            if status == "PASSED":
                test_status_map[test_name] = TestStatus.PASSED
            elif status == "FAILED":
                test_status_map[test_name] = TestStatus.FAILED
    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_log_mocha_json(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for mocha test logs when run with --reporter json-stream.
    """

    statmap: dict[str, TestStatus] = {
        "pass": TestStatus.PASSED,
        "fail": TestStatus.FAILED,
    }

    def _parse(result: str, scenario: dict[str, str]) -> tuple[str, TestStatus] | None:
        if result in statmap:
            if "title" not in scenario or "file" not in scenario:
                return None

            # file (repo relative path + filename) is stable wrt to the test name
            ft_hash = hashlib.sha1(scenario["file"].encode("utf-8")).hexdigest()
            return f'[{ft_hash[:5]}] {scenario["title"]}', statmap[result]
        else:
            return None

    lines = log.split("\n")
    stats = []
    for line in lines:
        if not line:
            continue
        try:
            state = json.loads(line)
            if len(state) >= 2:
                stats.append(_parse(state[0], state[1]))
        except json.JSONDecodeError:
            continue
    return {ts[0]: ts[1].value for ts in stats if ts}


def parse_log_gotest_json(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for go test logs when run with --reporter json-stream.
    """

    statmap: dict[str, TestStatus] = {
        "pass": TestStatus.PASSED,
        "fail": TestStatus.FAILED,
    }

    def _stat(state: dict[str, str]) -> tuple[str, TestStatus] | None:
        action = state.get("Action")
        if action in statmap and "Test" in state:
            assert "Package" in state
            assert action is not None

            full_path = f"{state['Package']}/{state['Test']}"
            return full_path, statmap[action]
        else:
            return None

    lines = log.split("\n")
    stats = []
    for line in lines:
        try:
            state = json.loads(line)
        except json.JSONDecodeError:
            continue
        stats.append(_stat(state))
    return {ts[0]: ts[1].value for ts in stats if ts}


def parse_log_check_framework(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for C Check framework logs when run with various build systems.
    Handles multiple output formats including CTest, Check framework, and make check.
    """

    test_status_map: dict[str, TestStatus] = {}

    lines = log.split("\n")

    # Parse CTest output with final summary (most common for CMake projects)
    for line in lines:
        line = line.strip()

        # CTest individual test results
        ctest_match = re.match(
            r"^\d+/\d+\s+Test\s+#\d+:\s+(.+?)\s+\.+\s+(Passed|Failed)", line
        )
        if ctest_match:
            test_name, status = ctest_match.groups()
            if status == "Passed":
                test_status_map[test_name] = TestStatus.PASSED
            else:
                test_status_map[test_name] = TestStatus.FAILED
            continue

        # Check framework detailed output with individual test results
        check_detail_match = re.match(r"^(.+?):(P|F|E):(.*)$", line)
        if check_detail_match:
            test_name, status, _ = check_detail_match.groups()
            if status == "P":
                test_status_map[test_name] = TestStatus.PASSED
            elif status == "F":
                test_status_map[test_name] = TestStatus.FAILED
            else:  # E for Error
                test_status_map[test_name] = TestStatus.ERROR
            continue

        # Standardized format output (from format_test_output bash function)
        # Format: "PASS: test_name", "FAIL: test_name", "ERROR: test_name"
        standardized_match = re.match(r"^(PASS|FAIL|ERROR):\s+(.+)$", line)
        if standardized_match:
            status, test_name = standardized_match.groups()
            if status == "PASS":
                test_status_map[test_name] = TestStatus.PASSED
            elif status == "FAIL":
                test_status_map[test_name] = TestStatus.FAILED
            else:  # ERROR
                test_status_map[test_name] = TestStatus.ERROR
            continue

        # Legacy make test output with PASS/FAIL indicators
        make_test_match = re.match(
            r"^(PASS|FAIL|passed|failed).*?:\s*(.+)$", line, re.IGNORECASE
        )
        if make_test_match:
            status, test_name = make_test_match.groups()
            if status.lower() in ["pass", "passed"]:
                test_status_map[test_name] = TestStatus.PASSED
            else:
                test_status_map[test_name] = TestStatus.FAILED
            continue

        # Simple test executable output
        simple_test_match = re.match(
            r"^Test\s+(.+?)\s+(passed|failed)$", line, re.IGNORECASE
        )
        if simple_test_match:
            test_name, status = simple_test_match.groups()
            if status.lower() == "passed":
                test_status_map[test_name] = TestStatus.PASSED
            else:
                test_status_map[test_name] = TestStatus.FAILED
            continue

    # If no individual test results found, try to parse summary information
    if not test_status_map:
        log_content = "\n".join(lines)

        # Parse Check framework summary (e.g., "Checks: 15, Failures: 2, Errors: 1")
        check_summary_match = re.search(
            r"Checks:\s*(\d+),\s*Failures:\s*(\d+),\s*Errors:\s*(\d+)", log_content
        )
        if check_summary_match:
            total_checks, failures, errors = map(int, check_summary_match.groups())
            passed_checks = total_checks - failures - errors

            # Create synthetic test names when only summary is available
            for i in range(passed_checks):
                test_status_map[f"test_{i+1}"] = TestStatus.PASSED
            for i in range(failures):
                test_status_map[f"test_{passed_checks + i + 1}"] = TestStatus.FAILED
            for i in range(errors):
                test_status_map[f"test_{passed_checks + failures + i + 1}"] = (
                    TestStatus.ERROR
                )

        # Parse CTest summary (e.g., "100% tests passed, 0 tests failed out of 3")
        elif re.search(r"\d+%\s+tests\s+passed.*out\s+of\s+\d+", log_content):
            ctest_summary_match = re.search(
                r"(\d+)%\s+tests\s+passed,\s*(\d+)\s+tests\s+failed\s+out\s+of\s+(\d+)",
                log_content,
            )
            if ctest_summary_match:
                _, failed_count, total_count = map(int, ctest_summary_match.groups())
                passed_count = total_count - failed_count

                # Create synthetic test names when only summary is available
                for i in range(passed_count):
                    test_status_map[f"test_{i+1}"] = TestStatus.PASSED
                for i in range(failed_count):
                    test_status_map[f"test_{passed_count + i + 1}"] = TestStatus.FAILED

        # Parse Make check output with TOTAL/PASS/FAIL/ERROR summary
        elif re.search(r"#\s+TOTAL:\s*\d+", log_content) and re.search(
            r"#\s+PASS:\s*\d+", log_content
        ):
            total_match = re.search(r"#\s+TOTAL:\s*(\d+)", log_content)
            pass_match = re.search(r"#\s+PASS:\s*(\d+)", log_content)
            fail_match = re.search(r"#\s+FAIL:\s*(\d+)", log_content)
            error_match = re.search(r"#\s+ERROR:\s*(\d+)", log_content)

            if total_match and pass_match:
                passed_tests = int(pass_match.group(1))
                failed_tests = int(fail_match.group(1)) if fail_match else 0
                error_tests = int(error_match.group(1)) if error_match else 0

                # Create synthetic test names when only summary is available
                for i in range(passed_tests):
                    test_status_map[f"test_{i+1}"] = TestStatus.PASSED
                for i in range(failed_tests):
                    test_status_map[f"test_{passed_tests + i + 1}"] = TestStatus.FAILED
                for i in range(error_tests):
                    test_status_map[f"test_{passed_tests + failed_tests + i + 1}"] = (
                        TestStatus.ERROR
                    )

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def _parse_junit_xml(xml_content: str) -> dict[str, str]:
    """
    Internal helper to parse JUnit XML format into test status map.
    Handles both <testsuites> and <testsuite> as root elements.

    Args:
        xml_content: The XML content string to parse

    Returns:
        Dictionary mapping test names to their status.
        Returns empty dict if XML is corrupted (contains system-out instead of proper test info)
    """
    test_status_map: dict[str, TestStatus] = {}

    try:
        # Parse the XML content
        root = ET.fromstring(xml_content.strip())

        # Handle both <testsuites> and <testsuite> as root elements
        if root.tag == "testsuites":
            # Find all testsuite elements within testsuites
            testsuites = root.findall(".//testsuite")
        elif root.tag == "testsuite":
            # Single testsuite as root
            testsuites = [root]
        else:
            # Not a valid JUnit XML format
            return {
                test_name: test_status.value
                for test_name, test_status in test_status_map.items()
            }

        # Process each testsuite
        for testsuite in testsuites:
            # Find all testcase elements
            for testcase in testsuite.findall("testcase"):
                # Check if XML is corrupted (contains system-out instead of proper test info)
                # In corrupted XML, testcase elements contain system-out with test output
                # instead of proper test results
                system_out_elements = testcase.findall("system-out")
                if system_out_elements:
                    # XML is corrupted, return empty dict to try other parsers
                    return {}

                # Get test name, preferring classname.name format if classname exists
                test_name = testcase.get("name", "")
                classname = testcase.get("classname", "")

                # Create full test name
                if classname:
                    full_test_name = f"{classname}.{test_name}"
                else:
                    full_test_name = test_name

                # Check for failure elements to determine status
                failure_elements = testcase.findall("failure")
                error_elements = testcase.findall("error")
                skipped_elements = testcase.findall("skipped")

                if failure_elements or error_elements:
                    test_status_map[full_test_name] = TestStatus.FAILED
                elif skipped_elements:
                    test_status_map[full_test_name] = TestStatus.SKIPPED
                else:
                    # No failure, error, or skipped elements means the test passed
                    test_status_map[full_test_name] = TestStatus.PASSED

    except ET.ParseError:
        # If XML parsing fails, return empty dict
        pass
    except Exception:
        # Handle any other parsing errors
        pass

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_cpp_test_log(log: str) -> dict[str, str]:
    """
    Unified parser for C++ test logs (Google Test, Catch2, etc.).
    Handles both JUnit XML output and plain text formats.

    The log may contain build output followed by XML content, or plain text
    test results. All formats are parsed using a unified list of parsers.

    Args:
        log: The test log output string

    Returns:
        Dictionary mapping test names to their status.
        Order is preserved as tests appear in the log (Python 3.7+ dict insertion order).
    """
    # Try all parsers until we get valid test results
    # Note: XML parser is tried first as it's the most structured format
    # Autotools parser should be tried before Catch2 because both handle
    # PASS:/FAIL: format, but Autotools can extract suite name
    # Parser order is maintained in the list below - parsers are tried sequentially
    parsers = [
        parse_junit_xml_format,
        parse_google_test_format,
        parse_autotools_format,
        parse_catch2_format,
        parse_tap_format,
        parse_boost_test_format,
        parse_ctest_format,
        parse_doctest_format,
    ]

    for parser in parsers:
        try:
            test_status_map = parser(log)
            # If we got valid test results, break
            if test_status_map:
                return test_status_map
        except Exception:
            # If parser fails, try the next one
            continue

    return {}


def parse_junit_xml_format(log_content: str) -> dict[str, str]:
    """
    Parse JUnit XML format from log content.
    Extracts and parses XML content that may be embedded in build output.

    Args:
        log_content: The log content that may contain XML

    Returns:
        Dictionary mapping test names to their status, or empty dict if no XML found
    """
    # Look for XML content in the log
    xml_start = log_content.find("<?xml")
    if xml_start == -1:
        return {}

    # Extract XML content starting from <?xml
    xml_content = log_content[xml_start:]

    # Find the end of the XML by looking for the closing tag of the root element
    # Handle both <testsuites> and <testsuite> as possible root elements
    if "<testsuites" in xml_content:
        xml_end = xml_content.rfind("</testsuites>")
        if xml_end != -1:
            xml_content = xml_content[: xml_end + len("</testsuites>")]
    elif "<testsuite" in xml_content:
        xml_end = xml_content.rfind("</testsuite>")
        if xml_end != -1:
            xml_content = xml_content[: xml_end + len("</testsuite>")]

    # Parse the extracted XML
    return _parse_junit_xml(xml_content)


def parse_log_google_test(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for Google Test logs in JUnit XML format.
    Handles Google Test XML output with testsuites and testcase elements.
    The log may contain build output followed by XML content, or plain text
    test results in PASS:/FAIL: format.
    """
    return parse_cpp_test_log(log)


def parse_log_catch2(log: str, grading_spec: CodingAgentGradingSpec) -> dict[str, str]:
    """
    Parser for Catch2 test logs in JUnit XML format.
    Handles Catch2 XML output with testsuites and testcase elements.
    The log may contain build output followed by XML content, or plain text
    test results in PASS:/FAIL: format.
    """
    return parse_cpp_test_log(log)


def parse_google_test_format(stdout_content: str) -> dict[str, str]:
    """
    Parse Google Test stdout format directly to test status map.
    Parses Google Test output that contains lines like:
    - [ RUN      ] TestSuite.TestName
    - [       OK ] TestSuite.TestName (1 ms)
    - [  FAILED  ] TestSuite.TestName (2 ms)

    Args:
        stdout_content: Google Test stdout output

    Returns:
        Dictionary mapping test names to their status
    """
    test_status_map: dict[str, TestStatus] = {}

    for line in stdout_content.splitlines():
        # Match "[       OK ] TestSuite.TestName" for passed tests
        ok_match = re.search(r"\[\s*OK\s*\]\s+(.+?)\s*\(", line)
        if ok_match:
            test_name = ok_match.group(1).strip()
            test_status_map[test_name] = TestStatus.PASSED
            continue

        # Match "[  FAILED  ] TestSuite.TestName" for failed tests
        failed_match = re.search(r"\[\s*FAILED\s*\]\s+(.+?)\s*\(", line)
        if failed_match:
            test_name = failed_match.group(1).strip()
            test_status_map[test_name] = TestStatus.FAILED
            continue

    # If no tests were parsed from individual lines, fall back to summary counts
    if not test_status_map:
        passed_count = 0
        failed_count = 0

        for line in stdout_content.splitlines():
            # Match "[  PASSED  ] N tests"
            match = re.search(r"\[  PASSED  \]\s+(\d+)\s+test", line)
            if match:
                passed_count += int(match.group(1))

            # Match "[  FAILED  ] N tests"
            match = re.search(r"\[  FAILED  \]\s+(\d+)\s+test", line)
            if match:
                failed_count += int(match.group(1))

        # Generate generic test names
        for i in range(passed_count):
            test_status_map[f"test_{i}"] = TestStatus.PASSED
        for i in range(passed_count, passed_count + failed_count):
            test_status_map[f"test_{i}"] = TestStatus.FAILED

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_catch2_format(stdout_content: str) -> dict[str, str]:
    """
    Parse Catch2/PASS:/FAIL: format directly to test status map.
    Parses PASS:/FAIL: lines from test output.

    Note: If the output contains "Testsuite summary", it's likely Autotools format
    and should be handled by parse_autotools_format instead.

    Args:
        stdout_content: The plain text test output containing PASS: and FAIL: lines

    Returns:
        Dictionary mapping test names to their status
    """
    # Check if this looks like Autotools output (has suite summary)
    # If so, return empty dict so Autotools parser can handle it
    if "Testsuite summary" in stdout_content:
        return {}

    test_status_map: dict[str, TestStatus] = {}

    for line in stdout_content.splitlines():
        # Match PASS: test_name
        if line.startswith("PASS:"):
            test_name = line[5:].strip()
            if test_name:
                test_status_map[test_name] = TestStatus.PASSED
        # Match FAIL: test_name
        elif line.startswith("FAIL:"):
            test_name = line[5:].strip()
            if test_name:
                test_status_map[test_name] = TestStatus.FAILED

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_autotools_format(stdout_content: str) -> dict[str, str]:
    """
    Parse Autotools test stdout format directly to test status map.
    Parses Autotools output that contains:
    - PASS: test_name / FAIL: test_name lines
    - Testsuite summary for <suite_name>
    - # TOTAL: N, # PASS: N, # FAIL: N, # ERROR: N

    Args:
        stdout_content: Autotools test stdout output

    Returns:
        Dictionary mapping test names to their status
    """
    test_status_map: dict[str, TestStatus] = {}
    extracted_suite_name: str | None = None

    # First pass: Extract suite name
    for line in stdout_content.splitlines():
        match = re.search(r"Testsuite summary for\s+(\S+)", line)
        if match:
            extracted_suite_name = match.group(1)
            break

    # Second pass: Parse test results using the extracted suite name
    for line in stdout_content.splitlines():
        # Match PASS: test_name
        if line.startswith("PASS:"):
            test_name = line[5:].strip()
            if test_name:
                full_name = (
                    f"{extracted_suite_name}.{test_name}"
                    if extracted_suite_name
                    else test_name
                )
                test_status_map[full_name] = TestStatus.PASSED
        # Match FAIL: test_name
        elif line.startswith("FAIL:"):
            test_name = line[5:].strip()
            if test_name:
                full_name = (
                    f"{extracted_suite_name}.{test_name}"
                    if extracted_suite_name
                    else test_name
                )
                test_status_map[full_name] = TestStatus.FAILED

    # If no individual tests found, fall back to parsing summary counts
    if not test_status_map:
        total_tests = 0
        passed_count = 0
        failed_count = 0

        for line in stdout_content.splitlines():
            match = re.match(r"^#\s+TOTAL:\s*(\d+)", line)
            if match:
                total_tests += int(match.group(1))

            match = re.match(r"^#\s+PASS:\s*(\d+)", line)
            if match:
                passed_count += int(match.group(1))

            match = re.match(r"^#\s+FAIL:\s*(\d+)", line)
            if match:
                failed_count += int(match.group(1))

            match = re.match(r"^#\s+ERROR:\s*(\d+)", line)
            if match:
                failed_count += int(match.group(1))

            match = re.match(r"^#\s+XPASS:\s*(\d+)", line)
            if match:
                failed_count += int(match.group(1))

        # Calculate passed tests if not explicitly specified
        if passed_count == 0 and total_tests > 0:
            passed_count = total_tests - failed_count

        # Generate generic test names with suite prefix if available
        suite_prefix = f"{extracted_suite_name}." if extracted_suite_name else ""
        for i in range(passed_count):
            test_status_map[f"{suite_prefix}test_{i}"] = TestStatus.PASSED
        for i in range(passed_count, passed_count + failed_count):
            test_status_map[f"{suite_prefix}test_{i}"] = TestStatus.FAILED

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_tap_format(stdout_content: str) -> dict[str, str]:
    """
    Parse TAP (Test Anything Protocol) format directly to test status map.
    Parses TAP output that contains lines like:
    - ok 1 test_name
    - not ok 2 test_name

    Args:
        stdout_content: TAP test stdout output

    Returns:
        Dictionary mapping test names to their status
    """
    test_status_map: dict[str, TestStatus] = {}

    for line in stdout_content.splitlines():
        # Match "ok N test_name" or "ok N"
        match = re.match(r"^ok\s+(\d+)\s*(.*)?$", line)
        if match:
            test_name = (
                match.group(2).strip() if match.group(2) else f"test_{match.group(1)}"
            )
            # Extract only the test name (before the colon if present)
            if ":" in test_name:
                test_name = test_name.split(":")[0].strip()
            # Remove leading "- " if present
            if test_name.startswith("- "):
                test_name = test_name[2:].strip()
            test_status_map[test_name] = TestStatus.PASSED

        # Match "not ok N test_name" or "not ok N"
        match = re.match(r"^not\s+ok\s+(\d+)\s*(.*)?$", line)
        if match:
            test_name = (
                match.group(2).strip() if match.group(2) else f"test_{match.group(1)}"
            )
            # Extract only the test name (before the colon if present)
            if ":" in test_name:
                test_name = test_name.split(":")[0].strip()
            # Remove leading "- " if present
            if test_name.startswith("- "):
                test_name = test_name[2:].strip()
            test_status_map[test_name] = TestStatus.FAILED

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_boost_test_format(stdout_content: str) -> dict[str, str]:
    """
    Parse Boost.Test stdout format directly to test status map.
    Parses Boost.Test output that contains lines like:
    - Running N test cases...
    - test_file.cpp(line): error: in "SuiteName/TestName": assertion failed
    - *** N failures detected
    - *** No errors detected

    Args:
        stdout_content: Boost.Test stdout output

    Returns:
        Dictionary mapping test names to their status
    """
    test_status_map: dict[str, TestStatus] = {}
    total_tests = 0
    failed_tests: list[tuple[str, str]] = []  # List of (suite_name, test_name) tuples
    default_suite_name = "BoostTests"

    for line in stdout_content.splitlines():
        # Match "Running N test cases..."
        match = re.search(r"Running\s+(\d+)\s+test\s+case", line)
        if match:
            total_tests = int(match.group(1))

        # Extract suite name from error messages: 'in "SuiteName/TestName"' or 'in "TestName"'
        match = re.search(r'in\s+"([^"]+)"', line)
        if match:
            full_name = match.group(1)
            # Check if it contains a suite name (format: Suite/Test)
            if "/" in full_name:
                suite, test = full_name.split("/", 1)
                failed_tests.append((suite, test))
            else:
                failed_tests.append((default_suite_name, full_name))

        # Extract default suite name from summary line if present
        match = re.search(r'test suite\s+"([^"]+)"', line)
        if match:
            default_suite_name = match.group(1)

    # Add failed tests to the map
    for suite, test in failed_tests:
        test_status_map[f"{suite}.{test}"] = TestStatus.FAILED

    # Add passed tests as generic names
    passed_count = total_tests - len(failed_tests)
    # Determine which suite to use for passed tests
    if failed_tests and len({suite for suite, _ in failed_tests}) == 1:
        # All failed tests from same suite, use that suite for passed tests
        passed_suite = failed_tests[0][0]
    else:
        # Use default suite for passed tests
        passed_suite = default_suite_name

    for i in range(passed_count):
        test_status_map[f"{passed_suite}.test_{i}"] = TestStatus.PASSED

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_doctest_format(stdout_content: str) -> dict[str, str]:
    """
    Parse doctest stdout format directly to test status map.
    Parses doctest output that contains lines like:
    - [doctest] test cases:  N |  N passed | N failed | N skipped

    Args:
        stdout_content: doctest stdout output

    Returns:
        Dictionary mapping test names to their status
    """
    test_status_map: dict[str, TestStatus] = {}
    passed_tests = 0
    failed_tests = 0

    for line in stdout_content.splitlines():
        # Match "[doctest] test cases:  N |  N passed | N failed"
        match = re.search(
            r"test\s+cases:\s+(\d+)\s+\|\s+(\d+)\s+passed\s+\|\s+(\d+)\s+failed", line
        )
        if match:
            passed_tests = int(match.group(2))
            failed_tests = int(match.group(3))

    # Generate test names
    for i in range(passed_tests):
        test_status_map[f"test_{i}"] = TestStatus.PASSED
    for i in range(passed_tests, passed_tests + failed_tests):
        test_status_map[f"test_{i}"] = TestStatus.FAILED

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def parse_ctest_format(stdout_content: str) -> dict[str, str]:
    """
    Parse CTest text stdout format directly to test status map.
    Parses CTest output that contains lines like:
    - 1/3 Test #1: test_foo .........................   Passed    0.02 sec
    - 2/3 Test #2: test_bar .........................***Failed    0.01 sec

    Args:
        stdout_content: CTest text stdout output

    Returns:
        Dictionary mapping test names to their status
    """
    test_status_map: dict[str, TestStatus] = {}

    for line in stdout_content.splitlines():
        # Match test result lines: "N/M Test #N: name ... Passed" or "***Failed"
        match = re.search(r"Test\s+\#\d+:\s+(\S+).*Passed", line)
        if match:
            test_status_map[match.group(1)] = TestStatus.PASSED

        match = re.search(r"Test\s+\#\d+:\s+(\S+).*\*\*\*Failed", line)
        if match:
            test_status_map[match.group(1)] = TestStatus.FAILED

    # Check for summary line if no individual tests found
    if not test_status_map:
        for line in stdout_content.splitlines():
            # Match "N tests failed out of M"
            match = re.search(r"(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)", line)
            if match:
                failed_count = int(match.group(1))
                total_count = int(match.group(2))
                passed_count = total_count - failed_count
                for i in range(passed_count):
                    test_status_map[f"test_{i}"] = TestStatus.PASSED
                for i in range(passed_count, total_count):
                    test_status_map[f"test_{i}"] = TestStatus.FAILED
                break

            # Match "100% tests passed, 0 tests failed out of N"
            match = re.search(r"100%\s+tests\s+passed.*out\s+of\s+(\d+)", line)
            if match:
                total_count = int(match.group(1))
                for i in range(total_count):
                    test_status_map[f"test_{i}"] = TestStatus.PASSED
                break

    return {
        test_name: test_status.value
        for test_name, test_status in test_status_map.items()
    }


def convert_catch2_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "Catch2Tests",
) -> str:
    """
    Convert plain text test output to JUnit XML format.
    Parses PASS:/FAIL: lines from autotools/Make test output.

    Args:
        stdout_content: The plain text test output containing PASS: and FAIL: lines
        test_suite_name: Name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string

    Example input:
        PASS: test_addition
        PASS: test_subtraction
        FAIL: test_multiplication

    Example output:
        <?xml version="1.0" encoding="UTF-8"?>
        <testsuites tests="3" failures="1" errors="0" name="Catch2Tests">
          <testsuite name="Catch2Tests" tests="3" failures="1" errors="0">
            <testcase name="test_addition" classname="Catch2Tests" time="0"/>
            <testcase name="test_subtraction" classname="Catch2Tests" time="0"/>
            <testcase name="test_multiplication" classname="Catch2Tests" time="0">
              <failure message="Test failed">test_multiplication failed</failure>
            </testcase>
          </testsuite>
        </testsuites>
    """
    passed_tests: list[str] = []
    failed_tests: list[str] = []

    # Parse test results from stdout content
    for line in stdout_content.splitlines():
        # Match PASS: test_name
        if line.startswith("PASS:"):
            test_name = line[5:].strip()
            if test_name:
                passed_tests.append(test_name)
        # Match FAIL: test_name
        elif line.startswith("FAIL:"):
            test_name = line[5:].strip()
            if test_name:
                failed_tests.append(test_name)

    # Calculate totals
    total_tests = len(passed_tests) + len(failed_tests)

    # Build JUnit XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites tests="{total_tests}" failures="{len(failed_tests)}" errors="0" name="{html.escape(test_suite_name)}">',
        f'  <testsuite name="{html.escape(test_suite_name)}" tests="{total_tests}" failures="{len(failed_tests)}" errors="0">',
    ]

    # Add passed tests
    for test_name in passed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(f'    <testcase name="{sanitized_name}" time="0"/>')

    # Add failed tests
    for test_name in failed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(f'    <testcase name="{sanitized_name}" time="0">')
        xml_lines.append(
            f'      <failure message="Test failed">{sanitized_name} failed</failure>'
        )
        xml_lines.append("    </testcase>")

    xml_lines.append("  </testsuite>")
    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def convert_google_test_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "GoogleTests",
) -> str:
    """
    Convert Google Test stdout format to JUnit XML format.
    Parses Google Test output that contains lines like:
    - [ RUN      ] TestSuite.TestName
    - [       OK ] TestSuite.TestName (1 ms)
    - [  FAILED  ] TestSuite.TestName (2 ms)

    Args:
        stdout_content: Google Test stdout output
        test_suite_name: Name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string
    """
    passed_tests: list[str] = []
    failed_tests: list[str] = []

    for line in stdout_content.splitlines():
        # Match "[       OK ] TestSuite.TestName" for passed tests
        ok_match = re.search(r"\[\s*OK\s*\]\s+(.+?)\s*\(", line)
        if ok_match:
            test_name = ok_match.group(1).strip()
            passed_tests.append(test_name)
            continue

        # Match "[  FAILED  ] TestSuite.TestName" for failed tests
        failed_match = re.search(r"\[\s*FAILED\s*\]\s+(.+?)\s*\(", line)
        if failed_match:
            test_name = failed_match.group(1).strip()
            failed_tests.append(test_name)
            continue

    # If no tests were parsed from individual lines, fall back to summary counts
    if not passed_tests and not failed_tests:
        passed_count = 0
        failed_count = 0

        for line in stdout_content.splitlines():
            # Match "[  PASSED  ] N tests" (with or without numeric prefix)
            match = re.search(r"\[  PASSED  \]\s+(\d+)\s+test", line)
            if match:
                passed_count += int(match.group(1))

            # Match "[  FAILED  ] N tests" (with or without numeric prefix)
            match = re.search(r"\[  FAILED  \]\s+(\d+)\s+test", line)
            if match:
                failed_count += int(match.group(1))

        # Generate generic test names
        passed_tests = [f"test_{i}" for i in range(passed_count)]
        failed_tests = [
            f"test_{i}" for i in range(passed_count, passed_count + failed_count)
        ]

    total_tests = len(passed_tests) + len(failed_tests)

    # Build JUnit XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites tests="{total_tests}" failures="{len(failed_tests)}" errors="0" name="{html.escape(test_suite_name)}">',
        f'  <testsuite name="{html.escape(test_suite_name)}" tests="{total_tests}" failures="{len(failed_tests)}" errors="0">',
    ]

    # Add passed tests
    for test_name in passed_tests:
        # Split test name into suite and test if it contains '.'
        if "." in test_name:
            suite_name, test_part = test_name.rsplit(".", 1)
            xml_lines.append(
                f'    <testcase name="{html.escape(test_part)}" classname="{html.escape(suite_name)}" time="0"/>'
            )
        else:
            xml_lines.append(
                f'    <testcase name="{html.escape(test_name)}" classname="{html.escape(test_suite_name)}" time="0"/>'
            )

    # Add failed tests
    for test_name in failed_tests:
        # Split test name into suite and test if it contains '.'
        if "." in test_name:
            suite_name, test_part = test_name.rsplit(".", 1)
            xml_lines.append(
                f'    <testcase name="{html.escape(test_part)}" classname="{html.escape(suite_name)}" time="0">'
            )
            xml_lines.append(
                f'      <failure message="Test failed">{html.escape(test_name)} failed</failure>'
            )
            xml_lines.append("    </testcase>")
        else:
            xml_lines.append(
                f'    <testcase name="{html.escape(test_name)}" classname="{html.escape(test_suite_name)}" time="0">'
            )
            xml_lines.append(
                f'      <failure message="Test failed">{html.escape(test_name)} failed</failure>'
            )
            xml_lines.append("    </testcase>")

    xml_lines.append("  </testsuite>")
    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def convert_autotools_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "AutotoolsTests",
) -> str:
    """
    Convert Autotools test stdout format to JUnit XML format.
    Parses Autotools output that contains:
    - PASS: test_name / FAIL: test_name lines
    - Testsuite summary for <suite_name>
    - # TOTAL: N, # PASS: N, # FAIL: N, # ERROR: N

    Args:
        stdout_content: Autotools test stdout output
        test_suite_name: Default name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string
    """
    passed_tests: list[str] = []
    failed_tests: list[str] = []
    extracted_suite_name: str | None = None

    # Parse test results and extract suite name
    for line in stdout_content.splitlines():
        # Extract test suite name from "Testsuite summary for <name> <version>"
        match = re.search(r"Testsuite summary for\s+(\S+)", line)
        if match:
            extracted_suite_name = match.group(1)

        # Match PASS: test_name
        if line.startswith("PASS:"):
            test_name = line[5:].strip()
            if test_name:
                passed_tests.append(test_name)
        # Match FAIL: test_name
        elif line.startswith("FAIL:"):
            test_name = line[5:].strip()
            if test_name:
                failed_tests.append(test_name)

    # Use extracted suite name if found, otherwise use default
    suite_name = extracted_suite_name if extracted_suite_name else test_suite_name

    # If no individual tests found, fall back to parsing summary counts
    if not passed_tests and not failed_tests:
        total_tests = 0
        passed_count = 0
        failed_count = 0

        for line in stdout_content.splitlines():
            match = re.match(r"^#\s+TOTAL:\s*(\d+)", line)
            if match:
                total_tests += int(match.group(1))

            match = re.match(r"^#\s+PASS:\s*(\d+)", line)
            if match:
                passed_count += int(match.group(1))

            match = re.match(r"^#\s+FAIL:\s*(\d+)", line)
            if match:
                failed_count += int(match.group(1))

            match = re.match(r"^#\s+ERROR:\s*(\d+)", line)
            if match:
                failed_count += int(match.group(1))

            match = re.match(r"^#\s+XPASS:\s*(\d+)", line)
            if match:
                failed_count += int(match.group(1))

        # Calculate passed tests if not explicitly specified
        if passed_count == 0 and total_tests > 0:
            passed_count = total_tests - failed_count

        # Generate generic test names
        passed_tests = [f"test_{i}" for i in range(passed_count)]
        failed_tests = [
            f"test_{i}" for i in range(passed_count, passed_count + failed_count)
        ]

    total_tests = len(passed_tests) + len(failed_tests)

    # Build JUnit XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites tests="{total_tests}" failures="{len(failed_tests)}" errors="0" name="{html.escape(suite_name)}">',
        f'  <testsuite name="{html.escape(suite_name)}" tests="{total_tests}" failures="{len(failed_tests)}" errors="0">',
    ]

    # Add passed tests
    for test_name in passed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(
            f'    <testcase name="{sanitized_name}" classname="{html.escape(suite_name)}" time="0"/>'
        )

    # Add failed tests
    for test_name in failed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(
            f'    <testcase name="{sanitized_name}" classname="{html.escape(suite_name)}" time="0">'
        )
        xml_lines.append(
            f'      <failure message="Test failed">{sanitized_name} failed</failure>'
        )
        xml_lines.append("    </testcase>")

    xml_lines.append("  </testsuite>")
    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def convert_tap_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "TAPTests",
) -> str:
    """
    Convert TAP (Test Anything Protocol) format to JUnit XML format.
    Parses TAP output that contains lines like:
    - ok 1 test_name
    - not ok 2 test_name

    Args:
        stdout_content: TAP test stdout output
        test_suite_name: Name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string
    """
    passed_tests: list[str] = []
    failed_tests: list[str] = []

    for line in stdout_content.splitlines():
        # Match "ok N test_name" or "ok N"
        match = re.match(r"^ok\s+(\d+)\s*(.*)?$", line)
        if match:
            test_name = (
                match.group(2).strip() if match.group(2) else f"test_{match.group(1)}"
            )
            # Extract only the test name (before the colon if present)
            # e.g., "- test_addition: Adding two positive numbers" -> "test_addition"
            if ":" in test_name:
                test_name = test_name.split(":")[0].strip()
            # Remove leading "- " if present
            if test_name.startswith("- "):
                test_name = test_name[2:].strip()
            passed_tests.append(test_name)

        # Match "not ok N test_name" or "not ok N"
        match = re.match(r"^not\s+ok\s+(\d+)\s*(.*)?$", line)
        if match:
            test_name = (
                match.group(2).strip() if match.group(2) else f"test_{match.group(1)}"
            )
            # Extract only the test name (before the colon if present)
            # e.g., "- test_addition: Adding two positive numbers" -> "test_addition"
            if ":" in test_name:
                test_name = test_name.split(":")[0].strip()
            # Remove leading "- " if present
            if test_name.startswith("- "):
                test_name = test_name[2:].strip()
            failed_tests.append(test_name)

    total_tests = len(passed_tests) + len(failed_tests)

    # Build JUnit XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites tests="{total_tests}" failures="{len(failed_tests)}" errors="0" name="{html.escape(test_suite_name)}">',
        f'  <testsuite name="{html.escape(test_suite_name)}" tests="{total_tests}" failures="{len(failed_tests)}" errors="0">',
    ]

    # Add passed tests
    for test_name in passed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(f'    <testcase name="{sanitized_name}" time="0"/>')

    # Add failed tests
    for test_name in failed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(f'    <testcase name="{sanitized_name}" time="0">')
        xml_lines.append(
            f'      <failure message="Test failed">{sanitized_name} failed</failure>'
        )
        xml_lines.append("    </testcase>")

    xml_lines.append("  </testsuite>")
    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def convert_boost_test_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "BoostTests",
) -> str:
    """
    Convert Boost.Test stdout format to JUnit XML format.
    Parses Boost.Test output that contains lines like:
    - Running N test cases...
    - test_file.cpp(line): error: in "SuiteName/TestName": assertion failed
    - *** N failures detected
    - *** No errors detected

    Args:
        stdout_content: Boost.Test stdout output
        test_suite_name: Name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string
    """
    total_tests = 0
    failed_tests: list[tuple[str, str]] = []  # List of (suite_name, test_name) tuples
    default_suite_name = test_suite_name

    for line in stdout_content.splitlines():
        # Match "Running N test cases..."
        match = re.search(r"Running\s+(\d+)\s+test\s+case", line)
        if match:
            total_tests = int(match.group(1))

        # Extract suite name from error messages: 'in "SuiteName/TestName"' or 'in "TestName"'
        match = re.search(r'in\s+"([^"]+)"', line)
        if match:
            full_name = match.group(1)
            # Check if it contains a suite name (format: Suite/Test)
            if "/" in full_name:
                suite, test = full_name.split("/", 1)
                failed_tests.append((suite, test))
            else:
                failed_tests.append((default_suite_name, full_name))

        # Extract default suite name from summary line if present
        match = re.search(r'test suite\s+"([^"]+)"', line)
        if match:
            default_suite_name = match.group(1)

    failed_count = len(failed_tests)
    passed_count = total_tests - failed_count

    # Group failed tests by suite
    failed_by_suite: dict[str, list[str]] = {}
    for suite, test in failed_tests:
        if suite not in failed_by_suite:
            failed_by_suite[suite] = []
        failed_by_suite[suite].append(test)

    # Build JUnit XML with multiple testsuites if we have tests from different suites
    if len(failed_by_suite) > 1 or (
        failed_by_suite and default_suite_name not in failed_by_suite
    ):
        # Multiple suites - create a testsuite for each
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuites tests="{total_tests}" failures="{failed_count}" errors="0" name="BoostTests">',
        ]

        # Add testsuites for each suite with failed tests
        for suite_name, suite_tests in failed_by_suite.items():
            xml_lines.append(
                f'  <testsuite name="{html.escape(suite_name)}" tests="{len(suite_tests)}" failures="{len(suite_tests)}" errors="0">'
            )
            for test_name in suite_tests:
                sanitized_name = html.escape(test_name)
                xml_lines.append(
                    f'    <testcase name="{sanitized_name}" classname="{html.escape(suite_name)}" time="0">'
                )
                xml_lines.append(
                    f'      <failure message="Test failed">{sanitized_name} failed</failure>'
                )
                xml_lines.append("    </testcase>")
            xml_lines.append("  </testsuite>")

        # Add passed tests in a default suite if there are any
        if passed_count > 0:
            xml_lines.append(
                f'  <testsuite name="{html.escape(default_suite_name)}" tests="{passed_count}" failures="0" errors="0">'
            )
            for i in range(passed_count):
                xml_lines.append(
                    f'    <testcase name="test_{i}" classname="{html.escape(default_suite_name)}" time="0"/>'
                )
            xml_lines.append("  </testsuite>")

    else:
        # Single suite - use traditional format
        suite_name = default_suite_name
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            f'<testsuites tests="{total_tests}" failures="{failed_count}" errors="0" name="{html.escape(suite_name)}">',
            f'  <testsuite name="{html.escape(suite_name)}" tests="{total_tests}" failures="{failed_count}" errors="0">',
        ]

        # Add passed tests as generic test cases
        for i in range(passed_count):
            xml_lines.append(
                f'    <testcase name="test_{i}" classname="{html.escape(suite_name)}" time="0"/>'
            )

        # Add failed tests with their actual names
        for suite, test_name in failed_tests:
            sanitized_name = html.escape(test_name)
            xml_lines.append(
                f'    <testcase name="{sanitized_name}" classname="{html.escape(suite)}" time="0">'
            )
            xml_lines.append(
                f'      <failure message="Test failed">{sanitized_name} failed</failure>'
            )
            xml_lines.append("    </testcase>")

        xml_lines.append("  </testsuite>")

    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def convert_doctest_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "DoctestTests",
) -> str:
    """
    Convert doctest stdout format to JUnit XML format.
    Parses doctest output that contains lines like:
    - [doctest] test cases:  N |  N passed | N failed | N skipped

    Args:
        stdout_content: doctest stdout output
        test_suite_name: Name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string
    """
    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for line in stdout_content.splitlines():
        # Match "[doctest] test cases:  N |  N passed | N failed"
        match = re.search(
            r"test\s+cases:\s+(\d+)\s+\|\s+(\d+)\s+passed\s+\|\s+(\d+)\s+failed", line
        )
        if match:
            total_tests = int(match.group(1))
            passed_tests = int(match.group(2))
            failed_tests = int(match.group(3))

    # Build JUnit XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites tests="{total_tests}" failures="{failed_tests}" errors="0" name="{html.escape(test_suite_name)}">',
        f'  <testsuite name="{html.escape(test_suite_name)}" tests="{total_tests}" failures="{failed_tests}" errors="0">',
    ]

    # Add passed tests as generic test cases
    for i in range(passed_tests):
        xml_lines.append(f'    <testcase name="test_{i}" time="0"/>')

    # Add failed tests as generic test cases
    for i in range(failed_tests):
        xml_lines.append(f'    <testcase name="test_{passed_tests + i}" time="0">')
        xml_lines.append(
            f'      <failure message="Test failed">test_{passed_tests + i} failed</failure>'
        )
        xml_lines.append("    </testcase>")

    xml_lines.append("  </testsuite>")
    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def convert_ctest_text_stdout_to_junit_xml(
    stdout_content: str,
    grading_spec: CodingAgentGradingSpec,
    test_suite_name: str = "CTestTests",
) -> str:
    """
    Convert CTest text stdout format to JUnit XML format.
    Parses CTest output that contains lines like:
    - 1/3 Test #1: test_foo .........................   Passed    0.02 sec
    - 2/3 Test #2: test_bar .........................***Failed    0.01 sec

    Args:
        stdout_content: CTest text stdout output
        test_suite_name: Name of the test suite for the XML output

    Returns:
        A JUnit XML formatted string
    """
    passed_tests: list[str] = []
    failed_tests: list[str] = []

    for line in stdout_content.splitlines():
        # Match test result lines: "N/M Test #N: name ... Passed" or "***Failed"
        match = re.search(r"Test\s+\#\d+:\s+(\S+).*Passed", line)
        if match:
            passed_tests.append(match.group(1))

        match = re.search(r"Test\s+\#\d+:\s+(\S+).*\*\*\*Failed", line)
        if match:
            failed_tests.append(match.group(1))

    # Check for summary line if no individual tests found
    if not passed_tests and not failed_tests:
        for line in stdout_content.splitlines():
            # Match "N tests failed out of M"
            match = re.search(r"(\d+)\s+tests?\s+failed\s+out\s+of\s+(\d+)", line)
            if match:
                failed_count = int(match.group(1))
                total_count = int(match.group(2))
                passed_count = total_count - failed_count
                passed_tests = [f"test_{i}" for i in range(passed_count)]
                failed_tests = [f"test_{i}" for i in range(passed_count, total_count)]
                break

            # Match "100% tests passed, 0 tests failed out of N"
            match = re.search(r"100%\s+tests\s+passed.*out\s+of\s+(\d+)", line)
            if match:
                total_count = int(match.group(1))
                passed_tests = [f"test_{i}" for i in range(total_count)]
                break

    total_tests = len(passed_tests) + len(failed_tests)

    # Build JUnit XML
    xml_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<testsuites tests="{total_tests}" failures="{len(failed_tests)}" errors="0" name="{html.escape(test_suite_name)}">',
        f'  <testsuite name="{html.escape(test_suite_name)}" tests="{total_tests}" failures="{len(failed_tests)}" errors="0">',
    ]

    # Add passed tests
    for test_name in passed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(f'    <testcase name="{sanitized_name}" time="0"/>')

    # Add failed tests
    for test_name in failed_tests:
        sanitized_name = html.escape(test_name)
        xml_lines.append(f'    <testcase name="{sanitized_name}" time="0">')
        xml_lines.append(
            f'      <failure message="Test failed">{sanitized_name} failed</failure>'
        )
        xml_lines.append("    </testcase>")

    xml_lines.append("  </testsuite>")
    xml_lines.append("</testsuites>")

    return "\n".join(xml_lines)


def parse_log_cargo_test(
    log: str, grading_spec: CodingAgentGradingSpec
) -> dict[str, str]:
    """
    Parser for Rust cargo test logs in human-readable format.

    Cargo test outputs lines like:
    - "test test_name ... ok" for passing tests
    - "test test_name ... FAILED" for failing tests
    - "test test_name ... ignored" for ignored tests

    Example output:
        running 4 tests
        test peer::media::transitable_state::test::cancel_transition ... ok
        test platform::transceiver::tests::enable_works_correctly ... FAILED

        failures:
            platform::transceiver::tests::enable_works_correctly

        test result: FAILED. 3 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
    """

    results: dict[str, TestStatus] = {}

    # Pattern to match test result lines: "test <test_name> ... <status>"
    # Example: "test my_test::nested::test_something ... ok"
    test_pattern = re.compile(
        r"^test\s+([^\s]+)\s+\.\.\.\s+(ok|FAILED|ignored)", re.MULTILINE
    )

    for match in test_pattern.finditer(log):
        test_name = match.group(1)
        status = match.group(2)

        if status == "ok":
            results[test_name] = TestStatus.PASSED
        elif status == "FAILED":
            results[test_name] = TestStatus.FAILED
        # Note: We don't include "ignored" tests in the results as they weren't executed

    return {test_name: test_status.value for test_name, test_status in results.items()}
