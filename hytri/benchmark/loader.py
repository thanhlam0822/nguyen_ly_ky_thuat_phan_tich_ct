"""BugsInPy Dataset Loader.

Loads and parses the BugsInPy benchmark dataset for failure triage evaluation.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hytri.models import Build, FailureInstance, GroundTruth, StackFrame


@dataclass
class BugInfo:
    """Information about a single bug in BugsInPy."""

    project: str
    bug_id: str
    buggy_commit: str
    fixed_commit: str
    test_file: str
    failing_tests: list[str]
    changed_files: list[str]


class BugsInPyLoader:
    """Loader for BugsInPy benchmark dataset.

    BugsInPy is a benchmark of real bugs from popular Python projects.
    Each bug has:
    - Buggy and fixed versions
    - Failing test cases
    - Changed files for the fix

    This loader can:
    - Load bug metadata
    - Run failing tests and capture stack traces
    - Create FailureInstance objects for evaluation
    """

    # Supported BugsInPy projects
    SUPPORTED_PROJECTS = [
        "ansible",
        "black",
        "cookiecutter",
        "fastapi",
        "httpie",
        "keras",
        "luigi",
        "matplotlib",
        "pandas",
        "PySnooper",
        "pytest",
        "sanic",
        "scrapy",
        "spacy",
        "thefuck",
        "tornado",
        "tqdm",
        "youtube-dl",
    ]

    def __init__(
        self,
        bugsinpy_path: Path | str | None = None,
        cache_dir: Path | str | None = None,
    ):
        """Initialize the loader.

        Args:
            bugsinpy_path: Path to BugsInPy repository.
            cache_dir: Directory to cache parsed data.
        """
        self.bugsinpy_path = Path(bugsinpy_path) if bugsinpy_path else None
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".hytri" / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def list_projects(self) -> list[str]:
        """List available projects."""
        if self.bugsinpy_path and self.bugsinpy_path.exists():
            projects_dir = self.bugsinpy_path / "projects"
            if projects_dir.exists():
                return [p.name for p in projects_dir.iterdir() if p.is_dir()]
        return self.SUPPORTED_PROJECTS

    def list_bugs(self, project: str) -> list[str]:
        """List bug IDs for a project."""
        if self.bugsinpy_path:
            bugs_dir = self.bugsinpy_path / "projects" / project / "bugs"
            if bugs_dir.exists():
                return sorted([b.name for b in bugs_dir.iterdir() if b.is_dir()])
        return []

    def load_bug_info(self, project: str, bug_id: str) -> BugInfo | None:
        """Load metadata for a specific bug.

        Args:
            project: Project name.
            bug_id: Bug ID (e.g., "1", "2").

        Returns:
            BugInfo if found, None otherwise.
        """
        if not self.bugsinpy_path:
            return None

        bug_dir = self.bugsinpy_path / "projects" / project / "bugs" / bug_id
        if not bug_dir.exists():
            return None

        # Parse bug.info file
        bug_info_file = bug_dir / "bug.info"
        if not bug_info_file.exists():
            return None

        info = self._parse_bug_info(bug_info_file)
        info["project"] = project
        info["bug_id"] = bug_id

        return BugInfo(
            project=project,
            bug_id=bug_id,
            buggy_commit=info.get("buggy_commit_id", ""),
            fixed_commit=info.get("fixed_commit_id", ""),
            test_file=info.get("test_file", ""),
            failing_tests=info.get("failing_tests", []),
            changed_files=info.get("changed_files", []),
        )

    def _parse_bug_info(self, path: Path) -> dict:
        """Parse a bug.info file."""
        info = {}
        current_key = None

        with open(path) as f:
            for line in f:
                line = line.strip()
                if "=" in line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"')

                    if key in ("failing_tests", "changed_files"):
                        info[key] = [v.strip() for v in value.split(",") if v.strip()]
                    else:
                        info[key] = value
                    current_key = key
                elif current_key and line:
                    # Continuation of previous value
                    if isinstance(info.get(current_key), list):
                        info[current_key].extend(
                            [v.strip() for v in line.split(",") if v.strip()]
                        )

        return info

    def run_failing_tests(
        self,
        project: str,
        bug_id: str,
        timeout: int = 300,
    ) -> list[FailureInstance]:
        """Run failing tests for a bug and capture stack traces.

        This requires the project to be checked out at the buggy commit.

        Args:
            project: Project name.
            bug_id: Bug ID.
            timeout: Timeout in seconds.

        Returns:
            List of FailureInstance objects.
        """
        bug_info = self.load_bug_info(project, bug_id)
        if not bug_info:
            return []

        failures = []
        build_id = f"{project}_{bug_id}"

        for test_name in bug_info.failing_tests:
            # Run pytest and capture output
            result = self._run_pytest(test_name, timeout)

            if result["failed"]:
                failure = FailureInstance(
                    build_id=build_id,
                    failure_id=f"{build_id}::{test_name}",
                    test_name=test_name,
                    exception_type=result.get("exception_type", "UnknownError"),
                    exception_message=result.get("exception_message", ""),
                    stack_trace=result.get("stack_trace", ""),
                    stderr=result.get("stderr", ""),
                    stdout=result.get("stdout", ""),
                    project=project,
                    bug_ids=[bug_id],
                    ground_truth=GroundTruth(
                        root_cause_id=bug_id,
                        changed_files=bug_info.changed_files,
                    ),
                )
                failures.append(failure)

        return failures

    def _run_pytest(self, test_name: str, timeout: int) -> dict:
        """Run a single pytest test and capture output."""
        try:
            result = subprocess.run(
                ["pytest", test_name, "-v", "--tb=long"],
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            return {
                "failed": result.returncode != 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "stack_trace": self._extract_traceback(result.stdout + result.stderr),
                "exception_type": self._extract_exception_type(result.stdout + result.stderr),
                "exception_message": self._extract_exception_message(result.stdout + result.stderr),
            }
        except subprocess.TimeoutExpired:
            return {
                "failed": True,
                "exception_type": "TimeoutError",
                "exception_message": f"Test timed out after {timeout}s",
                "stack_trace": "",
                "stdout": "",
                "stderr": "",
            }
        except Exception as e:
            return {
                "failed": True,
                "exception_type": type(e).__name__,
                "exception_message": str(e),
                "stack_trace": "",
                "stdout": "",
                "stderr": "",
            }

    def _extract_traceback(self, output: str) -> str:
        """Extract traceback from pytest output."""
        lines = output.split("\n")
        in_traceback = False
        traceback_lines = []

        for line in lines:
            if "Traceback (most recent call last):" in line:
                in_traceback = True
                traceback_lines = [line]
            elif in_traceback:
                traceback_lines.append(line)
                # End of traceback detection
                if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                    if "Error" in line or "Exception" in line:
                        break

        return "\n".join(traceback_lines)

    def _extract_exception_type(self, output: str) -> str:
        """Extract exception type from output."""
        import re

        # Look for common exception patterns
        patterns = [
            r"(\w+Error):",
            r"(\w+Exception):",
            r"E\s+(\w+Error):",
            r"E\s+(\w+Exception):",
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1)

        return "UnknownError"

    def _extract_exception_message(self, output: str) -> str:
        """Extract exception message from output."""
        import re

        # Look for message after exception type
        patterns = [
            r"(?:\w+Error|\w+Exception):\s*(.+?)(?:\n|$)",
            r"E\s+(?:\w+Error|\w+Exception):\s*(.+?)(?:\n|$)",
        ]

        for pattern in patterns:
            match = re.search(pattern, output)
            if match:
                return match.group(1).strip()

        return ""

    def load_from_jsonl(self, path: Path | str) -> list[FailureInstance]:
        """Load failure instances from a JSONL file.

        This is the recommended way to load pre-captured failure data.

        Args:
            path: Path to JSONL file.

        Returns:
            List of FailureInstance objects.
        """
        path = Path(path)
        failures = []

        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)

                # Parse frames
                frames = [
                    StackFrame(**frame_data)
                    for frame_data in data.get("frames", [])
                ]

                # Parse ground truth
                gt_data = data.get("ground_truth")
                ground_truth = GroundTruth(**gt_data) if gt_data else None

                failure = FailureInstance(
                    build_id=data.get("build_id", ""),
                    failure_id=data.get("failure_id", ""),
                    test_name=data.get("test_name", ""),
                    exception_type=data.get("exception_type", ""),
                    exception_message=data.get("exception_message", ""),
                    stack_trace=data.get("stack_trace", ""),
                    frames=frames,
                    stderr=data.get("stderr", ""),
                    stdout=data.get("stdout", ""),
                    project=data.get("project", ""),
                    bug_ids=data.get("bug_ids", []),
                    ground_truth=ground_truth,
                )
                failures.append(failure)

        return failures

    def save_to_jsonl(
        self,
        failures: list[FailureInstance],
        path: Path | str,
    ) -> None:
        """Save failure instances to a JSONL file.

        Args:
            failures: List of failures to save.
            path: Output path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            for failure in failures:
                data = failure.model_dump()
                f.write(json.dumps(data, default=str) + "\n")

    def load_build_from_jsonl(self, path: Path | str) -> Build:
        """Load build metadata from a JSONL file.

        Args:
            path: Path to builds JSONL file.

        Returns:
            Build object.
        """
        path = Path(path)

        with open(path) as f:
            line = f.readline().strip()
            data = json.loads(line)

            return Build(
                build_id=data.get("build_id", ""),
                project=data.get("project", ""),
                bug_ids=data.get("bug_ids", []),
                num_failures=data.get("num_failures", 0),
                num_tests_total=data.get("num_tests_total", 0),
                env=data.get("env", {}),
                notes=data.get("notes", ""),
            )


