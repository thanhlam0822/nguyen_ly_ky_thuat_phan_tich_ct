"""Multi-Fault Build Constructor.

Creates synthetic multi-fault builds by combining failures from multiple
bug versions. This creates realistic triage scenarios where multiple
root causes exist in a single build.
"""

import json
import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from hytri.models import Build, FailureInstance, GroundTruth


@dataclass
class MultiFaultBuildConfig:
    """Configuration for multi-fault build generation."""

    min_bugs_per_build: int = 2
    max_bugs_per_build: int = 5
    min_failures_per_bug: int = 1
    max_failures_per_bug: int = 10


class MultiFaultBuildBuilder:
    """Builds synthetic multi-fault builds for benchmarking.

    Takes individual bug failure sets and combines them into builds
    with multiple root causes for realistic triage evaluation.
    """

    def __init__(
        self,
        config: MultiFaultBuildConfig | None = None,
        seed: int | None = None,
    ):
        """Initialize the builder.

        Args:
            config: Configuration for build generation.
            seed: Random seed for reproducibility.
        """
        self.config = config or MultiFaultBuildConfig()
        self.rng = random.Random(seed)

    def build_from_bug_failures(
        self,
        bug_failures: dict[str, list[FailureInstance]],
        num_builds: int = 10,
    ) -> list[tuple[Build, list[FailureInstance]]]:
        """Create multi-fault builds from bug failure sets.

        Args:
            bug_failures: Dict of bug_id -> list of failures for that bug.
            num_builds: Number of builds to generate.

        Returns:
            List of (Build, failures) tuples.
        """
        bug_ids = list(bug_failures.keys())
        if len(bug_ids) < self.config.min_bugs_per_build:
            raise ValueError(
                f"Need at least {self.config.min_bugs_per_build} bugs, "
                f"but only have {len(bug_ids)}"
            )

        builds = []

        for i in range(num_builds):
            # Select bugs for this build
            num_bugs = self.rng.randint(
                self.config.min_bugs_per_build,
                min(self.config.max_bugs_per_build, len(bug_ids)),
            )
            selected_bugs = self.rng.sample(bug_ids, num_bugs)

            # Collect failures from each bug
            build_failures = []
            for bug_id in selected_bugs:
                available = bug_failures[bug_id]
                if not available:
                    continue

                # Sample failures from this bug
                num_to_sample = self.rng.randint(
                    self.config.min_failures_per_bug,
                    min(self.config.max_failures_per_bug, len(available)),
                )
                sampled = self.rng.sample(available, min(num_to_sample, len(available)))
                build_failures.extend(sampled)

            if not build_failures:
                continue

            # Create build
            build_id = f"synthetic_build_{i:04d}_{uuid.uuid4().hex[:8]}"

            # Update failure build_ids
            for failure in build_failures:
                failure.build_id = build_id
                failure.failure_id = f"{build_id}::{failure.test_name}"

            # Determine project (use first failure's project)
            project = build_failures[0].project if build_failures else "unknown"

            build = Build(
                build_id=build_id,
                project=project,
                bug_ids=selected_bugs,
                num_failures=len(build_failures),
                num_tests_total=len(build_failures) * 10,  # Estimate
                env={"python": "3.10", "os": "linux"},
                timestamp=datetime.now(),
                notes=f"Synthetic multi-fault build with {num_bugs} bugs",
            )

            builds.append((build, build_failures))

        return builds

    def build_from_single_bugs(
        self,
        single_bug_failures: list[list[FailureInstance]],
        num_builds: int = 10,
    ) -> list[tuple[Build, list[FailureInstance]]]:
        """Create multi-fault builds from lists of single-bug failures.

        Args:
            single_bug_failures: List of failure lists (one per bug).
            num_builds: Number of builds to generate.

        Returns:
            List of (Build, failures) tuples.
        """
        # Convert to bug_id -> failures mapping
        bug_failures = {}
        for i, failures in enumerate(single_bug_failures):
            if failures:
                # Use ground_truth bug_id if available, else generate
                bug_id = (
                    failures[0].ground_truth.root_cause_id
                    if failures[0].ground_truth
                    else f"bug_{i}"
                )
                bug_failures[bug_id] = failures

        return self.build_from_bug_failures(bug_failures, num_builds)

    def merge_failure_sets(
        self,
        failure_sets: list[list[FailureInstance]],
        build_id: str | None = None,
    ) -> tuple[Build, list[FailureInstance]]:
        """Merge multiple failure sets into a single build.

        Simple merge without sampling - includes all failures.

        Args:
            failure_sets: List of failure lists to merge.
            build_id: Optional build ID (generated if not provided).

        Returns:
            (Build, failures) tuple.
        """
        if not failure_sets:
            return Build(
                build_id=build_id or "empty_build",
                project="unknown",
                num_failures=0,
            ), []

        build_id = build_id or f"merged_build_{uuid.uuid4().hex[:8]}"
        all_failures = []
        bug_ids = set()

        for failures in failure_sets:
            for failure in failures:
                # Update build_id and failure_id
                failure.build_id = build_id
                failure.failure_id = f"{build_id}::{failure.test_name}"
                all_failures.append(failure)

                if failure.ground_truth:
                    bug_ids.add(failure.ground_truth.root_cause_id)

        project = all_failures[0].project if all_failures else "unknown"

        build = Build(
            build_id=build_id,
            project=project,
            bug_ids=list(bug_ids),
            num_failures=len(all_failures),
            timestamp=datetime.now(),
            notes="Merged failure sets",
        )

        return build, all_failures

    def save_benchmark(
        self,
        builds: list[tuple[Build, list[FailureInstance]]],
        output_dir: Path | str,
    ) -> None:
        """Save generated benchmark to disk.

        Creates:
        - failure_instances.jsonl: All failure instances
        - builds.jsonl: Build metadata

        Args:
            builds: List of (Build, failures) tuples.
            output_dir: Directory to save to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        failures_path = output_dir / "failure_instances.jsonl"
        builds_path = output_dir / "builds.jsonl"

        # Save failures
        with open(failures_path, "w") as f:
            for _, failures in builds:
                for failure in failures:
                    data = failure.model_dump()
                    f.write(json.dumps(data, default=str) + "\n")

        # Save builds
        with open(builds_path, "w") as f:
            for build, _ in builds:
                data = build.model_dump()
                f.write(json.dumps(data, default=str) + "\n")

    def load_benchmark(
        self,
        benchmark_dir: Path | str,
    ) -> list[tuple[Build, list[FailureInstance]]]:
        """Load a benchmark from disk.

        Args:
            benchmark_dir: Directory containing benchmark files.

        Returns:
            List of (Build, failures) tuples.
        """
        from hytri.benchmark.loader import BugsInPyLoader

        benchmark_dir = Path(benchmark_dir)
        loader = BugsInPyLoader()

        # Load failures
        failures_path = benchmark_dir / "failure_instances.jsonl"
        all_failures = loader.load_from_jsonl(failures_path)

        # Group by build_id
        build_failures: dict[str, list[FailureInstance]] = {}
        for failure in all_failures:
            if failure.build_id not in build_failures:
                build_failures[failure.build_id] = []
            build_failures[failure.build_id].append(failure)

        # Load builds
        builds_path = benchmark_dir / "builds.jsonl"
        builds = []

        with open(builds_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                build = Build(**data)
                failures = build_failures.get(build.build_id, [])
                builds.append((build, failures))

        return builds


def create_benchmark_from_bugs(
    bugs_data: dict[str, list[dict]],
    num_builds: int = 50,
    output_dir: Path | str | None = None,
    seed: int = 42,
) -> list[tuple[Build, list[FailureInstance]]]:
    """Convenience function to create a benchmark from bug data.

    Args:
        bugs_data: Dict of bug_id -> list of failure dicts.
        num_builds: Number of builds to generate.
        output_dir: Optional directory to save benchmark.
        seed: Random seed.

    Returns:
        List of (Build, failures) tuples.
    """
    # Convert to FailureInstance objects
    bug_failures = {}
    for bug_id, failures_data in bugs_data.items():
        failures = []
        for fd in failures_data:
            failure = FailureInstance(
                build_id=fd.get("build_id", bug_id),
                failure_id=fd.get("failure_id", f"{bug_id}::test"),
                test_name=fd.get("test_name", "test"),
                exception_type=fd.get("exception_type", "Error"),
                exception_message=fd.get("exception_message", ""),
                stack_trace=fd.get("stack_trace", ""),
                project=fd.get("project", "unknown"),
                ground_truth=GroundTruth(root_cause_id=bug_id),
            )
            failures.append(failure)
        bug_failures[bug_id] = failures

    # Build
    builder = MultiFaultBuildBuilder(seed=seed)
    builds = builder.build_from_bug_failures(bug_failures, num_builds)

    # Save if output_dir provided
    if output_dir:
        builder.save_benchmark(builds, output_dir)

    return builds


