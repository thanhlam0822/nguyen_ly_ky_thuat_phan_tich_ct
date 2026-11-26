#!/usr/bin/env python3
"""Export 30 Test builds (seed=42) + 15 disjoint Tuning builds (seed=123) into a lightweight JSON."""

import json
import random
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
OUT_FILE = DATA_DIR / "pilot_45_builds.json"

DATASETS = {
    "Defects4J-Real": DATA_DIR / "defects4j_multifault_realistic" / "synthetic_builds.jsonl",
    "Bears-Real": DATA_DIR / "bears_multifault_realistic" / "synthetic_builds.jsonl",
    "BugSwarm-Real": DATA_DIR / "bugswarm_multifault_realistic" / "synthetic_builds.jsonl",
}


def count_lines(path: Path) -> int:
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def load_by_indices(path: Path, indices: set[int]) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx in indices:
                items.append(json.loads(line))
                if len(items) >= len(indices):
                    break
    return items


def slim_build(build: dict, source: str, split: str) -> dict:
    """Keep all fields required by HyTri & baselines, trimming oversized raw traces."""
    slim_failures = []
    for fl in build["failures"]:
        slim_failures.append({
            "build_id": fl.get("synthetic_build_id", fl.get("build_id", build["build_id"])),
            "failure_id": fl.get("failure_id", ""),
            "test_name": fl.get("test_name", ""),
            "exception_type": fl.get("exception_type", "Unknown"),
            "exception_message": (fl.get("exception_message") or "")[:1000],
            "stack_trace": (fl.get("stack_trace") or "")[:2000],
            "frames": fl.get("frames", [])[:20],
            "project": fl.get("project", ""),
            "ground_truth_bug": fl.get("ground_truth_bug"),
            "ground_truth": fl.get("ground_truth"),
            "bug_id": fl.get("bug_id"),
        })

    # Keep diffs (truncate individual gigantic diffs > 20k chars to keep upload lightweight)
    raw_diffs = build.get("combined_file_diffs") or {}
    slim_diffs = {k: (v[:20000] if isinstance(v, str) else "") for k, v in raw_diffs.items()}

    return {
        "build_id": build["build_id"],
        "source": source,
        "split": split,
        "num_failures": len(slim_failures),
        "combined_changed_files": build.get("combined_changed_files") or [],
        "combined_file_diffs": slim_diffs,
        "failures": slim_failures,
    }


def main():
    exported = []

    for source, path in DATASETS.items():
        total = count_lines(path)
        rng_test = random.Random(42)
        test_idx = set(rng_test.sample(range(total), 10))

        # Sample 5 tuning builds strictly disjoint from test
        remaining = [i for i in range(total) if i not in test_idx]
        rng_tune = random.Random(123)
        tune_idx = set(rng_tune.sample(remaining, 5))

        test_builds = load_by_indices(path, test_idx)
        tune_builds = load_by_indices(path, tune_idx)

        for b in test_builds:
            exported.append(slim_build(b, source, "test"))
        for b in tune_builds:
            exported.append(slim_build(b, source, "tuning"))

        test_f = sum(len(b["failures"]) for b in test_builds)
        tune_f = sum(len(b["failures"]) for b in tune_builds)
        print(f"{source:<16} | Test: 10 builds ({test_f:4d} failures) | Tuning: 5 builds ({tune_f:4d} failures)")

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(exported, f)

    size_mb = OUT_FILE.stat().st_size / (1024 * 1024)
    total_f = sum(b["num_failures"] for b in exported)
    print(f"\nSaved {len(exported)} builds ({total_f} total failures) -> {OUT_FILE} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()