#!/usr/bin/env python3
"""Report complexity-gate trigger rate on MFCI evaluation builds (seed 42)."""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from hytri.ensemble.complexity_gate import ComplexityGate


def count_lines(filepath: Path) -> int:
    with open(filepath) as f:
        return sum(1 for _ in f)


def load_sampled_builds(dataset_path: Path, num_samples: int, seed: int) -> list[dict]:
    builds_file = dataset_path / "synthetic_builds.jsonl"
    if not builds_file.exists():
        return []
    total = count_lines(builds_file)
    rng = random.Random(seed)
    indices = set(rng.sample(range(total), min(num_samples, total)))
    builds = []
    with open(builds_file) as f:
        for idx, line in enumerate(f):
            if idx in indices:
                builds.append(json.loads(line))
    return builds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-builds", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / "data"
    datasets = {
        "Defects4J": data_dir / "defects4j_multifault_realistic",
        "Bears": data_dir / "bears_multifault_realistic",
        "BugSwarm": data_dir / "bugswarm_multifault_realistic",
    }

    gate = ComplexityGate(max_files_for_single_cluster=3)
    totals = {"builds": 0, "triggered": 0}
    by_source = defaultdict(lambda: {"builds": 0, "triggered": 0})
    by_k = defaultdict(lambda: {"builds": 0, "triggered": 0})
    reasons = defaultdict(int)

    for source, path in datasets.items():
        builds = load_sampled_builds(path, args.num_builds, args.seed)
        for build in builds:
            changed_files = build.get("combined_changed_files", [])
            file_diffs = build.get("combined_file_diffs", {})
            analysis = gate.analyze(changed_files, file_diffs)
            k = build.get("num_bugs", build.get("num_true_clusters", len(set(
                f.get("ground_truth_bug", f.get("bug_id", "")) for f in build.get("failures", [])
            ))))

            totals["builds"] += 1
            by_source[source]["builds"] += 1
            by_k[k]["builds"] += 1
            reasons[analysis.reason.split(" - ")[0][:60]] += 1

            if analysis.should_single_cluster:
                totals["triggered"] += 1
                by_source[source]["triggered"] += 1
                by_k[k]["triggered"] += 1

    print(f"Complexity gate analysis (seed={args.seed}, {args.num_builds} builds/dataset)")
    print(f"Total builds: {totals['builds']}")
    print(f"Gate triggered (single cluster): {totals['triggered']} ({100*totals['triggered']/totals['builds']:.1f}%)")
    print()
    print("By dataset:")
    for source in datasets:
        s = by_source[source]
        pct = 100 * s["triggered"] / s["builds"] if s["builds"] else 0
        print(f"  {source}: {s['triggered']}/{s['builds']} ({pct:.1f}%)")
    print()
    print("By fault count k:")
    for k in sorted(by_k):
        s = by_k[k]
        pct = 100 * s["triggered"] / s["builds"] if s["builds"] else 0
        print(f"  k={k}: {s['triggered']}/{s['builds']} ({pct:.1f}%)")
    print()
    print("Top suppression reasons (gate NOT triggered):")
    for reason, count in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
        print(f"  [{count}] {reason}")


if __name__ == "__main__":
    main()
