#!/usr/bin/env python3
"""
Convert coinse/Defects4J-multifault multiple_faults JSON into HyTri synthetic_builds.jsonl.

Source: https://github.com/coinse/Defects4J-multifault
  multiple_faults/{Lang,Chart,Time,Math,Closure}.json

Ground truth per failure is multi-label: Y(test) = bugs whose revealing-test list
includes that test in the target multi-fault version M.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

COINSE_PROJECTS = ("Lang", "Chart", "Time", "Math", "Closure")


def normalize_test_name(name: str) -> str:
    """Normalize test identifiers for matching (coinse '.' vs HyTri '::')."""
    return name.strip().replace("::", ".")


def load_diffs(diffs_path: Path) -> dict[str, dict]:
    with open(diffs_path, "r") as f:
        return json.load(f)


def parse_file_diffs(diff_content: str) -> dict[str, str]:
    file_diffs: dict[str, str] = {}
    if not diff_content:
        return file_diffs
    current_file = None
    current_diff: list[str] = []
    for line_content in diff_content.split("\n"):
        if line_content.startswith("diff --git") or line_content.startswith("Index:"):
            if current_file and current_diff:
                file_diffs[current_file] = "\n".join(current_diff)
            if "diff --git" in line_content:
                parts = line_content.split(" b/")
                current_file = parts[-1] if len(parts) > 1 else None
            elif "Index:" in line_content:
                current_file = line_content.replace("Index:", "").strip()
            current_diff = [line_content]
        else:
            current_diff.append(line_content)
    if current_file and current_diff:
        file_diffs[current_file] = "\n".join(current_diff)
    return file_diffs


def load_local_failures(
    failures_path: Path,
    diffs_path: Path,
) -> tuple[dict[str, dict], dict[tuple[str, str], dict], dict[tuple[str, str], dict]]:
    """
    Returns:
      bugs_by_id: bug_id -> aggregated bug data
      failure_by_bug_test: (bug_id, norm_test) -> failure template
      failure_by_project_test: (project, norm_test) -> failure template (first seen)
    """
    diffs_data = load_diffs(diffs_path)
    bugs_by_id: dict[str, dict] = {}
    failure_by_bug_test: dict[tuple[str, str], dict] = {}
    failure_by_project_test: dict[tuple[str, str], dict] = {}

    with open(failures_path, "r") as f:
        for line in f:
            failure = json.loads(line)
            bug_id = failure.get("bug_id")
            if not bug_id:
                continue
            project = failure.get("project") or bug_id.split("-")[0]
            test_name = failure.get("test_name", "")
            norm = normalize_test_name(test_name)

            diff_info = diffs_data.get(bug_id, {})
            changed_files = diff_info.get("changed_files", [])
            file_diffs = parse_file_diffs(diff_info.get("diff", ""))

            failure["changed_files"] = changed_files
            failure["file_diffs"] = file_diffs
            failure["diff_stats"] = {
                "additions": diff_info.get("additions", 0),
                "deletions": diff_info.get("deletions", 0),
            }

            if bug_id not in bugs_by_id:
                bugs_by_id[bug_id] = {
                    "project": project,
                    "failures": [],
                    "failing_tests": set(),
                    "changed_files": set(),
                    "file_diffs": {},
                    "diff_stats": {"additions": 0, "deletions": 0},
                }
            bugs_by_id[bug_id]["failures"].append(failure)
            bugs_by_id[bug_id]["failing_tests"].add(test_name)
            bugs_by_id[bug_id]["changed_files"].update(changed_files)
            bugs_by_id[bug_id]["file_diffs"].update(file_diffs)
            bugs_by_id[bug_id]["diff_stats"]["additions"] = diff_info.get("additions", 0)
            bugs_by_id[bug_id]["diff_stats"]["deletions"] = diff_info.get("deletions", 0)

            failure_by_bug_test[(bug_id, norm)] = failure
            key_pt = (project, norm)
            if key_pt not in failure_by_project_test:
                failure_by_project_test[key_pt] = failure

    return bugs_by_id, failure_by_bug_test, failure_by_project_test


def lookup_failure(
    project: str,
    norm_test: str,
    bug_ids: list[str],
    failure_by_bug_test: dict[tuple[str, str], dict],
    failure_by_project_test: dict[tuple[str, str], dict],
) -> dict | None:
    for bug_id in bug_ids:
        f = failure_by_bug_test.get((bug_id, norm_test))
        if f:
            return f
    return failure_by_project_test.get((project, norm_test))


def load_coinse_project_json(coinse_dir: Path, project: str) -> dict[str, Any]:
    path = coinse_dir / "multiple_faults" / f"{project}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing coinse file: {path}")
    with open(path, "r") as f:
        return json.load(f)


def build_test_to_bugs(
    version_entry: dict[str, list[str]],
    project: str,
) -> dict[str, set[str]]:
    """Map normalized test name -> set of bug IDs for one multi-fault version."""
    test_to_bugs: dict[str, set[str]] = defaultdict(set)
    for bug_num, tests in version_entry.items():
        bug_id = f"{project}-{bug_num}"
        for raw_test in tests:
            norm = normalize_test_name(raw_test)
            test_to_bugs[norm].add(bug_id)
    return dict(test_to_bugs)


def create_transplant_build(
    project: str,
    version_m: str,
    test_to_bugs: dict[str, set[str]],
    build_idx: int,
    bugs_by_id: dict[str, dict],
    failure_by_bug_test: dict[tuple[str, str], dict],
    failure_by_project_test: dict[tuple[str, str], dict],
    stats: dict[str, int],
) -> dict | None:
    build_id = f"transplant_{project.lower()}_{version_m}"
    all_failures: list[dict] = []
    all_changed_files: set[str] = set()
    all_file_diffs: dict[str, str] = {}
    component_bugs: list[dict] = []
    component_bug_ids = sorted({b for bugs in test_to_bugs.values() for b in bugs})

    for bug_id in component_bug_ids:
        bug_data = bugs_by_id.get(bug_id)
        if not bug_data:
            stats["missing_bug_data"] += 1
            continue
        component_bugs.append({
            "bug_id": bug_id,
            "num_failures": len(bug_data["failures"]),
            "changed_files": list(bug_data["changed_files"]),
            "file_diffs": bug_data.get("file_diffs", {}),
            "diff_stats": bug_data.get("diff_stats", {}),
        })
        all_changed_files.update(bug_data["changed_files"])
        all_file_diffs.update(bug_data.get("file_diffs", {}))

    for norm_test, gt_bugs in sorted(test_to_bugs.items()):
        bug_list = sorted(gt_bugs)
        template = lookup_failure(
            project, norm_test, bug_list,
            failure_by_bug_test, failure_by_project_test,
        )
        if template is None:
            stats["unmapped_tests"] += 1
            continue

        synth = dict(template)
        test_name = template.get("test_name", norm_test)
        synth["failure_id"] = f"{build_id}::{norm_test}"
        synth["synthetic_build_id"] = build_id
        synth["original_build_id"] = template.get("build_id")
        synth["test_name"] = test_name
        synth["project"] = project
        synth["ground_truth_bugs"] = bug_list
        synth["ground_truth_bug"] = bug_list[0]
        synth["ground_truth"] = {
            "root_cause_id": bug_list[0],
            "root_cause_ids": bug_list,
            "changed_files": synth.get("changed_files", []),
            "original_bug_id": bug_list[0],
        }
        synth["dataset_source"] = "coinse-defects4j-multifault"
        synth["transplant_version"] = version_m
        all_failures.append(synth)

    if not all_failures:
        stats["empty_builds"] += 1
        return None

    multi_label_count = sum(1 for f in all_failures if len(f.get("ground_truth_bugs", [])) > 1)

    return {
        "build_id": build_id,
        "project": project,
        "num_bugs": len(component_bug_ids),
        "num_failures": len(all_failures),
        "difficulty": "transplant",
        "similarity_score": 0.0,
        "component_bugs": component_bugs,
        "combined_changed_files": list(all_changed_files),
        "combined_file_diffs": all_file_diffs,
        "combined_diff_stats": {
            "additions": sum(
                bugs_by_id.get(b, {}).get("diff_stats", {}).get("additions", 0)
                for b in component_bug_ids
            ),
            "deletions": sum(
                bugs_by_id.get(b, {}).get("diff_stats", {}).get("deletions", 0)
                for b in component_bug_ids
            ),
        },
        "failures": all_failures,
        "created_at": datetime.now().isoformat(),
        "dataset_source": "coinse-defects4j-multifault",
        "transplant_version": version_m,
        "multilabel_failure_count": multi_label_count,
        "multilabel_fraction": multi_label_count / len(all_failures),
    }


def convert_coinse(
    coinse_dir: Path,
    failures_path: Path,
    diffs_path: Path,
    output_dir: Path,
    projects: list[str] | None = None,
    max_builds_per_project: int | None = None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    projects = projects or list(COINSE_PROJECTS)

    print("Loading local Defects4J failures...")
    bugs_by_id, failure_by_bug_test, failure_by_project_test = load_local_failures(
        failures_path, diffs_path
    )
    print(f"  {len(bugs_by_id)} bugs, {len(failure_by_bug_test)} (bug, test) pairs")

    stats = defaultdict(int)
    builds: list[dict] = []
    build_idx = 0

    for project in projects:
        print(f"Processing coinse project {project}...")
        version_map = load_coinse_project_json(coinse_dir, project)
        version_keys = sorted(version_map.keys(), key=lambda x: int(x))
        if max_builds_per_project:
            version_keys = version_keys[:max_builds_per_project]

        for version_m in version_keys:
            test_to_bugs = build_test_to_bugs(version_map[version_m], project)
            build = create_transplant_build(
                project, version_m, test_to_bugs, build_idx,
                bugs_by_id, failure_by_bug_test, failure_by_project_test, stats,
            )
            if build:
                builds.append(build)
                build_idx += 1
                stats["builds_written"] += 1

    builds_path = output_dir / "synthetic_builds.jsonl"
    with open(builds_path, "w") as f:
        for b in builds:
            f.write(json.dumps(b) + "\n")
    print(f"Wrote {len(builds)} builds to {builds_path}")

    failures_out = output_dir / "failure_instances_multifault.jsonl"
    total_failures = 0
    with open(failures_out, "w") as f:
        for build in builds:
            for failure in build["failures"]:
                row = {
                    "build_id": build["build_id"],
                    "failure_id": failure["failure_id"],
                    "test_name": failure.get("test_name"),
                    "exception_type": failure.get("exception_type"),
                    "exception_message": failure.get("exception_message"),
                    "stack_trace": failure.get("stack_trace"),
                    "frames": failure.get("frames", []),
                    "project": build["project"],
                    "ground_truth_bugs": failure.get("ground_truth_bugs", []),
                    "ground_truth_bug": failure.get("ground_truth_bug"),
                    "ground_truth": failure.get("ground_truth"),
                    "language": "java",
                    "changed_files": failure.get("changed_files", []),
                    "file_diffs": failure.get("file_diffs", {}),
                    "dataset_source": "coinse-defects4j-multifault",
                }
                f.write(json.dumps(row) + "\n")
                total_failures += 1
    print(f"Wrote {total_failures} failures to {failures_out}")

    multi_total = sum(b.get("multilabel_failure_count", 0) for b in builds)
    summary = {
        "created_at": datetime.now().isoformat(),
        "source": "coinse-defects4j-multifault",
        "coinse_dir": str(coinse_dir),
        "config": {
            "projects": projects,
            "max_builds_per_project": max_builds_per_project,
        },
        "stats": {
            "total_builds": len(builds),
            "total_failures": total_failures,
            "multilabel_failures": multi_total,
            "multilabel_fraction": multi_total / total_failures if total_failures else 0,
            **{k: int(v) for k, v in stats.items()},
        },
        "by_project": {},
    }
    by_proj: dict[str, list] = defaultdict(list)
    for b in builds:
        by_proj[b["project"]].append(b)
    for proj, blist in by_proj.items():
        mf = sum(x.get("multilabel_failure_count", 0) for x in blist)
        tf = sum(x["num_failures"] for x in blist)
        summary["by_project"][proj] = {
            "builds": len(blist),
            "failures": tf,
            "multilabel_failures": mf,
            "multilabel_fraction": mf / tf if tf else 0,
        }

    summary_path = output_dir / "dataset_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary: {summary_path}")
    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Convert coinse Defects4J-multifault JSON to HyTri synthetic_builds.jsonl"
    )
    root = Path(__file__).parent.parent
    parser.add_argument(
        "--coinse-dir",
        type=Path,
        default=root / "external" / "Defects4J-multifault",
        help="Path to cloned coinse/Defects4J-multifault repo",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "data" / "defects4j_multifault_transplant",
    )
    parser.add_argument(
        "--failures-path",
        type=Path,
        default=root / "data" / "defects4j_full" / "failure_instances_defects4j.jsonl",
    )
    parser.add_argument(
        "--diffs-path",
        type=Path,
        default=root / "data" / "defects4j_full" / "defects4j_diffs.json",
    )
    parser.add_argument(
        "--projects",
        type=str,
        default=None,
        help="Comma-separated projects (default: Lang,Chart,Time,Math,Closure)",
    )
    parser.add_argument(
        "--max-builds-per-project",
        type=int,
        default=None,
        help="Limit builds per project (for smoke tests)",
    )
    args = parser.parse_args()

    if not args.coinse_dir.exists():
        print(f"Error: coinse repo not found at {args.coinse_dir}")
        print("Clone: git clone https://github.com/coinse/Defects4J-multifault.git "
              f"{args.coinse_dir}")
        raise SystemExit(1)

    projects = None
    if args.projects:
        projects = [p.strip() for p in args.projects.split(",")]

    convert_coinse(
        args.coinse_dir,
        args.failures_path,
        args.diffs_path,
        args.output_dir,
        projects=projects,
        max_builds_per_project=args.max_builds_per_project,
    )


if __name__ == "__main__":
    main()
