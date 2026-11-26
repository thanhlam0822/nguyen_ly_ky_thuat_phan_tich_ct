#!/usr/bin/env python3
"""
Evaluate HyTri and baselines on multi-fault builds from datasets:
- Defects4J-Real (MFCI synthetic)
- Bears-Real, BugSwarm-Real
- Defects4J-Transplant (coinse/Defects4J-multifault, multi-label GT)

Methods evaluated:
- TraceSim (Jiang et al., ASE 2020)
- PWOverlap (position-weighted stack frame overlap)
- ReBucket (Lerch & Mezini, FSE 2013)
- BuildSheriff (full version, ICSE 2022)
- GPTrace (Herter et al., 2025)
- HyTri (ensemble: BuildSheriff rules + LLM semantics)
"""

import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, v_measure_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from hytri.models import FailureInstance, StackFrame
from hytri.baselines.methods import (
    RebucketClusterer,
    PWOverlapClusterer,
    TraceSimOfficialClusterer,
    BuildSheriffClusterer,
    GPTraceClusterer,
)
from hytri.llm.openai_client import OpenAIClient
from hytri.pipeline import HyTriPipeline
from hytri.evaluation.multilabel import (
    compute_pairwise_multilabel_metrics,
    filter_single_label_indices,
    get_multilabel_ground_truth_sets,
    multilabel_overlap_prevalence,
)
from hytri.evaluation.stats_reporting import (
    bootstrap_ci,
    compare_methods_paired,
    format_ci_line,
    wilcoxon_signed_rank_pvalue,
)

TRANSPLANT_SOURCE = "Defects4J-Transplant"

TRANSPLANT_METRIC_COLS = [
    "pairwise_precision",
    "pairwise_recall",
    "pairwise_f1",
    "multilabel_fraction",
    "ari_single_label",
    "nmi_single_label",
    "bcubed_f1_single_label",
]


# =============================================================================
# Metric Computation
# =============================================================================

def compute_bcubed_precision(true_labels, pred_labels):
    if not true_labels:
        return 0.0
    true_arr, pred_arr = np.array(true_labels), np.array(pred_labels)
    n = len(true_arr)
    total = 0.0
    for i in range(n):
        same_pred = pred_arr == pred_arr[i]
        same_true_in_pred = true_arr[same_pred] == true_arr[i]
        total += same_true_in_pred.sum() / same_pred.sum()
    return total / n


def compute_bcubed_recall(true_labels, pred_labels):
    if not true_labels:
        return 0.0
    true_arr, pred_arr = np.array(true_labels), np.array(pred_labels)
    n = len(true_arr)
    total = 0.0
    for i in range(n):
        same_true = true_arr == true_arr[i]
        same_pred_in_true = pred_arr[same_true] == pred_arr[i]
        total += same_pred_in_true.sum() / same_true.sum()
    return total / n


def compute_bcubed_f1(true_labels, pred_labels):
    p = compute_bcubed_precision(true_labels, pred_labels)
    r = compute_bcubed_recall(true_labels, pred_labels)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_all_metrics(true_labels, pred_labels):
    """Compute all clustering metrics."""
    return {
        "ari": adjusted_rand_score(true_labels, pred_labels),
        "nmi": normalized_mutual_info_score(true_labels, pred_labels),
        "v_measure": v_measure_score(true_labels, pred_labels),
        "bcubed_precision": compute_bcubed_precision(true_labels, pred_labels),
        "bcubed_recall": compute_bcubed_recall(true_labels, pred_labels),
        "bcubed_f1": compute_bcubed_f1(true_labels, pred_labels),
    }


def count_lines(filepath: Path) -> int:
    """Count lines in a file without loading it all into memory."""
    count = 0
    with open(filepath, "r") as f:
        for _ in f:
            count += 1
    return count


def load_multifault_builds_sampled(dataset_path: Path, num_samples: int, seed: int = 42) -> list[dict]:
    """Load a random sample of builds without loading all into memory.

    Uses two-pass approach: first count lines, then load only sampled indices.
    """
    builds_file = dataset_path / "synthetic_builds.jsonl"

    if not builds_file.exists():
        return []

    total_builds = count_lines(builds_file)
    print(f"    Total builds in file: {total_builds}")

    if total_builds == 0:
        return []

    rng = random.Random(seed)
    sample_size = min(num_samples, total_builds)
    sampled_indices = set(rng.sample(range(total_builds), sample_size))

    builds = []
    with open(builds_file, "r") as f:
        for idx, line in enumerate(f):
            if idx in sampled_indices:
                build = json.loads(line)
                builds.append(build)
                if len(builds) >= sample_size:
                    break

    return builds


def reindex_cluster_labels(labels: list[int]) -> list[int]:
    """Map arbitrary cluster ids to contiguous 0..k-1."""
    mapping: dict[int, int] = {}
    out: list[int] = []
    for lab in labels:
        if lab not in mapping:
            mapping[lab] = len(mapping)
        out.append(mapping[lab])
    return out


def is_transplant_build(build: dict, source: str | None = None) -> bool:
    src = source or build.get("source", "")
    if src == TRANSPLANT_SOURCE:
        return True
    return build.get("dataset_source") == "coinse-defects4j-multifault"


def enrich_transplant_metrics(
    failures_raw: list[dict],
    pred_labels: list[int],
    base_metrics: dict,
) -> dict:
    """Add overlap-aware primary metrics and single-label subset comparability."""
    true_sets = get_multilabel_ground_truth_sets(failures_raw)
    base_metrics.update(compute_pairwise_multilabel_metrics(true_sets, pred_labels))
    base_metrics.update(multilabel_overlap_prevalence(true_sets))

    idx = filter_single_label_indices(true_sets)
    if len(idx) >= 2:
        sl_failures = [failures_raw[i] for i in idx]
        sl_gt = get_ground_truth_labels(sl_failures)
        sl_pred = reindex_cluster_labels([pred_labels[i] for i in idx])
        sl_metrics = compute_all_metrics(sl_gt, sl_pred)
        base_metrics["ari_single_label"] = sl_metrics["ari"]
        base_metrics["nmi_single_label"] = sl_metrics["nmi"]
        base_metrics["bcubed_f1_single_label"] = sl_metrics["bcubed_f1"]
    else:
        base_metrics["ari_single_label"] = 0.0
        base_metrics["nmi_single_label"] = 0.0
        base_metrics["bcubed_f1_single_label"] = 0.0

    return base_metrics


def default_transplant_metric_zeros() -> dict:
    return {col: 0.0 for col in TRANSPLANT_METRIC_COLS}


def get_ground_truth_labels(failures: list[dict]) -> list[int]:
    """Extract ground truth labels from failures."""
    bug_to_label = {}
    labels = []

    for f in failures:
        if "ground_truth_bug" in f:
            bug_id = f["ground_truth_bug"]
        elif "ground_truth" in f:
            gt = f["ground_truth"]
            bug_id = gt.get("root_cause_id", gt.get("original_bug_id", "unknown"))
        else:
            bug_id = f.get("bug_id", "unknown")

        if bug_id not in bug_to_label:
            bug_to_label[bug_id] = len(bug_to_label)
        labels.append(bug_to_label[bug_id])

    return labels


def convert_to_instances(failures: list[dict]) -> list[FailureInstance]:
    """Convert raw failure dicts to FailureInstance objects."""
    instances = []
    for f in failures:
        frames = [
            StackFrame(
                file=fr.get("file", ""),
                line=fr.get("line", 0),
                func=fr.get("func", "")
            )
            for fr in f.get("frames", [])[:20]
        ]
        instance = FailureInstance(
            build_id=f.get("synthetic_build_id", f.get("build_id", "")),
            failure_id=f.get("failure_id", ""),
            test_name=f.get("test_name", ""),
            exception_type=f.get("exception_type", "Unknown"),
            exception_message=f.get("exception_message", ""),
            stack_trace=f.get("stack_trace", "")[:2000],
            frames=frames,
            project=f.get("project", ""),
        )
        instances.append(instance)
    return instances


def evaluate_baseline(
    clusterer,
    failures: list[FailureInstance],
    ground_truth: list[int],
    failures_raw: list[dict] | None = None,
    source: str | None = None,
) -> dict:
    """Evaluate a baseline clusterer."""
    try:
        clusters, noise = clusterer.cluster(failures)

        fid_to_label = {}
        for i, c in enumerate(clusters):
            for fid in c.failure_ids:
                fid_to_label[fid] = i

        noise_label = len(clusters)
        for fid in noise:
            fid_to_label[fid] = noise_label

        pred_labels = [fid_to_label.get(f.failure_id, noise_label) for f in failures]

        metrics = compute_all_metrics(ground_truth, pred_labels)
        metrics["num_pred_clusters"] = len(clusters) + (1 if noise else 0)
        metrics["num_true_clusters"] = len(set(ground_truth))

        if failures_raw and (source == TRANSPLANT_SOURCE or (
            failures_raw and failures_raw[0].get("ground_truth_bugs")
        )):
            enrich_transplant_metrics(failures_raw, pred_labels, metrics)
        else:
            metrics.update(default_transplant_metric_zeros())

        return metrics
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        out = {
            "ari": 0.0, "nmi": 0.0, "v_measure": 0.0,
            "bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0,
            "num_pred_clusters": 0, "num_true_clusters": len(set(ground_truth)) if ground_truth else 0,
        }
        out.update(default_transplant_metric_zeros())
        return out


def evaluate_hytri(
    pipeline: HyTriPipeline,
    failures: list[FailureInstance],
    ground_truth: list[int],
    build_id: str,
    failures_raw: list[dict] | None = None,
    source: str | None = None,
) -> dict:
    """Evaluate HyTri pipeline."""
    try:
        result = pipeline.triage(failures, build_id=build_id)
        clusters = result.clusters
        noise = result.noise_failure_ids or []

        fid_to_label = {}
        for i, c in enumerate(clusters):
            for fid in c.failure_ids:
                fid_to_label[fid] = i

        noise_label = len(clusters)
        for fid in noise:
            fid_to_label[fid] = noise_label

        pred_labels = [fid_to_label.get(f.failure_id, noise_label) for f in failures]

        metrics = compute_all_metrics(ground_truth, pred_labels)
        metrics["num_pred_clusters"] = len(clusters) + (1 if noise else 0)
        metrics["num_true_clusters"] = len(set(ground_truth))

        if failures_raw and (source == TRANSPLANT_SOURCE or (
            failures_raw and failures_raw[0].get("ground_truth_bugs")
        )):
            enrich_transplant_metrics(failures_raw, pred_labels, metrics)
        else:
            metrics.update(default_transplant_metric_zeros())

        return metrics
    except Exception as e:
        print(f"    Error: {e}")
        import traceback
        traceback.print_exc()
        out = {
            "ari": 0.0, "nmi": 0.0, "v_measure": 0.0,
            "bcubed_precision": 0.0, "bcubed_recall": 0.0, "bcubed_f1": 0.0,
            "num_pred_clusters": 0, "num_true_clusters": len(set(ground_truth)) if ground_truth else 0,
        }
        out.update(default_transplant_metric_zeros())
        return out


RESULT_CSV_HEADER = [
    "build_id", "source", "num_failures", "num_true_clusters", "method",
]


def _completed_build_ids(rows: list[dict], methods: list[str]) -> set[str]:
    """Build IDs that already have a row for every method in this run."""
    expected = set(methods)
    by_build: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        by_build[row["build_id"]].add(row["method"])
    return {bid for bid, seen in by_build.items() if expected <= seen}


def _prepare_resume_output(
    output_path: Path,
    methods: list[str],
    metric_cols: list[str],
) -> tuple[set[str], int]:
    """
    Keep only fully-completed builds in the CSV; drop partial rows for re-run.
    Returns (completed_build_ids, num_partial_requeued).
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set(), 0

    with open(output_path, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return set(), 0
        rows = list(reader)

    if not rows:
        return set(), 0

    complete = _completed_build_ids(rows, methods)
    partial_ids = {r["build_id"] for r in rows} - complete
    kept = [r for r in rows if r["build_id"] in complete]

    header = RESULT_CSV_HEADER + metric_cols
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in kept:
            row = [
                r["build_id"], r["source"], r["num_failures"],
                r["num_true_clusters"], r["method"],
            ]
            row += [r.get(m, "") for m in metric_cols]
            writer.writerow(row)

    return complete, len(partial_ids)


def main():
    parser = argparse.ArgumentParser(description="Evaluate on multi-fault builds from all datasets")
    parser.add_argument("--num-builds", type=int, default=10, help="Number of builds per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=Path, default=Path("results/full_multifault_eval.csv"))
    parser.add_argument(
        "--gptrace-model",
        type=str,
        default="text-embedding-3-small",
        help="OpenAI embedding model for GPTrace (e.g. text-embedding-3-large)",
    )
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated methods to run (default: all). "
             "Options: TraceSim,PWOverlap,ReBucket,BuildSheriff,GPTrace,HyTri",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset sources to evaluate (e.g. Defects4J-Transplant). "
             "Default: all available datasets.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue an interrupted run: keep completed builds in --output, "
             "re-run partial builds, append the rest (requires same --seed and --methods).",
    )
    args = parser.parse_args()

    random.seed(args.seed)

    # Load all datasets
    data_dir = Path(__file__).parent.parent / "data"

    bugswarm_realistic = data_dir / "bugswarm_multifault_realistic"
    bugswarm_original = data_dir / "bugswarm_multifault"

    if bugswarm_realistic.exists() and (bugswarm_realistic / "synthetic_builds.jsonl").exists():
        bugswarm_path = bugswarm_realistic
        bugswarm_name = "BugSwarm-Real"
        print("Using REALISTIC BugSwarm dataset (harder, more representative)")
    else:
        bugswarm_path = bugswarm_original
        bugswarm_name = "BugSwarm"
        print("Using original BugSwarm dataset (to generate realistic: python scripts/create_bugswarm_multifault_realistic.py)")

    bears_realistic = data_dir / "bears_multifault_realistic"
    bears_original = data_dir / "bears_multifault"

    if bears_realistic.exists() and (bears_realistic / "synthetic_builds.jsonl").exists():
        bears_path = bears_realistic
        bears_name = "Bears-Real"
        print("Using REALISTIC Bears dataset (with actual diffs)")
    else:
        bears_path = bears_original
        bears_name = "Bears"
        print("Using original Bears dataset (to generate realistic: python scripts/create_bears_multifault_realistic.py)")

    defects4j_realistic = data_dir / "defects4j_multifault_realistic"
    defects4j_original = data_dir / "defects4j_multifault"

    if defects4j_realistic.exists() and (defects4j_realistic / "synthetic_builds.jsonl").exists():
        defects4j_path = defects4j_realistic
        defects4j_name = "Defects4J-Real"
        print("Using REALISTIC Defects4J dataset (with actual diffs and difficulty tiers)")
    else:
        defects4j_path = defects4j_original
        defects4j_name = "Defects4J"
        print("Using original Defects4J dataset (to generate realistic: python scripts/create_defects4j_multifault_realistic.py)")

    datasets = {
        defects4j_name: defects4j_path,
        bears_name: bears_path,
        bugswarm_name: bugswarm_path,
    }

    transplant_path = data_dir / "defects4j_multifault_transplant"
    if transplant_path.exists() and (transplant_path / "synthetic_builds.jsonl").exists():
        datasets["Defects4J-Transplant"] = transplant_path
        print("Using coinse Defects4J-multifault (transplant) dataset")
    else:
        print(
            "Defects4J-Transplant not found (optional). "
            "Run: python scripts/create_defects4j_multifault_transplant.py"
        )

    if args.datasets:
        allowed = {d.strip() for d in args.datasets.split(",")}
        datasets = {k: v for k, v in datasets.items() if k in allowed}
        if not datasets:
            print(f"Error: --datasets {args.datasets!r} matched no loaded datasets.")
            sys.exit(1)
        print(f"Evaluating subset: {', '.join(datasets.keys())}")

    selected_builds = []
    for name, path in datasets.items():
        print(f"Loading {name} from {path}...")
        builds = load_multifault_builds_sampled(path, args.num_builds, seed=args.seed)
        for b in builds:
            b["source"] = name
        print(f"  Sampled {len(builds)} builds")
        selected_builds.extend(builds)

    if not selected_builds:
        print("Error: No builds found!")
        sys.exit(1)

    print(f"\nTotal selected: {len(selected_builds)} builds")

    metric_cols = [
        "ari", "nmi", "v_measure", "bcubed_precision", "bcubed_recall",
        "bcubed_f1", "num_pred_clusters",
    ]
    all_metric_cols = metric_cols + TRANSPLANT_METRIC_COLS

    all_methods = ["TraceSim", "PWOverlap", "ReBucket", "BuildSheriff", "GPTrace", "HyTri"]
    if args.methods:
        requested = {m.strip() for m in args.methods.split(",")}
    else:
        requested = set(all_methods)

    needs_openai = (
        "HyTri" in requested
        or "GPTrace" in requested
        or "GPTrace-large" in requested
        or "GPTrace-small" in requested
        or not args.methods
    )
    openai_key = os.environ.get("OPENAI_API_KEY")
    if needs_openai and not openai_key:
        print("Error: OPENAI_API_KEY not set (required for GPTrace/HyTri)")
        print("Export it: export OPENAI_API_KEY='your-key'")
        sys.exit(1)

    openai_client = OpenAIClient(api_key=openai_key) if openai_key else None

    # Initialize all methods
    baselines = {
        "TraceSim": TraceSimOfficialClusterer(distance_threshold=0.5),
        "PWOverlap": PWOverlapClusterer(distance_threshold=0.5),
        "ReBucket": RebucketClusterer(distance_threshold=0.5),
    }

    buildsheriff = BuildSheriffClusterer()

    gptrace_label = "GPTrace"
    if args.gptrace_model != "text-embedding-3-small":
        gptrace_label = f"GPTrace-{args.gptrace_model.split('-')[-1]}"
    gptrace = None
    if openai_client:
        gptrace = GPTraceClusterer(
            openai_client=openai_client,
            embedding_model=args.gptrace_model,
        )

    hytri_pipeline = None
    if openai_key:
        hytri_pipeline = HyTriPipeline(
            openai_api_key=openai_key,
            clustering_mode="ensemble",
            enable_complexity_gate=True,
            enable_stratification=True,
            enable_hypothesis=True,
            generate_explanations=False,
            verify_explanations=False,
        )

    # Prepare output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    method_list = list(baselines.keys()) + ["BuildSheriff", gptrace_label, "HyTri"]
    alias = {"GPTrace-large": gptrace_label, "GPTrace-small": gptrace_label}
    methods = []
    for m in method_list:
        if m in requested or alias.get(m) == m or (
            m == gptrace_label and ("GPTrace" in requested or gptrace_label in requested)
        ):
            methods.append(m)
    if not methods:
        print(f"Error: none of --methods {args.methods!r} matched available methods: {method_list}")
        sys.exit(1)

    header = RESULT_CSV_HEADER + all_metric_cols
    completed_build_ids: set[str] = set()
    num_partial_requeued = 0

    if args.resume:
        completed_build_ids, num_partial_requeued = _prepare_resume_output(
            args.output, methods, all_metric_cols
        )
        if completed_build_ids or num_partial_requeued:
            print(
                f"Resume: {len(completed_build_ids)} builds complete, "
                f"{num_partial_requeued} partial build(s) will be re-run"
            )
        else:
            print("Resume: no completed builds in output yet; starting from build 1")
    else:
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerow(header)

    if not args.output.exists():
        with open(args.output, "w", newline="") as f:
            csv.writer(f).writerow(header)

    print(f"\n{'='*80}")
    print("FULL MULTI-FAULT EVALUATION")
    print(f"Methods: {', '.join(methods)}")
    print(f"{'='*80}\n")

    diff_stats = {"with_diffs": 0, "without_diffs": 0, "total_diff_files": 0}

    skipped_complete = 0
    for i, build in enumerate(selected_builds):
        if build["build_id"] in completed_build_ids:
            skipped_complete += 1
            continue

        failures_raw = build["failures"]
        ground_truth = get_ground_truth_labels(failures_raw)
        num_true_clusters = len(set(ground_truth))

        file_diffs = build.get("combined_file_diffs", {})
        num_diffs = len(file_diffs) if file_diffs else 0
        diff_stats["total_diff_files"] += num_diffs
        if num_diffs > 0:
            diff_stats["with_diffs"] += 1
        else:
            diff_stats["without_diffs"] += 1

        build_source = build["source"]
        is_transplant = build_source == TRANSPLANT_SOURCE

        print(f"[{i+1}/{len(selected_builds)}] {build['build_id']} ({build_source})")
        print(f"    {build['num_failures']} failures, {num_true_clusters} true clusters, {num_diffs} file diffs")
        if is_transplant:
            ml_frac = build.get("multilabel_fraction")
            if ml_frac is None and failures_raw:
                ts = get_multilabel_ground_truth_sets(failures_raw)
                prev = multilabel_overlap_prevalence(ts)
                ml_frac = prev["multilabel_fraction"]
            if ml_frac is not None:
                print(f"    multi-label failures: {ml_frac:.1%}")

        failures = convert_to_instances(failures_raw)

        build_results = []

        changed_files = build.get("combined_changed_files", [])
        file_diffs = build.get("combined_file_diffs", {})

        eval_kw = {"failures_raw": failures_raw, "source": build_source}

        # Evaluate algorithmic baselines (TraceSim, PWOverlap, ReBucket)
        for method_name, clusterer in baselines.items():
            if method_name not in methods:
                continue
            metrics = evaluate_baseline(clusterer, failures, ground_truth, **eval_kw)
            build_results.append({
                "build_id": build["build_id"],
                "source": build_source,
                "num_failures": build["num_failures"],
                "num_true_clusters": num_true_clusters,
                "method": method_name,
                **metrics
            })
            if is_transplant:
                print(
                    f"    {method_name}: pairwise F1={metrics['pairwise_f1']:.3f} "
                    f"(ARI single-label={metrics['ari_single_label']:.3f})"
                )
            else:
                print(f"    {method_name}: ARI={metrics['ari']:.3f}, BCubed F1={metrics['bcubed_f1']:.3f}")

        # Evaluate BuildSheriff with change info
        if "BuildSheriff" in methods:
            buildsheriff.set_changes(changed_files, file_diffs)
            metrics = evaluate_baseline(buildsheriff, failures, ground_truth, **eval_kw)
            build_results.append({
                "build_id": build["build_id"],
                "source": build_source,
                "num_failures": build["num_failures"],
                "num_true_clusters": num_true_clusters,
                "method": "BuildSheriff",
                **metrics
            })
            if is_transplant:
                print(
                    f"    BuildSheriff: pairwise F1={metrics['pairwise_f1']:.3f} "
                    f"(ARI single-label={metrics['ari_single_label']:.3f})"
                )
            else:
                print(f"    BuildSheriff: ARI={metrics['ari']:.3f}, BCubed F1={metrics['bcubed_f1']:.3f}")

        # Evaluate GPTrace (LLM embedding + HDBSCAN)
        if gptrace_label in methods and gptrace is not None:
            metrics = evaluate_baseline(gptrace, failures, ground_truth, **eval_kw)
            build_results.append({
                "build_id": build["build_id"],
                "source": build_source,
                "num_failures": build["num_failures"],
                "num_true_clusters": num_true_clusters,
                "method": gptrace_label,
                **metrics
            })
            if is_transplant:
                print(
                    f"    {gptrace_label}: pairwise F1={metrics['pairwise_f1']:.3f} "
                    f"(ARI single-label={metrics['ari_single_label']:.3f})"
                )
            else:
                print(f"    {gptrace_label}: ARI={metrics['ari']:.3f}, BCubed F1={metrics['bcubed_f1']:.3f}")

        # Evaluate HyTri (ensemble pipeline)
        if "HyTri" in methods and hytri_pipeline is not None:
            hytri_pipeline.set_build_changes(changed_files, build.get("combined_file_diffs", {}))
            metrics = evaluate_hytri(
                hytri_pipeline, failures, ground_truth, build["build_id"], **eval_kw
            )
            build_results.append({
                "build_id": build["build_id"],
                "source": build_source,
                "num_failures": build["num_failures"],
                "num_true_clusters": num_true_clusters,
                "method": "HyTri",
                **metrics
            })
            if is_transplant:
                print(
                    f"    HyTri: pairwise F1={metrics['pairwise_f1']:.3f} "
                    f"(ARI single-label={metrics['ari_single_label']:.3f})"
                )
            else:
                print(f"    HyTri: ARI={metrics['ari']:.3f}, BCubed F1={metrics['bcubed_f1']:.3f}")

        # Write results
        with open(args.output, "a", newline="") as f:
            writer = csv.writer(f)
            for result in build_results:
                row = [result["build_id"], result["source"], result["num_failures"],
                       result["num_true_clusters"], result["method"]]
                row += [result.get(m, 0.0) for m in all_metric_cols]
                writer.writerow(row)

        print()

    if args.resume and skipped_complete:
        print(
            f"Resume: skipped {skipped_complete} already-complete build(s); "
            f"evaluated {len(selected_builds) - skipped_complete} this run"
        )

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY BY METHOD")
    print(f"{'='*80}\n")

    results_by_method = defaultdict(list)
    with open(args.output, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results_by_method[row["method"]].append(row)

    print(f"{'Method':<20} {'ARI':>10} {'NMI':>10} {'BCubed F1':>12} {'Pred Clusters':>15}")
    print("-" * 70)

    for method in methods:
        rows = results_by_method[method]
        if rows:
            avg_ari = sum(float(r["ari"]) for r in rows) / len(rows)
            avg_nmi = sum(float(r["nmi"]) for r in rows) / len(rows)
            avg_f1 = sum(float(r["bcubed_f1"]) for r in rows) / len(rows)
            avg_pred = sum(float(r["num_pred_clusters"]) for r in rows) / len(rows)
            print(f"{method:<20} {avg_ari:>10.3f} {avg_nmi:>10.3f} {avg_f1:>12.3f} {avg_pred:>15.1f}")

    # Summary by dataset
    print(f"\n{'='*80}")
    print("SUMMARY BY DATASET")
    print(f"{'='*80}\n")

    for source in datasets.keys():
        print(f"\n{source}:")
        print(f"{'Method':<20} {'ARI':>10} {'BCubed F1':>12}")
        print("-" * 45)

        for method in methods:
            rows = [r for r in results_by_method[method] if r["source"] == source]
            if rows:
                avg_ari = sum(float(r["ari"]) for r in rows) / len(rows)
                avg_f1 = sum(float(r["bcubed_f1"]) for r in rows) / len(rows)
                print(f"{method:<20} {avg_ari:>10.3f} {avg_f1:>12.3f}")

    # Print diff usage summary
    print(f"\n{'='*80}")
    print("DIFF USAGE SUMMARY (for HyTri)")
    print(f"{'='*80}")
    print(f"Builds with diffs: {diff_stats['with_diffs']}/{len(selected_builds)}")
    print(f"Builds without diffs: {diff_stats['without_diffs']}/{len(selected_builds)}")
    print(f"Total diff files processed: {diff_stats['total_diff_files']}")
    if diff_stats['without_diffs'] > 0:
        print(f"WARNING: {diff_stats['without_diffs']} builds have no diffs - diff_line signal disabled for those")
    else:
        print("All builds have diffs - diff_line signal and diff-enhanced hypotheses active for all builds")

    # Transplant-specific summary (primary: pairwise F1)
    if TRANSPLANT_SOURCE in datasets:
        print(f"\n{'='*80}")
        print(f"SUMMARY: {TRANSPLANT_SOURCE} (overlap-aware primary metrics)")
        print(f"{'='*80}\n")
        print(f"{'Method':<20} {'Pairwise F1':>12} {'Pair P':>10} {'Pair R':>10} {'ARI (|Y|=1)':>12}")
        print("-" * 70)
        transplant_scores: dict[str, list[float]] = defaultdict(list)
        for method in methods:
            rows = [r for r in results_by_method[method] if r["source"] == TRANSPLANT_SOURCE]
            if not rows:
                continue
            pw_f1 = [float(r["pairwise_f1"]) for r in rows]
            transplant_scores[method] = pw_f1
            avg_pw = sum(pw_f1) / len(pw_f1)
            avg_pp = sum(float(r["pairwise_precision"]) for r in rows) / len(rows)
            avg_pr = sum(float(r["pairwise_recall"]) for r in rows) / len(rows)
            avg_ari_sl = sum(float(r["ari_single_label"]) for r in rows) / len(rows)
            print(f"{method:<20} {avg_pw:>12.3f} {avg_pp:>10.3f} {avg_pr:>10.3f} {avg_ari_sl:>12.3f}")

        stats_path = args.output.parent / "transplant_eval_stats.txt"
        with open(stats_path, "w") as sf:
            sf.write(f"Transplant evaluation statistics\n")
            sf.write(f"Generated: {datetime.now().isoformat()}\n\n")
            for metric_key, label in [
                ("pairwise_f1", "Pairwise F1 (primary)"),
                ("pairwise_precision", "Pairwise precision"),
                ("pairwise_recall", "Pairwise recall"),
                ("ari_single_label", "ARI single-label subset"),
            ]:
                sf.write(f"\n{label}\n")
                for method in methods:
                    rows = [r for r in results_by_method[method] if r["source"] == TRANSPLANT_SOURCE]
                    if not rows:
                        continue
                    vals = [float(r[metric_key]) for r in rows]
                    sf.write(format_ci_line(method, vals, seed=args.seed) + "\n")

            if "HyTri" in transplant_scores and "BuildSheriff" in transplant_scores:
                a = [float(r["pairwise_f1"]) for r in results_by_method["BuildSheriff"]
                     if r["source"] == TRANSPLANT_SOURCE]
                b = [float(r["pairwise_f1"]) for r in results_by_method["HyTri"]
                     if r["source"] == TRANSPLANT_SOURCE]
                if len(a) == len(b) and len(a) >= 2:
                    p = wilcoxon_signed_rank_pvalue(b, a)
                    sf.write(f"\nHyTri vs BuildSheriff (pairwise F1): p={p:.4g}\n")

            comparisons = compare_methods_paired(
                {m: [float(r["pairwise_f1"]) for r in results_by_method[m]
                     if r["source"] == TRANSPLANT_SOURCE]
                 for m in methods if results_by_method[m]},
                "pairwise_f1",
                baseline="BuildSheriff",
            )
            for c in comparisons:
                sf.write(
                    f"Wilcoxon {c['method_a']} vs {c['method_b']}: "
                    f"p={c['p_value']:.4g} "
                    f"(means {c['mean_a']:.4f} vs {c['mean_b']:.4f})\n"
                )
        print(f"\nTransplant stats written to: {stats_path}")

    # Save per-dataset CSV files
    output_dir = args.output.parent
    for source in datasets.keys():
        safe_name = source.lower().replace("-", "_").replace(" ", "_")
        dataset_output = output_dir / f"multifault_eval_{safe_name}.csv"

        with open(dataset_output, "w", newline="") as f:
            writer = csv.writer(f)
            header = [
                "build_id", "source", "num_failures", "num_true_clusters", "method",
            ] + all_metric_cols
            writer.writerow(header)

            for method in methods:
                rows = [r for r in results_by_method[method] if r["source"] == source]
                for r in rows:
                    row = [r["build_id"], r["source"], r["num_failures"],
                           r["num_true_clusters"], r["method"]]
                    row += [r.get(m, "") for m in all_metric_cols]
                    writer.writerow(row)

        print(f"Per-dataset results saved to: {dataset_output}")

    print(f"\nAll results saved to: {args.output}")
    print(f"Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
