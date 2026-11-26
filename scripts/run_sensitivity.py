#!/usr/bin/env python3
"""Run one-factor-at-a-time sensitivity analysis for HyTri ensemble mode.

Design goals:
- Reuse expensive LLM-derived signals once per build.
- Sweep major tunable parameters requested by reviewers.
- Report compact, paper-friendly outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata, wilcoxon
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).parent.parent))

from hytri.models import FailureInstance, StackFrame
from hytri.llm.openai_client import OpenAIClient
from hytri.ensemble.hypothesis import HypothesisGenerator
from hytri.ensemble.signals import MultiSignalComputer, SimilaritySignals
from hytri.ensemble.aggregator import AggregationEnsemble
from hytri.ensemble.threshold import AdaptiveThreshold, ThresholdConfig
from hytri.ensemble.complexity_gate import ComplexityGate
from hytri.evaluation.stats_reporting import bootstrap_ci


@dataclass(frozen=True)
class SignalVariantConfig:
    diff_line_tolerance: int = 5
    suspected_change_match_similarity: float = 0.9
    message_confidence_cap: float = 0.6
    test_name_confidence_cap: float = 0.5
    semantic_confidence: float = 0.7
    hypothesis_confidence: float = 0.8


def compute_bcubed_precision(true_labels: list[int], pred_labels: list[int]) -> float:
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


def compute_bcubed_recall(true_labels: list[int], pred_labels: list[int]) -> float:
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


def compute_bcubed_f1(true_labels: list[int], pred_labels: list[int]) -> float:
    p = compute_bcubed_precision(true_labels, pred_labels)
    r = compute_bcubed_recall(true_labels, pred_labels)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def compute_metrics(true_labels: list[int], pred_labels: list[int]) -> dict[str, float]:
    return {
        "ari": float(adjusted_rand_score(true_labels, pred_labels)),
        "nmi": float(normalized_mutual_info_score(true_labels, pred_labels)),
        "bcubed_f1": float(compute_bcubed_f1(true_labels, pred_labels)),
    }


def count_lines(filepath: Path) -> int:
    count = 0
    with open(filepath, "r") as f:
        for _ in f:
            count += 1
    return count


def load_multifault_builds_sampled(dataset_path: Path, num_samples: int, seed: int = 42) -> list[dict]:
    builds_file = dataset_path / "synthetic_builds.jsonl"
    if not builds_file.exists():
        return []

    total_builds = count_lines(builds_file)
    if total_builds == 0:
        return []

    rng = random.Random(seed)
    sample_size = min(num_samples, total_builds)
    sampled_indices = set(rng.sample(range(total_builds), sample_size))

    builds = []
    with open(builds_file, "r") as f:
        for idx, line in enumerate(f):
            if idx in sampled_indices:
                builds.append(json.loads(line))
                if len(builds) >= sample_size:
                    break
    return builds


def get_ground_truth_labels(failures: list[dict]) -> list[int]:
    bug_to_label: dict[str, int] = {}
    labels: list[int] = []
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
    instances = []
    for f in failures:
        frames = [
            StackFrame(file=fr.get("file", ""), line=fr.get("line", 0), func=fr.get("func", ""))
            for fr in f.get("frames", [])[:20]
        ]
        instances.append(
            FailureInstance(
                build_id=f.get("synthetic_build_id", f.get("build_id", "")),
                failure_id=f.get("failure_id", ""),
                test_name=f.get("test_name", ""),
                exception_type=f.get("exception_type", "Unknown"),
                exception_message=f.get("exception_message", ""),
                stack_trace=f.get("stack_trace", "")[:2000],
                frames=frames,
                project=f.get("project", ""),
            )
        )
    return instances


def labels_from_clusters(failures: list[FailureInstance], clusters) -> list[int]:
    fid_to_label: dict[str, int] = {}
    for i, c in enumerate(clusters):
        for fid in c.failure_ids:
            fid_to_label[fid] = i
    fallback_label = len(clusters)
    return [fid_to_label.get(f.failure_id, fallback_label) for f in failures]


def resolve_datasets(root: Path) -> dict[str, Path]:
    data_dir = root / "data"
    candidates = [
        ("Defects4J", data_dir / "defects4j_multifault_realistic", data_dir / "defects4j_multifault"),
        ("Bears", data_dir / "bears_multifault_realistic", data_dir / "bears_multifault"),
        ("BugSwarm", data_dir / "bugswarm_multifault_realistic", data_dir / "bugswarm_multifault"),
    ]
    out: dict[str, Path] = {}
    for name, preferred, fallback in candidates:
        if (preferred / "synthetic_builds.jsonl").exists():
            out[name] = preferred
        elif (fallback / "synthetic_builds.jsonl").exists():
            out[name] = fallback
    return out


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(v, 0.0) for v in weights.values())
    if total <= 0:
        n = len(weights)
        return {k: 1.0 / n for k in weights}
    return {k: max(v, 0.0) / total for k, v in weights.items()}


def scaled_type_multipliers(base: dict[tuple[str, str], dict[str, float]], scale: float):
    out: dict[tuple[str, str], dict[str, float]] = {}
    for pair, sigs in base.items():
        out[pair] = {}
        for name, value in sigs.items():
            out[pair][name] = value if abs(value - 1.0) < 1e-9 else value * scale
    return out


def with_multiplier_overrides(
    base: dict[tuple[str, str], dict[str, float]],
    overrides: dict[tuple[str, str], dict[str, float]] | None,
) -> dict[tuple[str, str], dict[str, float]]:
    if not overrides:
        return base
    out: dict[tuple[str, str], dict[str, float]] = {}
    for pair, sigs in base.items():
        out[pair] = dict(sigs)
    for pair, sigs in overrides.items():
        if pair not in out:
            out[pair] = {}
        out[pair].update(sigs)
    return out


@dataclass
class BuildArtifact:
    build_id: str
    source: str
    project: str
    failures: list[FailureInstance]
    ground_truth: list[int]
    changed_files: list[str]
    file_diffs: dict[str, str]
    signals: SimilaritySignals
    hypotheses: list | None


def clone_signals(signals: SimilaritySignals) -> SimilaritySignals:
    return SimilaritySignals(
        s_pdm=signals.s_pdm.copy() if signals.s_pdm.size else np.array([]),
        s_overlap=signals.s_overlap.copy() if signals.s_overlap.size else np.array([]),
        s_semantic=signals.s_semantic.copy() if signals.s_semantic.size else np.array([]),
        s_message=signals.s_message.copy() if signals.s_message.size else np.array([]),
        s_hypothesis=signals.s_hypothesis.copy() if signals.s_hypothesis.size else np.array([]),
        s_test_name=signals.s_test_name.copy() if signals.s_test_name.size else np.array([]),
        s_diff_line=signals.s_diff_line.copy() if signals.s_diff_line.size else np.array([]),
        confidences=dict(signals.confidences),
        n_failures=signals.n_failures,
    )


def signal_computer_for_variant(
    signal_cfg: SignalVariantConfig,
    signal_ops_cache: dict[SignalVariantConfig, MultiSignalComputer],
) -> MultiSignalComputer:
    cached = signal_ops_cache.get(signal_cfg)
    if cached is not None:
        return cached
    ops = MultiSignalComputer(
        openai_client=None,
        use_semantic=False,
        message_confidence_cap=signal_cfg.message_confidence_cap,
        test_name_confidence_cap=signal_cfg.test_name_confidence_cap,
        semantic_confidence=signal_cfg.semantic_confidence,
        hypothesis_confidence=signal_cfg.hypothesis_confidence,
        suspected_change_match_similarity=signal_cfg.suspected_change_match_similarity,
        diff_line_tolerance=signal_cfg.diff_line_tolerance,
    )
    signal_ops_cache[signal_cfg] = ops
    return ops


def materialize_variant_signals(
    art: BuildArtifact,
    *,
    signal_cfg: SignalVariantConfig,
    baseline_signal_cfg: SignalVariantConfig,
    signal_ops_cache: dict[SignalVariantConfig, MultiSignalComputer],
    signal_variant_cache: dict[tuple[str, SignalVariantConfig], SimilaritySignals],
) -> SimilaritySignals:
    if signal_cfg == baseline_signal_cfg:
        return art.signals

    cache_key = (art.build_id, signal_cfg)
    cached = signal_variant_cache.get(cache_key)
    if cached is not None:
        return cached

    ops = signal_computer_for_variant(signal_cfg, signal_ops_cache)
    variant = clone_signals(art.signals)

    variant.confidences["message"] = ops._compute_message_confidence(art.failures)
    variant.confidences["test_name"] = ops._compute_test_name_confidence(art.failures)
    variant.confidences["semantic"] = signal_cfg.semantic_confidence

    if art.hypotheses:
        variant.s_hypothesis = ops._compute_hypothesis_similarity(art.hypotheses)
        variant.confidences["hypothesis"] = signal_cfg.hypothesis_confidence
    else:
        variant.confidences["hypothesis"] = 0.0

    if art.changed_files and art.file_diffs:
        variant.s_diff_line = ops._compute_diff_line_similarity(art.failures, art.changed_files, art.file_diffs)
        variant.confidences["diff_line"] = ops._compute_diff_line_confidence(art.failures, variant.s_diff_line)
    else:
        n = len(art.failures)
        variant.s_diff_line = np.zeros((n, n))
        variant.confidences["diff_line"] = 0.0

    signal_variant_cache[cache_key] = variant
    return variant


def precompute_build_artifacts(
    builds: list[dict],
    signal_computer: MultiSignalComputer,
    hypothesis_generator: HypothesisGenerator,
) -> list[BuildArtifact]:
    artifacts: list[BuildArtifact] = []
    for idx, build in enumerate(builds, start=1):
        failures_raw = build["failures"]
        failures = convert_to_instances(failures_raw)
        gt = get_ground_truth_labels(failures_raw)
        changed_files = build.get("combined_changed_files", [])
        file_diffs = build.get("combined_file_diffs", {})

        hypotheses = None
        if changed_files:
            batch = hypothesis_generator.generate(failures, changed_files, file_diffs)
            hypotheses = batch.hypotheses

        signals = signal_computer.compute(
            failures,
            changed_files=changed_files,
            file_diffs=file_diffs,
            hypotheses=hypotheses,
        )

        artifacts.append(
            BuildArtifact(
                build_id=build["build_id"],
                source=build["source"],
                project=build.get("project", failures[0].project if failures else ""),
                failures=failures,
                ground_truth=gt,
                changed_files=changed_files,
                file_diffs=file_diffs,
                signals=signals,
                hypotheses=hypotheses,
            )
        )
        print(f"Precomputed [{idx}/{len(builds)}] {build['build_id']} ({build['source']})")
    return artifacts


def evaluate_variant(
    artifacts: list[BuildArtifact],
    *,
    signal_weights: dict[str, float],
    lambda_penalty: float,
    threshold_cfg: ThresholdConfig,
    multiplier_scale: float,
    pair_multiplier_overrides: dict[tuple[str, str], dict[str, float]] | None,
    signal_cfg: SignalVariantConfig,
    baseline_signal_cfg: SignalVariantConfig,
    signal_ops_cache: dict[SignalVariantConfig, MultiSignalComputer],
    signal_variant_cache: dict[tuple[str, SignalVariantConfig], SimilaritySignals],
    linkage_method: str,
    delta_method: int,
    max_files: int,
) -> dict[str, list[float]]:
    rows: dict[str, list[float]] = {"ari": [], "nmi": [], "bcubed_f1": []}

    for art in artifacts:
        gate = ComplexityGate(
            delta_exception=delta_method,
            delta_assertion=delta_method,
            max_files_for_single_cluster=max_files,
            require_file_overlap=True,
        )
        gate_decision = gate.analyze(art.changed_files, art.file_diffs)

        if gate_decision.should_single_cluster:
            pred = [0 for _ in art.failures]
        else:
            signals = materialize_variant_signals(
                art,
                signal_cfg=signal_cfg,
                baseline_signal_cfg=baseline_signal_cfg,
                signal_ops_cache=signal_ops_cache,
                signal_variant_cache=signal_variant_cache,
            )
            aggregator = AggregationEnsemble(
                signal_weights=signal_weights,
                use_confidence_weighting=True,
                disagreement_penalty=lambda_penalty,
                min_cluster_size=1,
                use_type_adaptation=True,
            )
            aggregator.LINKAGE_METHOD = linkage_method
            base_multipliers = scaled_type_multipliers(
                AggregationEnsemble.TYPE_PAIR_MULTIPLIERS.copy(),
                multiplier_scale,
            )
            aggregator.TYPE_PAIR_MULTIPLIERS = with_multiplier_overrides(
                base_multipliers,
                pair_multiplier_overrides,
            )
            selector = AdaptiveThreshold(config=threshold_cfg)
            threshold = selector.select_threshold(signals, failure_type="mixed")
            result = aggregator.cluster(art.failures, signals, distance_threshold=threshold)
            pred = labels_from_clusters(art.failures, result.clusters)

        m = compute_metrics(art.ground_truth, pred)
        rows["ari"].append(m["ari"])
        rows["nmi"].append(m["nmi"])
        rows["bcubed_f1"].append(m["bcubed_f1"])

    return rows


def mean_metrics(rows: dict[str, list[float]]) -> dict[str, float]:
    return {
        "ari": float(np.mean(rows["ari"])) if rows["ari"] else 0.0,
        "nmi": float(np.mean(rows["nmi"])) if rows["nmi"] else 0.0,
        "bcubed_f1": float(np.mean(rows["bcubed_f1"])) if rows["bcubed_f1"] else 0.0,
    }


def choose_best_variant_on_subset(
    subset: list[BuildArtifact],
    variants: list[dict[str, Any]],
    *,
    baseline_signal_cfg: SignalVariantConfig,
    signal_ops_cache: dict[SignalVariantConfig, MultiSignalComputer],
    signal_variant_cache: dict[tuple[str, SignalVariantConfig], SimilaritySignals],
) -> tuple[dict[str, Any], float]:
    if not subset:
        raise ValueError("Subset cannot be empty when selecting best variant.")

    best_variant = variants[0]
    best_train_ari = float("-inf")
    for v in variants:
        train_rows = evaluate_variant(
            subset,
            signal_weights=v["weights"],
            lambda_penalty=v["lambda_penalty"],
            threshold_cfg=v["threshold_cfg"],
            multiplier_scale=v["multiplier_scale"],
            pair_multiplier_overrides=v["pair_multiplier_overrides"],
            signal_cfg=v["signal_cfg"],
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
            linkage_method=v["linkage"],
            delta_method=v["delta_method"],
            max_files=v["max_files"],
        )
        train_ari = float(np.mean(train_rows["ari"])) if train_rows["ari"] else float("-inf")
        if train_ari > best_train_ari:
            best_train_ari = train_ari
            best_variant = v
    return best_variant, best_train_ari


def run_leave_out_robustness(
    artifacts: list[BuildArtifact],
    variants: list[dict[str, Any]],
    *,
    baseline_signal_cfg: SignalVariantConfig,
    signal_ops_cache: dict[SignalVariantConfig, MultiSignalComputer],
    signal_variant_cache: dict[tuple[str, SignalVariantConfig], SimilaritySignals],
    min_project_builds: int,
    max_heldout_projects: int,
) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    baseline_variant = variants[0]

    datasets = sorted({a.source for a in artifacts})
    for heldout_dataset in datasets:
        test = [a for a in artifacts if a.source == heldout_dataset]
        train = [a for a in artifacts if a.source != heldout_dataset]
        if not train or not test:
            continue

        best_variant, train_best_ari = choose_best_variant_on_subset(
            train,
            variants,
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
        )
        test_rows = evaluate_variant(
            test,
            signal_weights=best_variant["weights"],
            lambda_penalty=best_variant["lambda_penalty"],
            threshold_cfg=best_variant["threshold_cfg"],
            multiplier_scale=best_variant["multiplier_scale"],
            pair_multiplier_overrides=best_variant["pair_multiplier_overrides"],
            signal_cfg=best_variant["signal_cfg"],
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
            linkage_method=best_variant["linkage"],
            delta_method=best_variant["delta_method"],
            max_files=best_variant["max_files"],
        )
        baseline_test_rows = evaluate_variant(
            test,
            signal_weights=baseline_variant["weights"],
            lambda_penalty=baseline_variant["lambda_penalty"],
            threshold_cfg=baseline_variant["threshold_cfg"],
            multiplier_scale=baseline_variant["multiplier_scale"],
            pair_multiplier_overrides=baseline_variant["pair_multiplier_overrides"],
            signal_cfg=baseline_variant["signal_cfg"],
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
            linkage_method=baseline_variant["linkage"],
            delta_method=baseline_variant["delta_method"],
            max_files=baseline_variant["max_files"],
        )

        test_means = mean_metrics(test_rows)
        baseline_means = mean_metrics(baseline_test_rows)
        rows.append(
            {
                "scope": "dataset",
                "holdout": heldout_dataset,
                "n_train_builds": len(train),
                "n_test_builds": len(test),
                "selected_group": best_variant["group"],
                "selected_parameter": best_variant["parameter"],
                "selected_value": best_variant["value"],
                "train_mean_ari": train_best_ari,
                "test_mean_ari": test_means["ari"],
                "test_delta_vs_default": test_means["ari"] - baseline_means["ari"],
                "test_mean_nmi": test_means["nmi"],
                "test_mean_bcubed_f1": test_means["bcubed_f1"],
            }
        )

    grouped: dict[str, list[BuildArtifact]] = {}
    for art in artifacts:
        project_key = f"{art.source}:{art.project or 'UNKNOWN'}"
        grouped.setdefault(project_key, []).append(art)

    eligible_projects = [(project, rows_) for project, rows_ in grouped.items() if len(rows_) >= min_project_builds]
    eligible_projects.sort(key=lambda item: len(item[1]), reverse=True)
    if max_heldout_projects > 0:
        eligible_projects = eligible_projects[:max_heldout_projects]

    for project_key, test in eligible_projects:
        train = [a for a in artifacts if f"{a.source}:{a.project or 'UNKNOWN'}" != project_key]
        if not train or not test:
            continue
        best_variant, train_best_ari = choose_best_variant_on_subset(
            train,
            variants,
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
        )
        test_rows = evaluate_variant(
            test,
            signal_weights=best_variant["weights"],
            lambda_penalty=best_variant["lambda_penalty"],
            threshold_cfg=best_variant["threshold_cfg"],
            multiplier_scale=best_variant["multiplier_scale"],
            pair_multiplier_overrides=best_variant["pair_multiplier_overrides"],
            signal_cfg=best_variant["signal_cfg"],
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
            linkage_method=best_variant["linkage"],
            delta_method=best_variant["delta_method"],
            max_files=best_variant["max_files"],
        )
        baseline_test_rows = evaluate_variant(
            test,
            signal_weights=baseline_variant["weights"],
            lambda_penalty=baseline_variant["lambda_penalty"],
            threshold_cfg=baseline_variant["threshold_cfg"],
            multiplier_scale=baseline_variant["multiplier_scale"],
            pair_multiplier_overrides=baseline_variant["pair_multiplier_overrides"],
            signal_cfg=baseline_variant["signal_cfg"],
            baseline_signal_cfg=baseline_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
            linkage_method=baseline_variant["linkage"],
            delta_method=baseline_variant["delta_method"],
            max_files=baseline_variant["max_files"],
        )
        test_means = mean_metrics(test_rows)
        baseline_means = mean_metrics(baseline_test_rows)
        rows.append(
            {
                "scope": "project",
                "holdout": project_key,
                "n_train_builds": len(train),
                "n_test_builds": len(test),
                "selected_group": best_variant["group"],
                "selected_parameter": best_variant["parameter"],
                "selected_value": best_variant["value"],
                "train_mean_ari": train_best_ari,
                "test_mean_ari": test_means["ari"],
                "test_delta_vs_default": test_means["ari"] - baseline_means["ari"],
                "test_mean_nmi": test_means["nmi"],
                "test_mean_bcubed_f1": test_means["bcubed_f1"],
            }
        )

    return rows


def parse_numeric_variant_value(value: str | float | int) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.startswith("x"):
        text = text[1:]
    try:
        return float(text)
    except ValueError:
        return None


def write_sensitivity_plots(
    all_rows: list[dict[str, str | float]],
    output_dir: Path,
) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping sensitivity plots (matplotlib unavailable): {exc}")
        return []

    baseline_row = next((r for r in all_rows if r["group"] == "baseline"), None)
    baseline_ari = float(baseline_row["mean_ari"]) if baseline_row else None

    grouped: dict[tuple[str, str], list[tuple[float, float]]] = {}
    for row in all_rows:
        x = parse_numeric_variant_value(row["value"])
        if x is None:
            continue
        key = (str(row["group"]), str(row["parameter"]))
        grouped.setdefault(key, []).append((x, float(row["mean_ari"])))

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for (group, parameter), points in grouped.items():
        if len(points) < 2:
            continue
        points.sort(key=lambda item: item[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.figure(figsize=(6, 3.5))
        plt.plot(xs, ys, marker="o")
        if baseline_ari is not None:
            plt.axhline(baseline_ari, linestyle="--", linewidth=1.0, color="gray")
        plt.title(f"{group} / {parameter}")
        plt.xlabel("Variant value")
        plt.ylabel("Mean ARI")
        plt.grid(alpha=0.25)
        safe_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{group}_{parameter}").strip("_")
        out_path = output_dir / f"{safe_name}.png"
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        plt.close()
        written.append(out_path)

    return written


def holm_bonferroni_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        candidate = min(1.0, (m - rank) * raw)
        running_max = max(running_max, candidate)
        adjusted[idx] = min(1.0, running_max)
    return adjusted


def paired_effect_statistics(
    baseline_values: list[float],
    variant_values: list[float],
) -> dict[str, float | int]:
    baseline = np.array(baseline_values, dtype=float)
    variant = np.array(variant_values, dtype=float)
    if baseline.shape != variant.shape:
        raise ValueError("Baseline and variant arrays must have same shape.")
    diffs = variant - baseline
    n_pairs = int(diffs.size)
    n_nonzero = int(np.sum(np.abs(diffs) > 1e-12))
    wins = int(np.sum(diffs > 0))
    ties = int(np.sum(np.abs(diffs) <= 1e-12))
    common_language = (wins + 0.5 * ties) / max(n_pairs, 1)

    nonzero = diffs[np.abs(diffs) > 1e-12]
    if nonzero.size == 0:
        return {
            "mean_delta_per_build": float(np.mean(diffs)) if n_pairs else 0.0,
            "median_delta_per_build": float(np.median(diffs)) if n_pairs else 0.0,
            "common_language_effect": float(common_language),
            "rank_biserial": 0.0,
            "wilcoxon_statistic": 0.0,
            "wilcoxon_p": 1.0,
            "n_pairs": n_pairs,
            "n_nonzero_pairs": 0,
            "wins": wins,
            "losses": int(np.sum(diffs < 0)),
            "ties": ties,
        }

    try:
        test = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
        wilcoxon_statistic = float(test.statistic)
        wilcoxon_p = float(test.pvalue)
    except Exception:
        wilcoxon_statistic = 0.0
        wilcoxon_p = 1.0

    ranks = rankdata(np.abs(nonzero), method="average")
    w_plus = float(np.sum(ranks[nonzero > 0]))
    w_minus = float(np.sum(ranks[nonzero < 0]))
    denom = w_plus + w_minus
    rank_biserial = 0.0 if denom <= 0 else (w_plus - w_minus) / denom

    return {
        "mean_delta_per_build": float(np.mean(diffs)),
        "median_delta_per_build": float(np.median(diffs)),
        "common_language_effect": float(common_language),
        "rank_biserial": float(rank_biserial),
        "wilcoxon_statistic": wilcoxon_statistic,
        "wilcoxon_p": wilcoxon_p,
        "n_pairs": n_pairs,
        "n_nonzero_pairs": n_nonzero,
        "wins": wins,
        "losses": int(np.sum(diffs < 0)),
        "ties": ties,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HyTri sensitivity analysis runner")
    parser.add_argument("--num-builds", type=int, default=50, help="Builds per dataset")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/sensitivity/sensitivity_results.csv"),
        help="Output CSV path",
    )
    parser.add_argument(
        "--summary-txt",
        type=Path,
        default=Path("results/sensitivity/sensitivity_summary.txt"),
        help="Output summary text path",
    )
    parser.add_argument(
        "--robustness-csv",
        type=Path,
        default=Path("results/sensitivity/robustness_results.csv"),
        help="Output CSV path for leave-out robustness analysis",
    )
    parser.add_argument(
        "--effect-sizes-csv",
        type=Path,
        default=Path("results/sensitivity/sensitivity_effect_sizes.csv"),
        help="Output CSV path for paired effect-size statistics",
    )
    parser.add_argument(
        "--min-project-builds",
        type=int,
        default=4,
        help="Minimum builds in a project to include leave-project-out tuning",
    )
    parser.add_argument(
        "--max-heldout-projects",
        type=int,
        default=10,
        help="Max number of projects to include in leave-project-out analysis (0 = all)",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=Path("results/sensitivity/plots"),
        help="Directory for generated sensitivity plots",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Disable automatic sensitivity plot generation",
    )
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    datasets = resolve_datasets(project_root)
    if len(datasets) < 3:
        raise RuntimeError(f"Expected 3 datasets, found: {list(datasets.keys())}")

    random.seed(args.seed)

    selected_builds: list[dict] = []
    for name, path in datasets.items():
        sampled = load_multifault_builds_sampled(path, args.num_builds, seed=args.seed)
        for b in sampled:
            b["source"] = name
        print(f"Loaded {len(sampled)} builds from {name}")
        selected_builds.extend(sampled)

    if not selected_builds:
        raise RuntimeError("No builds loaded for sensitivity analysis.")

    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError("OPENAI_API_KEY is required.")

    openai_client = OpenAIClient(api_key=openai_key)
    signal_computer = MultiSignalComputer(openai_client=openai_client, use_semantic=True)
    hypothesis_generator = HypothesisGenerator(openai_client=openai_client, batch_size=10)

    artifacts = precompute_build_artifacts(selected_builds, signal_computer, hypothesis_generator)

    base_weights = normalize_weights(AggregationEnsemble.DEFAULT_WEIGHTS.copy())
    base_threshold = ThresholdConfig(
        dt_stack=0.57,
        dt_message=0.73,
        dt_test=0.50,
        agreement_factor=0.10,
        size_factor=0.05,
        min_threshold=0.2,
        max_threshold=0.8,
    )
    base_signal_cfg = SignalVariantConfig()
    base_linkage = "average"
    base_lambda = 0.2
    base_multiplier_scale = 1.0
    base_delta = 7
    base_max_files = 3

    def threshold_cfg(**overrides: float) -> ThresholdConfig:
        return ThresholdConfig(
            dt_stack=overrides.get("dt_stack", base_threshold.dt_stack),
            dt_message=overrides.get("dt_message", base_threshold.dt_message),
            dt_test=overrides.get("dt_test", base_threshold.dt_test),
            agreement_factor=overrides.get("agreement_factor", base_threshold.agreement_factor),
            size_factor=overrides.get("size_factor", base_threshold.size_factor),
            min_threshold=overrides.get("min_threshold", base_threshold.min_threshold),
            max_threshold=overrides.get("max_threshold", base_threshold.max_threshold),
        )

    def make_variant(
        *,
        group: str,
        parameter: str,
        value: str,
        weights: dict[str, float] | None = None,
        lambda_penalty: float | None = None,
        threshold_cfg_value: ThresholdConfig | None = None,
        multiplier_scale: float | None = None,
        pair_multiplier_overrides: dict[tuple[str, str], dict[str, float]] | None = None,
        signal_cfg: SignalVariantConfig | None = None,
        linkage: str | None = None,
        delta_method: int | None = None,
        max_files: int | None = None,
    ) -> dict[str, Any]:
        return {
            "group": group,
            "parameter": parameter,
            "value": value,
            "weights": weights if weights is not None else base_weights,
            "lambda_penalty": lambda_penalty if lambda_penalty is not None else base_lambda,
            "threshold_cfg": threshold_cfg_value if threshold_cfg_value is not None else base_threshold,
            "multiplier_scale": multiplier_scale if multiplier_scale is not None else base_multiplier_scale,
            "pair_multiplier_overrides": pair_multiplier_overrides,
            "signal_cfg": signal_cfg if signal_cfg is not None else base_signal_cfg,
            "linkage": linkage if linkage is not None else base_linkage,
            "delta_method": delta_method if delta_method is not None else base_delta,
            "max_files": max_files if max_files is not None else base_max_files,
        }

    variants: list[dict[str, Any]] = [
        make_variant(group="baseline", parameter="default", value="default")
    ]

    for lam in [0.0, 0.1, 0.3, 0.4]:
        variants.append(make_variant(group="penalty", parameter="disagreement_penalty", value=f"{lam:.2f}", lambda_penalty=lam))

    # Non-unit type-multiplier scales from the paper grid {0.25, 0.5, 0.75, 1.00, 1.25, 1.5};
    # 1.00 is the default (covered by the baseline variant).
    for scale in [0.25, 0.5, 0.75, 1.25, 1.5]:
        variants.append(make_variant(group="multipliers", parameter="nonunit_scale", value=f"{scale:.2f}", multiplier_scale=scale))

    failure_type_multiplier_sweeps = [
        (("exception", "exception"), "pdm", [1.2, 1.8]),
        (("assertion", "assertion"), "pdm", [0.2, 0.4]),
        (("exception", "assertion"), "pdm", [0.3, 0.7]),
        (("assertion", "exception"), "pdm", [0.3, 0.7]),
    ]
    for pair, signal_name, values in failure_type_multiplier_sweeps:
        for raw in values:
            label = f"{pair[0]}->{pair[1]}::{signal_name}"
            variants.append(
                make_variant(
                    group="failure_type_multipliers",
                    parameter=label,
                    value=f"{raw:.2f}",
                    pair_multiplier_overrides={pair: {signal_name: raw}},
                )
            )

    weight_keys = ["pdm", "overlap", "diff_line", "semantic", "message", "hypothesis", "test_name"]
    for key in weight_keys:
        for factor in [0.8, 1.2]:
            adjusted = base_weights.copy()
            adjusted[key] = adjusted[key] * factor
            adjusted = normalize_weights(adjusted)
            variants.append(make_variant(group="weights", parameter=key, value=f"x{factor:.2f}", weights=adjusted))

    for dt_stack in [0.52, 0.62]:
        variants.append(make_variant(group="thresholds", parameter="dt_stack", value=f"{dt_stack:.2f}", threshold_cfg_value=threshold_cfg(dt_stack=dt_stack)))
    for dt_message in [0.68, 0.78]:
        variants.append(make_variant(group="thresholds", parameter="dt_message", value=f"{dt_message:.2f}", threshold_cfg_value=threshold_cfg(dt_message=dt_message)))
    for dt_test in [0.45, 0.55]:
        variants.append(make_variant(group="thresholds", parameter="dt_test", value=f"{dt_test:.2f}", threshold_cfg_value=threshold_cfg(dt_test=dt_test)))
    for factor in [0.05, 0.15]:
        variants.append(make_variant(group="thresholds", parameter="agreement_factor", value=f"{factor:.2f}", threshold_cfg_value=threshold_cfg(agreement_factor=factor)))
    for factor in [0.03, 0.07]:
        variants.append(make_variant(group="thresholds", parameter="size_factor", value=f"{factor:.2f}", threshold_cfg_value=threshold_cfg(size_factor=factor)))
    for min_threshold in [0.15, 0.25]:
        variants.append(make_variant(group="thresholds", parameter="min_threshold", value=f"{min_threshold:.2f}", threshold_cfg_value=threshold_cfg(min_threshold=min_threshold)))
    for max_threshold in [0.75, 0.85]:
        variants.append(make_variant(group="thresholds", parameter="max_threshold", value=f"{max_threshold:.2f}", threshold_cfg_value=threshold_cfg(max_threshold=max_threshold)))

    for linkage in ["complete", "single"]:
        variants.append(make_variant(group="clustering", parameter="linkage", value=linkage, linkage=linkage))

    for delta in [5, 9]:
        variants.append(make_variant(group="complexity_gate", parameter="changed_method_threshold", value=str(delta), delta_method=delta))
    for max_files in [2, 4]:
        variants.append(make_variant(group="complexity_gate", parameter="changed_file_threshold", value=str(max_files), max_files=max_files))

    for tolerance in [3, 7]:
        variants.append(
            make_variant(
                group="diff_line_correlation",
                parameter="line_tolerance",
                value=str(tolerance),
                signal_cfg=SignalVariantConfig(
                    diff_line_tolerance=tolerance,
                    suspected_change_match_similarity=base_signal_cfg.suspected_change_match_similarity,
                    message_confidence_cap=base_signal_cfg.message_confidence_cap,
                    test_name_confidence_cap=base_signal_cfg.test_name_confidence_cap,
                    semantic_confidence=base_signal_cfg.semantic_confidence,
                    hypothesis_confidence=base_signal_cfg.hypothesis_confidence,
                ),
            )
        )

    for eq6_score in [0.8, 1.0]:
        variants.append(
            make_variant(
                group="eq6",
                parameter="suspected_change_match_similarity",
                value=f"{eq6_score:.2f}",
                signal_cfg=SignalVariantConfig(
                    diff_line_tolerance=base_signal_cfg.diff_line_tolerance,
                    suspected_change_match_similarity=eq6_score,
                    message_confidence_cap=base_signal_cfg.message_confidence_cap,
                    test_name_confidence_cap=base_signal_cfg.test_name_confidence_cap,
                    semantic_confidence=base_signal_cfg.semantic_confidence,
                    hypothesis_confidence=base_signal_cfg.hypothesis_confidence,
                ),
            )
        )

    for cap in [0.5, 0.7]:
        variants.append(
            make_variant(
                group="confidence_caps",
                parameter="message_confidence_cap",
                value=f"{cap:.2f}",
                signal_cfg=SignalVariantConfig(
                    diff_line_tolerance=base_signal_cfg.diff_line_tolerance,
                    suspected_change_match_similarity=base_signal_cfg.suspected_change_match_similarity,
                    message_confidence_cap=cap,
                    test_name_confidence_cap=base_signal_cfg.test_name_confidence_cap,
                    semantic_confidence=base_signal_cfg.semantic_confidence,
                    hypothesis_confidence=base_signal_cfg.hypothesis_confidence,
                ),
            )
        )

    for cap in [0.4, 0.6]:
        variants.append(
            make_variant(
                group="confidence_caps",
                parameter="test_name_confidence_cap",
                value=f"{cap:.2f}",
                signal_cfg=SignalVariantConfig(
                    diff_line_tolerance=base_signal_cfg.diff_line_tolerance,
                    suspected_change_match_similarity=base_signal_cfg.suspected_change_match_similarity,
                    message_confidence_cap=base_signal_cfg.message_confidence_cap,
                    test_name_confidence_cap=cap,
                    semantic_confidence=base_signal_cfg.semantic_confidence,
                    hypothesis_confidence=base_signal_cfg.hypothesis_confidence,
                ),
            )
        )

    for conf in [0.6, 0.8]:
        variants.append(
            make_variant(
                group="llm_confidence",
                parameter="semantic_confidence",
                value=f"{conf:.2f}",
                signal_cfg=SignalVariantConfig(
                    diff_line_tolerance=base_signal_cfg.diff_line_tolerance,
                    suspected_change_match_similarity=base_signal_cfg.suspected_change_match_similarity,
                    message_confidence_cap=base_signal_cfg.message_confidence_cap,
                    test_name_confidence_cap=base_signal_cfg.test_name_confidence_cap,
                    semantic_confidence=conf,
                    hypothesis_confidence=base_signal_cfg.hypothesis_confidence,
                ),
            )
        )

    for conf in [0.7, 0.9]:
        variants.append(
            make_variant(
                group="llm_confidence",
                parameter="hypothesis_confidence",
                value=f"{conf:.2f}",
                signal_cfg=SignalVariantConfig(
                    diff_line_tolerance=base_signal_cfg.diff_line_tolerance,
                    suspected_change_match_similarity=base_signal_cfg.suspected_change_match_similarity,
                    message_confidence_cap=base_signal_cfg.message_confidence_cap,
                    test_name_confidence_cap=base_signal_cfg.test_name_confidence_cap,
                    semantic_confidence=base_signal_cfg.semantic_confidence,
                    hypothesis_confidence=conf,
                ),
            )
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.summary_txt.parent.mkdir(parents=True, exist_ok=True)
    args.robustness_csv.parent.mkdir(parents=True, exist_ok=True)
    args.effect_sizes_csv.parent.mkdir(parents=True, exist_ok=True)
    if not args.skip_plots:
        args.plots_dir.mkdir(parents=True, exist_ok=True)

    signal_ops_cache: dict[SignalVariantConfig, MultiSignalComputer] = {}
    signal_variant_cache: dict[tuple[str, SignalVariantConfig], SimilaritySignals] = {}

    all_rows: list[dict[str, str | float]] = []
    variant_ari_rows: list[list[float]] = []
    baseline_ari = None

    for i, v in enumerate(variants, start=1):
        print(
            f"[{i}/{len(variants)}] {v['group']}::{v['parameter']}={v['value']} "
            f"(lambda={v['lambda_penalty']}, linkage={v['linkage']})"
        )
        out = evaluate_variant(
            artifacts,
            signal_weights=v["weights"],
            lambda_penalty=v["lambda_penalty"],
            threshold_cfg=v["threshold_cfg"],
            multiplier_scale=v["multiplier_scale"],
            pair_multiplier_overrides=v["pair_multiplier_overrides"],
            signal_cfg=v["signal_cfg"],
            baseline_signal_cfg=base_signal_cfg,
            signal_ops_cache=signal_ops_cache,
            signal_variant_cache=signal_variant_cache,
            linkage_method=v["linkage"],
            delta_method=v["delta_method"],
            max_files=v["max_files"],
        )
        means = mean_metrics(out)
        variant_ari_rows.append([float(x) for x in out["ari"]])
        _, lo, hi = bootstrap_ci(out["ari"], seed=args.seed)

        if v["group"] == "baseline":
            baseline_ari = means["ari"]

        delta_ari = 0.0 if baseline_ari is None else means["ari"] - baseline_ari
        all_rows.append(
            {
                "group": v["group"],
                "parameter": v["parameter"],
                "value": v["value"],
                "mean_ari": means["ari"],
                "delta_ari_vs_default": delta_ari,
                "ari_ci_low": lo,
                "ari_ci_high": hi,
                "mean_nmi": means["nmi"],
                "mean_bcubed_f1": means["bcubed_f1"],
                "n_builds": len(artifacts),
            }
        )

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "parameter",
                "value",
                "mean_ari",
                "delta_ari_vs_default",
                "ari_ci_low",
                "ari_ci_high",
                "mean_nmi",
                "mean_bcubed_f1",
                "n_builds",
            ],
        )
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    if not variant_ari_rows:
        raise RuntimeError("No variant results available for effect-size analysis.")
    baseline_ari_by_build = variant_ari_rows[0]
    effect_rows: list[dict[str, str | float | int]] = []
    raw_p_values: list[float] = []
    raw_p_effect_idx: list[int] = []
    for idx, (row, variant_ari_by_build) in enumerate(zip(all_rows, variant_ari_rows)):
        base = {
            "group": row["group"],
            "parameter": row["parameter"],
            "value": row["value"],
            "mean_ari": row["mean_ari"],
            "delta_ari_vs_default": row["delta_ari_vs_default"],
        }
        if idx == 0:
            base.update(
                {
                    "mean_delta_per_build": 0.0,
                    "median_delta_per_build": 0.0,
                    "common_language_effect": 0.5,
                    "rank_biserial": 0.0,
                    "wilcoxon_statistic": 0.0,
                    "wilcoxon_p": 1.0,
                    "wilcoxon_p_holm": 1.0,
                    "n_pairs": len(baseline_ari_by_build),
                    "n_nonzero_pairs": 0,
                    "wins": 0,
                    "losses": 0,
                    "ties": len(baseline_ari_by_build),
                }
            )
            effect_rows.append(base)
            continue

        stats = paired_effect_statistics(baseline_ari_by_build, variant_ari_by_build)
        base.update(stats)
        base["wilcoxon_p_holm"] = 1.0
        raw_p_values.append(float(stats["wilcoxon_p"]))
        raw_p_effect_idx.append(len(effect_rows))
        effect_rows.append(base)

    adjusted = holm_bonferroni_adjust(raw_p_values)
    for adj, idx in zip(adjusted, raw_p_effect_idx):
        effect_rows[idx]["wilcoxon_p_holm"] = float(adj)

    with open(args.effect_sizes_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "group",
                "parameter",
                "value",
                "mean_ari",
                "delta_ari_vs_default",
                "mean_delta_per_build",
                "median_delta_per_build",
                "common_language_effect",
                "rank_biserial",
                "wilcoxon_statistic",
                "wilcoxon_p",
                "wilcoxon_p_holm",
                "n_pairs",
                "n_nonzero_pairs",
                "wins",
                "losses",
                "ties",
            ],
        )
        writer.writeheader()
        for row in effect_rows:
            writer.writerow(row)

    plot_files: list[Path] = []
    if not args.skip_plots:
        plot_files = write_sensitivity_plots(all_rows, args.plots_dir)

    robustness_rows = run_leave_out_robustness(
        artifacts,
        variants,
        baseline_signal_cfg=base_signal_cfg,
        signal_ops_cache=signal_ops_cache,
        signal_variant_cache=signal_variant_cache,
        min_project_builds=args.min_project_builds,
        max_heldout_projects=args.max_heldout_projects,
    )
    with open(args.robustness_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scope",
                "holdout",
                "n_train_builds",
                "n_test_builds",
                "selected_group",
                "selected_parameter",
                "selected_value",
                "train_mean_ari",
                "test_mean_ari",
                "test_delta_vs_default",
                "test_mean_nmi",
                "test_mean_bcubed_f1",
            ],
        )
        writer.writeheader()
        for row in robustness_rows:
            writer.writerow(row)

    max_drop = min(all_rows, key=lambda r: float(r["delta_ari_vs_default"]))
    max_gain = max(all_rows, key=lambda r: float(r["delta_ari_vs_default"]))
    worst_nonbaseline = min(
        [r for r in all_rows if r["group"] != "baseline"],
        key=lambda r: float(r["delta_ari_vs_default"]),
    )

    with open(args.summary_txt, "w") as sf:
        sf.write("HyTri sensitivity summary\n")
        sf.write(f"Builds per dataset: {args.num_builds}\n")
        sf.write(f"Datasets: {', '.join(datasets.keys())}\n")
        sf.write(f"Total evaluated builds: {len(artifacts)}\n\n")
        sf.write(f"Default mean ARI: {baseline_ari:.4f}\n")
        sf.write(
            "Worst non-baseline delta ARI: "
            f"{float(worst_nonbaseline['delta_ari_vs_default']):.4f} "
            f"({worst_nonbaseline['group']}::{worst_nonbaseline['parameter']}={worst_nonbaseline['value']})\n"
        )
        sf.write(
            "Best non-baseline delta ARI: "
            f"{float(max_gain['delta_ari_vs_default']):.4f} "
            f"({max_gain['group']}::{max_gain['parameter']}={max_gain['value']})\n"
        )
        sf.write(
            "Worst overall row: "
            f"{float(max_drop['delta_ari_vs_default']):.4f} "
            f"({max_drop['group']}::{max_drop['parameter']}={max_drop['value']})\n"
        )
        if robustness_rows:
            worst_robust = min(robustness_rows, key=lambda r: float(r["test_delta_vs_default"]))
            best_robust = max(robustness_rows, key=lambda r: float(r["test_delta_vs_default"]))
            sf.write("\nLeave-out robustness summary\n")
            sf.write(f"Rows: {len(robustness_rows)}\n")
            sf.write(
                "Worst held-out delta vs default: "
                f"{float(worst_robust['test_delta_vs_default']):.4f} "
                f"({worst_robust['scope']}::{worst_robust['holdout']})\n"
            )
            sf.write(
                "Best held-out delta vs default: "
                f"{float(best_robust['test_delta_vs_default']):.4f} "
                f"({best_robust['scope']}::{best_robust['holdout']})\n"
            )
        holm_significant = sum(
            1
            for row in effect_rows
            if row["group"] != "baseline" and float(row["wilcoxon_p_holm"]) < 0.05
        )
        sf.write("\nPaired effect-size analysis\n")
        sf.write(f"Rows: {len(effect_rows)}\n")
        sf.write(f"Holm-corrected Wilcoxon significant variants (p<0.05): {holm_significant}\n")
        sf.write(f"Effect-size CSV: {args.effect_sizes_csv}\n")
        if not args.skip_plots:
            sf.write(f"\nSensitivity plot files: {len(plot_files)} (dir: {args.plots_dir})\n")

    print(f"\nWrote: {args.output_csv}")
    if not args.skip_plots:
        print(f"Wrote {len(plot_files)} plot files under: {args.plots_dir}")
    print(f"Wrote: {args.effect_sizes_csv}")
    print(f"Wrote: {args.robustness_csv}")
    print(f"Wrote: {args.summary_txt}")


if __name__ == "__main__":
    main()
