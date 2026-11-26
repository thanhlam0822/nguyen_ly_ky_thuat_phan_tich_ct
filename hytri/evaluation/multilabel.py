"""Multi-label clustering evaluation for transplantation benchmarks.

When a failing test can be caused by multiple coexisting bugs, ground truth is a
set Y(test) per failure. Clustering is evaluated with overlap-aware pairwise metrics.
"""

from __future__ import annotations

from typing import Iterable


def get_multilabel_ground_truth_sets(failures: list[dict]) -> list[set[str]]:
    """Extract per-failure ground-truth bug ID sets from build failure dicts."""
    sets: list[set[str]] = []
    for f in failures:
        if "ground_truth_bugs" in f and f["ground_truth_bugs"]:
            bugs = {str(b) for b in f["ground_truth_bugs"]}
        elif "ground_truth_bug" in f:
            bugs = {str(f["ground_truth_bug"])}
        elif "ground_truth" in f:
            gt = f["ground_truth"]
            rid = gt.get("root_cause_id", gt.get("original_bug_id"))
            bugs = {str(rid)} if rid else set()
        else:
            bugs = set()
        sets.append(bugs)
    return sets


def same_multilabel_gt(a: set[str], b: set[str]) -> bool:
    """True if two failures should be in the same cluster under multi-label GT."""
    if not a or not b:
        return False
    return bool(a & b)


def compute_pairwise_multilabel_metrics(
    true_sets: list[set[str]],
    pred_labels: list[int],
) -> dict[str, float]:
    """Pairwise precision/recall/F1 using overlap-aware ground truth.

    For each pair (i, j):
      same_gt  = |Y(i) ∩ Y(j)| > 0
      same_pred = pred_labels[i] == pred_labels[j]
    """
    n = len(true_sets)
    if n < 2 or len(pred_labels) != n:
        return {
            "pairwise_precision": 0.0,
            "pairwise_recall": 0.0,
            "pairwise_f1": 0.0,
            "pairwise_tp": 0.0,
            "pairwise_fp": 0.0,
            "pairwise_fn": 0.0,
            "pairwise_pairs": 0.0,
        }

    tp = fp = fn = 0
    for i in range(n):
        for j in range(i + 1, n):
            gt_ij = same_multilabel_gt(true_sets[i], true_sets[j])
            pred_ij = pred_labels[i] == pred_labels[j]
            if gt_ij and pred_ij:
                tp += 1
            elif not gt_ij and pred_ij:
                fp += 1
            elif gt_ij and not pred_ij:
                fn += 1

    num_pairs = n * (n - 1) // 2
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "pairwise_precision": precision,
        "pairwise_recall": recall,
        "pairwise_f1": f1,
        "pairwise_tp": float(tp),
        "pairwise_fp": float(fp),
        "pairwise_fn": float(fn),
        "pairwise_pairs": float(num_pairs),
    }


def multilabel_overlap_prevalence(true_sets: list[set[str]]) -> dict[str, float]:
    """Fraction of failures with |Y| > 1."""
    if not true_sets:
        return {"multilabel_fraction": 0.0, "num_multilabel_failures": 0.0}
    multi = sum(1 for s in true_sets if len(s) > 1)
    return {
        "multilabel_fraction": multi / len(true_sets),
        "num_multilabel_failures": float(multi),
    }


def filter_single_label_indices(true_sets: list[set[str]]) -> list[int]:
    """Indices where ground truth is exactly one bug (comparability subset)."""
    return [i for i, s in enumerate(true_sets) if len(s) == 1]


def subset_labels(labels: list[int], indices: Iterable[int]) -> list[int]:
    """Reindex labels to 0..k-1 on the given index subset."""
    idx_list = list(indices)
    if not idx_list:
        return []
    mapping = {old: new for new, old in enumerate(idx_list)}
    return [mapping[i] for i in idx_list]
