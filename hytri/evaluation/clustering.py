"""Clustering Evaluation Metrics.

Implements standard clustering quality metrics for evaluating failure triage.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from hytri.models import Cluster, FailureInstance


@dataclass
class ClusteringMetrics:
    """Results from clustering evaluation."""

    ari: float  # Adjusted Rand Index
    nmi: float  # Normalized Mutual Information
    v_measure: float  # V-measure (harmonic mean of homogeneity and completeness)
    purity: float  # Cluster purity
    cluster_count_error: int  # Difference in number of clusters
    num_predicted_clusters: int
    num_true_clusters: int


def evaluate_clustering(
    clusters: list[Cluster],
    failures: list[FailureInstance],
    noise_failure_ids: list[str] | None = None,
) -> ClusteringMetrics:
    """Evaluate clustering quality against ground truth.

    Args:
        clusters: Predicted clusters.
        failures: All failure instances with ground_truth set.
        noise_failure_ids: IDs of failures not assigned to any cluster.

    Returns:
        ClusteringMetrics with all evaluation scores.
    """
    # Build label arrays
    predicted_labels, true_labels = _build_label_arrays(
        clusters, failures, noise_failure_ids
    )

    if len(predicted_labels) == 0:
        return ClusteringMetrics(
            ari=0.0,
            nmi=0.0,
            v_measure=0.0,
            purity=0.0,
            cluster_count_error=0,
            num_predicted_clusters=0,
            num_true_clusters=0,
        )

    # Compute metrics
    ari = adjusted_rand_score(true_labels, predicted_labels)
    nmi = normalized_mutual_info_score(true_labels, predicted_labels)
    v_measure = v_measure_score(true_labels, predicted_labels)
    purity = _compute_purity(true_labels, predicted_labels)

    # Count clusters
    num_predicted = len(set(predicted_labels)) - (1 if -1 in predicted_labels else 0)
    num_true = len(set(true_labels)) - (1 if -1 in true_labels else 0)
    cluster_count_error = abs(num_predicted - num_true)

    return ClusteringMetrics(
        ari=ari,
        nmi=nmi,
        v_measure=v_measure,
        purity=purity,
        cluster_count_error=cluster_count_error,
        num_predicted_clusters=num_predicted,
        num_true_clusters=num_true,
    )


def _build_label_arrays(
    clusters: list[Cluster],
    failures: list[FailureInstance],
    noise_failure_ids: list[str] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build predicted and true label arrays.

    Args:
        clusters: Predicted clusters.
        failures: All failures with ground truth.
        noise_failure_ids: IDs of noise failures.

    Returns:
        Tuple of (predicted_labels, true_labels) arrays.
    """
    noise_ids = set(noise_failure_ids or [])

    # Build failure_id -> cluster_id mapping
    failure_to_cluster: dict[str, int] = {}
    for i, cluster in enumerate(clusters):
        for failure_id in cluster.failure_ids:
            failure_to_cluster[failure_id] = i

    for failure_id in noise_ids:
        failure_to_cluster[failure_id] = -1

    # Build true label mapping
    root_cause_to_label: dict[str, int] = {}
    next_label = 0

    predicted_labels = []
    true_labels = []

    for failure in failures:
        # Get predicted label
        pred_label = failure_to_cluster.get(failure.failure_id, -1)
        predicted_labels.append(pred_label)

        # Get true label
        if failure.ground_truth and failure.ground_truth.root_cause_id:
            root_cause = failure.ground_truth.root_cause_id
            if root_cause not in root_cause_to_label:
                root_cause_to_label[root_cause] = next_label
                next_label += 1
            true_labels.append(root_cause_to_label[root_cause])
        else:
            true_labels.append(-1)

    return np.array(predicted_labels), np.array(true_labels)


def _compute_purity(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
) -> float:
    """Compute cluster purity.

    Purity measures how pure each cluster is with respect to ground truth.
    Higher is better (1.0 = perfect).
    """
    if len(predicted_labels) == 0:
        return 0.0

    # Get unique predicted clusters (excluding noise)
    unique_clusters = set(predicted_labels) - {-1}
    if not unique_clusters:
        return 0.0

    total_correct = 0
    total_points = 0

    for cluster_id in unique_clusters:
        mask = predicted_labels == cluster_id
        cluster_true_labels = true_labels[mask]

        if len(cluster_true_labels) == 0:
            continue

        # Most common true label in this cluster
        unique, counts = np.unique(cluster_true_labels, return_counts=True)
        majority_count = counts.max()

        total_correct += majority_count
        total_points += len(cluster_true_labels)

    return total_correct / total_points if total_points > 0 else 0.0


def compare_clustering_methods(
    results: dict[str, tuple[list[Cluster], list[str]]],
    failures: list[FailureInstance],
) -> dict[str, ClusteringMetrics]:
    """Compare multiple clustering methods.

    Args:
        results: Dict of method_name -> (clusters, noise_ids).
        failures: All failure instances.

    Returns:
        Dict of method_name -> ClusteringMetrics.
    """
    metrics = {}
    for method_name, (clusters, noise_ids) in results.items():
        metrics[method_name] = evaluate_clustering(clusters, failures, noise_ids)
    return metrics


