"""Prioritization Evaluation Metrics.

Implements metrics for evaluating cluster prioritization quality.
"""

import math
from dataclasses import dataclass

from hytri.models import Cluster, FailureInstance


@dataclass
class PrioritizationMetrics:
    """Results from prioritization evaluation."""

    ndcg_at_1: float
    ndcg_at_3: float
    ndcg_at_5: float
    top_1_coverage: float  # Fraction of failures in top 1 cluster
    top_3_coverage: float  # Fraction of failures in top 3 clusters
    top_5_coverage: float  # Fraction of failures in top 5 clusters
    time_to_resolution_50: float  # Clusters needed to cover 50% of failures
    time_to_resolution_80: float  # Clusters needed to cover 80% of failures


def evaluate_prioritization(
    clusters: list[Cluster],
    priority_queue: list[str],
    failures: list[FailureInstance],
    relevance_weights: dict[str, float] | None = None,
) -> PrioritizationMetrics:
    """Evaluate prioritization quality.

    Args:
        clusters: List of clusters.
        priority_queue: Cluster IDs in predicted priority order.
        failures: All failure instances.
        relevance_weights: Optional cluster_id -> relevance weight mapping.
            If not provided, uses cluster size as relevance.

    Returns:
        PrioritizationMetrics with all scores.
    """
    if not clusters or not priority_queue:
        return PrioritizationMetrics(
            ndcg_at_1=0.0,
            ndcg_at_3=0.0,
            ndcg_at_5=0.0,
            top_1_coverage=0.0,
            top_3_coverage=0.0,
            top_5_coverage=0.0,
            time_to_resolution_50=float("inf"),
            time_to_resolution_80=float("inf"),
        )

    cluster_map = {c.cluster_id: c for c in clusters}
    total_failures = len(failures)

    # Compute relevance scores if not provided
    if relevance_weights is None:
        relevance_weights = {
            cluster.cluster_id: len(cluster.failure_ids)
            for cluster in clusters
        }

    # Order clusters by priority queue
    ordered_clusters = [
        cluster_map[cid] for cid in priority_queue if cid in cluster_map
    ]

    # Compute nDCG at various k
    ndcg_at_1 = _compute_ndcg(ordered_clusters, relevance_weights, k=1)
    ndcg_at_3 = _compute_ndcg(ordered_clusters, relevance_weights, k=3)
    ndcg_at_5 = _compute_ndcg(ordered_clusters, relevance_weights, k=5)

    # Compute coverage metrics
    top_1_coverage = _compute_coverage(ordered_clusters[:1], total_failures)
    top_3_coverage = _compute_coverage(ordered_clusters[:3], total_failures)
    top_5_coverage = _compute_coverage(ordered_clusters[:5], total_failures)

    # Compute time-to-resolution
    ttr_50 = _compute_time_to_resolution(ordered_clusters, total_failures, 0.5)
    ttr_80 = _compute_time_to_resolution(ordered_clusters, total_failures, 0.8)

    return PrioritizationMetrics(
        ndcg_at_1=ndcg_at_1,
        ndcg_at_3=ndcg_at_3,
        ndcg_at_5=ndcg_at_5,
        top_1_coverage=top_1_coverage,
        top_3_coverage=top_3_coverage,
        top_5_coverage=top_5_coverage,
        time_to_resolution_50=ttr_50,
        time_to_resolution_80=ttr_80,
    )


def _compute_ndcg(
    ordered_clusters: list[Cluster],
    relevance_weights: dict[str, float],
    k: int,
) -> float:
    """Compute nDCG@k for cluster ordering.

    Uses cluster size (or custom weights) as relevance.
    """
    if not ordered_clusters or k <= 0:
        return 0.0

    # Get relevance values
    relevances = [
        relevance_weights.get(c.cluster_id, 0.0)
        for c in ordered_clusters[:k]
    ]

    # Compute DCG
    dcg = sum(
        rel / math.log2(i + 2)  # i+2 because positions are 1-indexed
        for i, rel in enumerate(relevances)
    )

    # Compute ideal DCG (sort by relevance descending)
    ideal_relevances = sorted(relevance_weights.values(), reverse=True)[:k]
    idcg = sum(
        rel / math.log2(i + 2)
        for i, rel in enumerate(ideal_relevances)
    )

    return dcg / idcg if idcg > 0 else 0.0


def _compute_coverage(
    clusters: list[Cluster],
    total_failures: int,
) -> float:
    """Compute fraction of failures covered by clusters."""
    if total_failures == 0:
        return 0.0

    covered = sum(len(c.failure_ids) for c in clusters)
    return min(covered / total_failures, 1.0)


def _compute_time_to_resolution(
    ordered_clusters: list[Cluster],
    total_failures: int,
    target_coverage: float,
) -> float:
    """Compute number of clusters needed to reach target coverage.

    Returns:
        Number of clusters (or inf if target cannot be reached).
    """
    if total_failures == 0:
        return 0.0

    target_count = target_coverage * total_failures
    cumulative = 0

    for i, cluster in enumerate(ordered_clusters, 1):
        cumulative += len(cluster.failure_ids)
        if cumulative >= target_count:
            return float(i)

    return float("inf")


def compute_mrr(
    clusters: list[Cluster],
    priority_queue: list[str],
    target_cluster_id: str,
) -> float:
    """Compute Mean Reciprocal Rank for a target cluster.

    Args:
        clusters: All clusters.
        priority_queue: Cluster IDs in priority order.
        target_cluster_id: The cluster we want to find.

    Returns:
        Reciprocal rank (1/position), or 0 if not found.
    """
    for i, cluster_id in enumerate(priority_queue, 1):
        if cluster_id == target_cluster_id:
            return 1.0 / i
    return 0.0


def compute_average_mrr(
    clusters: list[Cluster],
    priority_queue: list[str],
    target_cluster_ids: list[str],
) -> float:
    """Compute average MRR over multiple target clusters.

    Args:
        clusters: All clusters.
        priority_queue: Cluster IDs in priority order.
        target_cluster_ids: List of target clusters.

    Returns:
        Average MRR across all targets.
    """
    if not target_cluster_ids:
        return 0.0

    mrr_sum = sum(
        compute_mrr(clusters, priority_queue, target_id)
        for target_id in target_cluster_ids
    )
    return mrr_sum / len(target_cluster_ids)


