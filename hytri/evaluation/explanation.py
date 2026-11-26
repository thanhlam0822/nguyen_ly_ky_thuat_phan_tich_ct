"""Explanation Evaluation Metrics.

Implements metrics for evaluating the quality and groundedness of
cluster explanations.
"""

from dataclasses import dataclass

from hytri.explanation.verifier import EvidenceVerifier, VerificationResult
from hytri.models import Cluster, FailureInstance


@dataclass
class ExplanationMetrics:
    """Results from explanation evaluation."""

    avg_groundedness: float  # Average groundedness score across clusters
    min_groundedness: float  # Minimum groundedness score
    max_groundedness: float  # Maximum groundedness score
    verified_ratio: float  # Fraction of explanations that passed verification
    coverage: float  # Fraction of clusters with explanations
    avg_summary_length: float  # Average number of summary bullets
    avg_unsupported_claims: float  # Average unsupported claims per explanation


def evaluate_explanation_groundedness(
    clusters: list[Cluster],
    failures: list[FailureInstance],
    strict_mode: bool = False,
) -> ExplanationMetrics:
    """Evaluate explanation quality across all clusters.

    Args:
        clusters: List of clusters with explanations.
        failures: All failure instances.
        strict_mode: If True, requires all claims to be grounded.

    Returns:
        ExplanationMetrics with aggregate scores.
    """
    if not clusters:
        return ExplanationMetrics(
            avg_groundedness=0.0,
            min_groundedness=0.0,
            max_groundedness=0.0,
            verified_ratio=0.0,
            coverage=0.0,
            avg_summary_length=0.0,
            avg_unsupported_claims=0.0,
        )

    verifier = EvidenceVerifier(strict_mode=strict_mode)

    groundedness_scores = []
    verified_count = 0
    explained_count = 0
    summary_lengths = []
    unsupported_claims_counts = []

    for cluster in clusters:
        if cluster.explanation is None:
            continue

        explained_count += 1

        # Verify explanation
        result = verifier.verify(cluster, cluster.explanation, failures)

        groundedness_scores.append(result.groundedness_score)
        if result.is_valid:
            verified_count += 1

        summary_lengths.append(len(cluster.explanation.summary))
        unsupported_claims_counts.append(len(result.unsupported_claims))

    if not groundedness_scores:
        return ExplanationMetrics(
            avg_groundedness=0.0,
            min_groundedness=0.0,
            max_groundedness=0.0,
            verified_ratio=0.0,
            coverage=0.0,
            avg_summary_length=0.0,
            avg_unsupported_claims=0.0,
        )

    return ExplanationMetrics(
        avg_groundedness=sum(groundedness_scores) / len(groundedness_scores),
        min_groundedness=min(groundedness_scores),
        max_groundedness=max(groundedness_scores),
        verified_ratio=verified_count / explained_count if explained_count > 0 else 0.0,
        coverage=explained_count / len(clusters),
        avg_summary_length=sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0.0,
        avg_unsupported_claims=sum(unsupported_claims_counts) / len(unsupported_claims_counts) if unsupported_claims_counts else 0.0,
    )


def verify_single_explanation(
    cluster: Cluster,
    failures: list[FailureInstance],
    strict_mode: bool = False,
) -> VerificationResult:
    """Verify a single cluster's explanation.

    Args:
        cluster: Cluster with explanation.
        failures: All failure instances.
        strict_mode: If True, requires all claims to be grounded.

    Returns:
        VerificationResult with detailed verification info.
    """
    verifier = EvidenceVerifier(strict_mode=strict_mode)

    if cluster.explanation is None:
        from hytri.explanation.verifier import VerificationResult
        return VerificationResult(
            is_valid=False,
            groundedness_score=0.0,
            evidence_matches=[],
            unsupported_claims=["No explanation provided"],
            details={},
        )

    return verifier.verify(cluster, cluster.explanation, failures)


def get_explanation_quality_report(
    clusters: list[Cluster],
    failures: list[FailureInstance],
) -> str:
    """Generate a human-readable report on explanation quality.

    Args:
        clusters: List of clusters with explanations.
        failures: All failure instances.

    Returns:
        Formatted report string.
    """
    metrics = evaluate_explanation_groundedness(clusters, failures)

    lines = [
        "=== Explanation Quality Report ===",
        "",
        f"Clusters with explanations: {int(metrics.coverage * len(clusters))}/{len(clusters)} ({metrics.coverage:.1%})",
        f"Verified explanations: {metrics.verified_ratio:.1%}",
        "",
        "Groundedness Scores:",
        f"  Average: {metrics.avg_groundedness:.2f}",
        f"  Min: {metrics.min_groundedness:.2f}",
        f"  Max: {metrics.max_groundedness:.2f}",
        "",
        f"Average summary bullets: {metrics.avg_summary_length:.1f}",
        f"Average unsupported claims: {metrics.avg_unsupported_claims:.1f}",
        "",
    ]

    # Add per-cluster details
    lines.append("Per-Cluster Details:")
    verifier = EvidenceVerifier()

    for cluster in clusters:
        if cluster.explanation is None:
            lines.append(f"  {cluster.cluster_id}: No explanation")
            continue

        result = verifier.verify(cluster, cluster.explanation, failures)
        status = "✓" if result.is_valid else "✗"
        lines.append(
            f"  {status} {cluster.cluster_id}: "
            f"groundedness={result.groundedness_score:.2f}, "
            f"title=\"{cluster.explanation.title[:40]}...\""
        )

    return "\n".join(lines)


