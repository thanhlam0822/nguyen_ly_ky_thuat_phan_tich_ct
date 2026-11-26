"""Component E: Prioritization / Triage Queue.

Implements severity scoring and priority ordering for failure clusters.
"""

import math
from dataclasses import dataclass
from datetime import datetime

from hytri.models import Cluster, FailureInstance


@dataclass
class PriorityWeights:
    """Weights for priority score components."""

    frequency: float = 0.3
    breadth: float = 0.25
    novelty: float = 0.25
    recency: float = 0.2


class PriorityScorer:
    """Computes priority scores for failure clusters.

    Score components:
    - Frequency: Number of failures in the cluster (log-scaled)
    - Breadth: Number of unique tests/modules affected
    - Novelty: Whether exception type is new/rare
    - Recency: How recently failures occurred (decay function)
    """

    def __init__(
        self,
        weights: PriorityWeights | None = None,
        known_exception_types: set[str] | None = None,
        recency_half_life_hours: float = 24.0,
    ):
        """Initialize the scorer.

        Args:
            weights: Weights for each score component.
            known_exception_types: Set of known exception types (for novelty).
            recency_half_life_hours: Half-life for recency decay.
        """
        self.weights = weights or PriorityWeights()
        self.known_exception_types = known_exception_types or set()
        self.recency_half_life_hours = recency_half_life_hours

    def score_cluster(
        self,
        cluster: Cluster,
        failures: list[FailureInstance],
        reference_time: datetime | None = None,
    ) -> float:
        """Compute priority score for a cluster.

        Args:
            cluster: The cluster to score.
            failures: All failure instances (for looking up details).
            reference_time: Reference time for recency calculation.

        Returns:
            Priority score (higher = more urgent).
        """
        # Get cluster failures
        failure_map = {f.failure_id: f for f in failures}
        cluster_failures = [
            failure_map[fid]
            for fid in cluster.failure_ids
            if fid in failure_map
        ]

        if not cluster_failures:
            return 0.0

        # Compute score components
        freq_score = self._frequency_score(len(cluster.failure_ids))
        breadth_score = self._breadth_score(cluster)
        novelty_score = self._novelty_score(cluster)
        recency_score = self._recency_score(cluster_failures, reference_time)

        # Weighted combination
        total_score = (
            self.weights.frequency * freq_score
            + self.weights.breadth * breadth_score
            + self.weights.novelty * novelty_score
            + self.weights.recency * recency_score
        )

        return total_score

    def _frequency_score(self, num_failures: int) -> float:
        """Compute frequency score (log-scaled).

        Log scaling prevents very large clusters from dominating.
        """
        if num_failures <= 0:
            return 0.0
        return math.log(num_failures + 1) / math.log(100)  # Normalize to ~1 at 100 failures

    def _breadth_score(self, cluster: Cluster) -> float:
        """Compute breadth score based on blast radius.

        Higher score when more unique tests/modules are affected.
        """
        test_score = min(cluster.num_unique_tests / 10, 1.0)
        module_score = min(cluster.num_unique_modules / 5, 1.0)
        return (test_score + module_score) / 2

    def _novelty_score(self, cluster: Cluster) -> float:
        """Compute novelty score based on exception type rarity.

        New/rare exception types get higher scores.
        """
        if not cluster.exception_types:
            return 0.5

        novel_count = sum(
            1 for exc in cluster.exception_types
            if exc not in self.known_exception_types
        )

        return novel_count / len(cluster.exception_types)

    def _recency_score(
        self,
        failures: list[FailureInstance],
        reference_time: datetime | None = None,
    ) -> float:
        """Compute recency score with exponential decay.

        More recent failures get higher scores.
        """
        if not failures:
            return 0.5

        reference = reference_time or datetime.now()

        # Get most recent failure timestamp
        timestamps = [f.timestamp for f in failures if f.timestamp]
        if not timestamps:
            return 0.5

        most_recent = max(timestamps)
        hours_ago = (reference - most_recent).total_seconds() / 3600

        # Exponential decay
        decay = 0.5 ** (hours_ago / self.recency_half_life_hours)
        return decay

    def score_clusters(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
        reference_time: datetime | None = None,
    ) -> list[Cluster]:
        """Score multiple clusters and update their severity_score.

        Args:
            clusters: List of clusters to score.
            failures: All failure instances.
            reference_time: Reference time for recency.

        Returns:
            Clusters with updated severity scores.
        """
        for cluster in clusters:
            cluster.severity_score = self.score_cluster(
                cluster, failures, reference_time
            )
        return clusters

    def rank_clusters(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
        reference_time: datetime | None = None,
    ) -> list[Cluster]:
        """Score and rank clusters by priority (highest first).

        Args:
            clusters: List of clusters to rank.
            failures: All failure instances.
            reference_time: Reference time for recency.

        Returns:
            Clusters sorted by priority (highest first).
        """
        self.score_clusters(clusters, failures, reference_time)
        return sorted(clusters, key=lambda c: c.severity_score, reverse=True)

    def get_priority_queue(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
        reference_time: datetime | None = None,
    ) -> list[str]:
        """Get cluster IDs in priority order.

        Args:
            clusters: List of clusters.
            failures: All failure instances.
            reference_time: Reference time for recency.

        Returns:
            List of cluster_ids in priority order (highest first).
        """
        ranked = self.rank_clusters(clusters, failures, reference_time)
        return [c.cluster_id for c in ranked]

    def update_known_exceptions(self, exception_types: set[str]) -> None:
        """Update the set of known exception types.

        Call this after processing builds to track what's "known".
        """
        self.known_exception_types.update(exception_types)


class ImpactAnalyzer:
    """Analyzes potential impact of failure clusters.

    Provides additional signals for prioritization:
    - Code ownership (which team owns affected modules)
    - Flakiness detection (if failures appear/disappear)
    - Dependency analysis (core vs. edge modules)
    """

    def __init__(self):
        """Initialize the analyzer."""
        self._flakiness_history: dict[str, list[bool]] = {}

    def analyze_flakiness(
        self,
        failure_id: str,
        is_failing: bool,
    ) -> float:
        """Track and analyze flakiness for a test.

        Args:
            failure_id: ID of the failure/test.
            is_failing: Whether the test is currently failing.

        Returns:
            Flakiness score (0 = stable, 1 = very flaky).
        """
        if failure_id not in self._flakiness_history:
            self._flakiness_history[failure_id] = []

        history = self._flakiness_history[failure_id]
        history.append(is_failing)

        # Keep last 10 runs
        if len(history) > 10:
            history.pop(0)

        if len(history) < 2:
            return 0.0

        # Count state changes
        changes = sum(
            1 for i in range(1, len(history))
            if history[i] != history[i - 1]
        )

        return changes / (len(history) - 1)

    def is_likely_infra(self, cluster: Cluster) -> bool:
        """Heuristic to detect infrastructure-related failures.

        Args:
            cluster: The cluster to analyze.

        Returns:
            True if likely infrastructure issue.
        """
        infra_indicators = {
            "timeout",
            "connection",
            "socket",
            "network",
            "dns",
            "ssl",
            "certificate",
            "memory",
            "disk",
            "permission",
            "resource",
        }

        # Check exception types
        for exc in cluster.exception_types:
            exc_lower = exc.lower()
            if any(ind in exc_lower for ind in infra_indicators):
                return True

        # Check top frames for common infra modules
        infra_modules = {"socket", "ssl", "urllib", "http", "subprocess"}
        for frame in cluster.top_frames:
            frame_lower = frame.lower()
            if any(mod in frame_lower for mod in infra_modules):
                return True

        return False


