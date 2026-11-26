"""Component D1: Cluster Summarizer.

Uses LLM to generate human-readable explanations for failure clusters.
"""

from hytri.llm.openai_client import OpenAIClient
from hytri.models import Cluster, ClusterExplanation, FailureInstance


class ClusterSummarizer:
    """Generates explanations for failure clusters using LLM.

    Given a cluster of related failures, produces:
    - A short, actionable title
    - 3-5 bullet points explaining the failures with evidence
    """

    def __init__(
        self,
        openai_client: OpenAIClient | None = None,
        max_representatives: int = 5,
    ):
        """Initialize the summarizer.

        Args:
            openai_client: OpenAI client for LLM operations.
            max_representatives: Maximum number of representative failures to include.
        """
        self.client = openai_client or OpenAIClient()
        self.max_representatives = max_representatives

    def summarize_cluster(
        self,
        cluster: Cluster,
        failures: list[FailureInstance],
    ) -> ClusterExplanation:
        """Generate explanation for a cluster.

        Args:
            cluster: The cluster to explain.
            failures: All failure instances (to look up representatives).

        Returns:
            ClusterExplanation with title and summary.
        """
        # Get representative failures
        failure_map = {f.failure_id: f for f in failures}
        representatives = [
            failure_map[fid]
            for fid in cluster.representative_ids
            if fid in failure_map
        ]

        if not representatives:
            # Fallback to first few failures in cluster
            representatives = [
                failure_map[fid]
                for fid in cluster.failure_ids[:self.max_representatives]
                if fid in failure_map
            ]

        # Build summary text for LLM
        summary_text = self._build_failures_summary(representatives, cluster)

        # Generate explanation
        explanation = self.client.generate_cluster_explanation(summary_text)

        return explanation

    def _build_failures_summary(
        self,
        failures: list[FailureInstance],
        cluster: Cluster,
    ) -> str:
        """Build a text summary of failures for the LLM.

        Args:
            failures: Representative failure instances.
            cluster: The cluster being summarized.

        Returns:
            Formatted text summary.
        """
        lines = [
            f"Cluster Statistics:",
            f"- Total failures: {len(cluster.failure_ids)}",
            f"- Unique tests affected: {cluster.num_unique_tests}",
            f"- Unique modules: {cluster.num_unique_modules}",
            f"- Exception types: {', '.join(cluster.exception_types)}",
            f"- Top frames: {', '.join(cluster.top_frames[:3])}",
            "",
            "Representative Failures:",
        ]

        for i, failure in enumerate(failures[:self.max_representatives], 1):
            lines.extend([
                f"",
                f"--- Failure {i} ---",
                f"Test: {failure.test_name}",
                f"Exception: {failure.exception_type}: {failure.exception_message[:200]}",
                f"",
                "Stack trace (top frames):",
            ])

            # Include top 5 frames from trace
            for frame in failure.frames[-5:]:
                lines.append(f"  {frame.file}:{frame.line} in {frame.func}")

            # Include snippet of stderr if available
            if failure.stderr:
                stderr_snippet = failure.stderr[:300].strip()
                lines.extend(["", "Stderr:", stderr_snippet])

        return "\n".join(lines)

    def summarize_clusters(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
    ) -> list[Cluster]:
        """Generate explanations for multiple clusters.

        Args:
            clusters: List of clusters to explain.
            failures: All failure instances.

        Returns:
            Clusters with explanations attached.
        """
        for cluster in clusters:
            explanation = self.summarize_cluster(cluster, failures)
            cluster.explanation = explanation

        return clusters


class FallbackSummarizer:
    """Fallback summarizer when LLM is unavailable.

    Generates basic explanations from cluster statistics.
    """

    def summarize_cluster(
        self,
        cluster: Cluster,
        failures: list[FailureInstance],
    ) -> ClusterExplanation:
        """Generate basic explanation without LLM.

        Args:
            cluster: The cluster to explain.
            failures: All failure instances.

        Returns:
            Basic ClusterExplanation.
        """
        # Get most common exception type
        primary_exception = cluster.exception_types[0] if cluster.exception_types else "Unknown"

        # Get top frame info
        top_frame = cluster.top_frames[0] if cluster.top_frames else "unknown location"

        title = f"{primary_exception} in {top_frame.split(':')[-1]}"[:60]

        summary = [
            f"Exception type: {primary_exception}",
            f"Affected {cluster.num_unique_tests} unique tests",
            f"Top location: {top_frame}",
            f"Total failures: {len(cluster.failure_ids)}",
        ]

        return ClusterExplanation(
            title=title,
            summary=summary,
        )


