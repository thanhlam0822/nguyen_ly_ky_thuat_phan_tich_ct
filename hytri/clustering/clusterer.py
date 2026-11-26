"""Component C2: Failure Clustering.

Implements HDBSCAN clustering on the fused similarity graph,
with fallback to Agglomerative clustering.
"""

from collections import Counter
from typing import Literal
import uuid

import hdbscan
import numpy as np
from sklearn.cluster import AgglomerativeClustering

from hytri.models import Cluster, FailureInstance


class FailureClusterer:
    """Clusters failures using HDBSCAN or Agglomerative clustering.

    HDBSCAN is the primary method as it:
    - Doesn't require specifying number of clusters
    - Handles noise points (unclustered failures)
    - Works well with varying cluster densities

    Falls back to Agglomerative when HDBSCAN produces too many noise points.
    """

    def __init__(
        self,
        method: Literal["hdbscan", "agglomerative", "auto"] = "auto",
        min_cluster_size: int = 2,
        min_samples: int = 1,
        cluster_selection_epsilon: float = 0.0,
        distance_threshold: float = 0.5,
        max_noise_ratio: float = 0.3,
    ):
        """Initialize the clusterer.

        Args:
            method: Clustering method to use.
            min_cluster_size: Minimum cluster size for HDBSCAN.
            min_samples: Minimum samples for HDBSCAN core points.
            cluster_selection_epsilon: HDBSCAN epsilon parameter.
            distance_threshold: Distance threshold for Agglomerative.
            max_noise_ratio: Max noise ratio before falling back to Agglomerative.
        """
        self.method = method
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_epsilon = cluster_selection_epsilon
        self.distance_threshold = distance_threshold
        self.max_noise_ratio = max_noise_ratio

    def cluster(
        self,
        similarity_matrix: np.ndarray,
        failures: list[FailureInstance],
    ) -> tuple[list[Cluster], list[str]]:
        """Cluster failures based on similarity matrix.

        Args:
            similarity_matrix: Pairwise similarity matrix (NxN).
            failures: List of failure instances corresponding to matrix rows.

        Returns:
            Tuple of (clusters, noise_failure_ids).
        """
        n = len(failures)
        if n == 0:
            return [], []

        if n == 1:
            # Single failure is its own cluster
            cluster = self._create_cluster([failures[0]])
            return [cluster], []

        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix

        # Ensure distance matrix is valid
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        np.fill_diagonal(distance_matrix, 0.0)

        # Choose clustering method
        if self.method == "auto":
            labels = self._auto_cluster(distance_matrix)
        elif self.method == "hdbscan":
            labels = self._hdbscan_cluster(distance_matrix)
        else:
            labels = self._agglomerative_cluster(distance_matrix)

        # Build clusters from labels
        return self._labels_to_clusters(labels, failures)

    def _hdbscan_cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using HDBSCAN."""
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="precomputed",
        )

        labels = clusterer.fit_predict(distance_matrix)
        return labels

    def _agglomerative_cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Cluster using Agglomerative clustering."""
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="precomputed",
            linkage="average",
        )

        labels = clusterer.fit_predict(distance_matrix)
        return labels

    def _auto_cluster(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Automatically choose best clustering method.

        Uses HDBSCAN first, falls back to Agglomerative if too much noise.
        """
        n = distance_matrix.shape[0]

        # Try HDBSCAN first
        hdbscan_labels = self._hdbscan_cluster(distance_matrix)

        # Check noise ratio
        noise_count = np.sum(hdbscan_labels == -1)
        noise_ratio = noise_count / n if n > 0 else 0.0

        if noise_ratio <= self.max_noise_ratio:
            return hdbscan_labels

        # Too much noise, try Agglomerative
        return self._agglomerative_cluster(distance_matrix)

    def _labels_to_clusters(
        self,
        labels: np.ndarray,
        failures: list[FailureInstance],
    ) -> tuple[list[Cluster], list[str]]:
        """Convert cluster labels to Cluster objects.

        Args:
            labels: Cluster label for each failure (-1 = noise).
            failures: List of failure instances.

        Returns:
            Tuple of (clusters, noise_failure_ids).
        """
        # Group failures by label
        label_to_failures: dict[int, list[FailureInstance]] = {}
        noise_ids: list[str] = []

        for failure, label in zip(failures, labels):
            if label == -1:
                noise_ids.append(failure.failure_id)
            else:
                if label not in label_to_failures:
                    label_to_failures[label] = []
                label_to_failures[label].append(failure)

        # Create Cluster objects
        clusters = []
        for label in sorted(label_to_failures.keys()):
            cluster_failures = label_to_failures[label]
            cluster = self._create_cluster(cluster_failures)
            clusters.append(cluster)

        return clusters, noise_ids

    def _create_cluster(self, failures: list[FailureInstance]) -> Cluster:
        """Create a Cluster object from a list of failures."""
        failure_ids = [f.failure_id for f in failures]

        # Compute cluster statistics
        exception_types = list(set(f.exception_type for f in failures))

        # Get top frames (most common file:func pairs)
        frame_counts: Counter = Counter()
        for f in failures:
            for frame in f.frames:
                frame_counts[f"{frame.file}:{frame.func}"] += 1
        top_frames = [frame for frame, _ in frame_counts.most_common(5)]

        # Count unique tests and modules
        unique_tests = set(f.test_name for f in failures)
        unique_modules = set()
        for f in failures:
            for frame in f.frames:
                # Extract module from file path
                parts = frame.file.replace("\\", "/").split("/")
                if parts:
                    unique_modules.add(parts[-1].replace(".py", ""))

        # Select representative failures (up to 5)
        representative_ids = failure_ids[:5]

        return Cluster(
            cluster_id=str(uuid.uuid4())[:8],
            failure_ids=failure_ids,
            representative_ids=representative_ids,
            exception_types=exception_types,
            top_frames=top_frames,
            num_unique_tests=len(unique_tests),
            num_unique_modules=len(unique_modules),
        )

    def get_cluster_assignments(
        self,
        similarity_matrix: np.ndarray,
        failures: list[FailureInstance],
    ) -> dict[str, str]:
        """Get mapping of failure_id to cluster_id.

        Args:
            similarity_matrix: Pairwise similarity matrix.
            failures: List of failure instances.

        Returns:
            Dict mapping failure_id -> cluster_id (or "noise" for unclustered).
        """
        clusters, noise_ids = self.cluster(similarity_matrix, failures)

        assignments = {}
        for cluster in clusters:
            for failure_id in cluster.failure_ids:
                assignments[failure_id] = cluster.cluster_id

        for failure_id in noise_ids:
            assignments[failure_id] = "noise"

        return assignments


