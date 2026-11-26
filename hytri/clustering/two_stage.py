"""Two-Stage Hierarchical Clustering.

Stage 1: Over-cluster using strict lexical similarity (like baselines)
Stage 2: Semantically merge related clusters using LLM understanding

This approach:
- Matches baseline precision in Stage 1
- Adds unique value through intelligent consolidation in Stage 2
- Produces explainable cluster merges
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import uuid
from typing import Optional

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering

from hytri.models import Cluster, FailureInstance
from hytri.llm.openai_client import OpenAIClient


@dataclass
class MergeDecision:
    """Record of a cluster merge decision."""
    cluster_a_id: str
    cluster_b_id: str
    merged: bool
    reason: str
    confidence: float


class TwoStageClusterer:
    """Two-stage clustering: lexical over-clustering + semantic merging.
    
    This approach achieves competitive metrics with baselines while
    producing more meaningful, explainable clusters.
    """

    def __init__(
        self,
        # Stage 1 parameters
        stage1_distance_threshold: float = 0.3,  # Strict = many clusters
        # Stage 2 parameters  
        semantic_merge_threshold: float = 0.7,  # Similarity needed to merge
        use_llm_validation: bool = True,
        openai_client: Optional[OpenAIClient] = None,
        # General
        min_cluster_size: int = 1,
    ):
        """Initialize two-stage clusterer.
        
        Args:
            stage1_distance_threshold: Distance threshold for initial clustering.
                                       Lower = stricter = more clusters.
            semantic_merge_threshold: Similarity threshold for merging clusters.
            use_llm_validation: Whether to use LLM to validate merge decisions.
            openai_client: OpenAI client for semantic operations.
            min_cluster_size: Minimum size for final clusters.
        """
        self.stage1_threshold = stage1_distance_threshold
        self.merge_threshold = semantic_merge_threshold
        self.use_llm_validation = use_llm_validation
        self.openai_client = openai_client
        self.min_cluster_size = min_cluster_size
        
        self.merge_history: list[MergeDecision] = []

    def cluster(
        self,
        failures: list[FailureInstance],
        lexical_similarity: np.ndarray,
        semantic_similarity: Optional[np.ndarray] = None,
    ) -> tuple[list[Cluster], list[str], list[MergeDecision]]:
        """Perform two-stage clustering.
        
        Args:
            failures: List of failure instances.
            lexical_similarity: NxN lexical similarity matrix.
            semantic_similarity: Optional NxN semantic similarity matrix.
            
        Returns:
            Tuple of (final_clusters, noise_ids, merge_history).
        """
        if len(failures) == 0:
            return [], [], []
        
        if len(failures) == 1:
            cluster = self._create_cluster(failures)
            return [cluster], [], []
        
        self.merge_history = []
        
        # ========== STAGE 1: Over-cluster with strict lexical ==========
        stage1_clusters = self._stage1_lexical_clustering(
            failures, lexical_similarity
        )
        
        print(f"  Stage 1: {len(stage1_clusters)} initial clusters")
        
        if semantic_similarity is None or len(stage1_clusters) <= 1:
            # No semantic merging possible
            return stage1_clusters, [], self.merge_history
        
        # ========== STAGE 2: Semantic merging ==========
        final_clusters = self._stage2_semantic_merging(
            stage1_clusters, failures, semantic_similarity
        )
        
        print(f"  Stage 2: {len(final_clusters)} final clusters (merged {len(stage1_clusters) - len(final_clusters)})")
        
        return final_clusters, [], self.merge_history

    def _stage1_lexical_clustering(
        self,
        failures: list[FailureInstance],
        similarity_matrix: np.ndarray,
    ) -> list[Cluster]:
        """Stage 1: Over-cluster using strict lexical similarity."""
        n = len(failures)
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        
        # Agglomerative clustering with strict threshold
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.stage1_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = clusterer.fit_predict(distance_matrix)
        except ValueError:
            # Fallback: each failure in its own cluster
            labels = np.arange(n)
        
        # Convert labels to Cluster objects
        return self._labels_to_clusters(labels, failures)

    def _stage2_semantic_merging(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
        semantic_similarity: np.ndarray,
    ) -> list[Cluster]:
        """Stage 2: Merge semantically similar clusters."""
        if len(clusters) <= 1:
            return clusters
        
        # Build failure lookup
        failure_map = {f.failure_id: (i, f) for i, f in enumerate(failures)}
        
        # Compute inter-cluster semantic similarity
        cluster_sim = self._compute_cluster_similarity(
            clusters, failure_map, semantic_similarity
        )
        
        # Greedy merging based on similarity
        merged_clusters = self._greedy_merge(
            clusters, cluster_sim, failures, failure_map
        )
        
        return merged_clusters

    def _compute_cluster_similarity(
        self,
        clusters: list[Cluster],
        failure_map: dict[str, tuple[int, FailureInstance]],
        semantic_similarity: np.ndarray,
    ) -> np.ndarray:
        """Compute pairwise semantic similarity between clusters."""
        n_clusters = len(clusters)
        cluster_sim = np.zeros((n_clusters, n_clusters))
        
        for i in range(n_clusters):
            for j in range(i + 1, n_clusters):
                # Average semantic similarity between all pairs of failures
                sim_sum = 0.0
                count = 0
                
                for fid_i in clusters[i].failure_ids:
                    for fid_j in clusters[j].failure_ids:
                        if fid_i in failure_map and fid_j in failure_map:
                            idx_i = failure_map[fid_i][0]
                            idx_j = failure_map[fid_j][0]
                            sim_sum += semantic_similarity[idx_i, idx_j]
                            count += 1
                
                avg_sim = sim_sum / count if count > 0 else 0.0
                cluster_sim[i, j] = avg_sim
                cluster_sim[j, i] = avg_sim
        
        np.fill_diagonal(cluster_sim, 1.0)
        return cluster_sim

    def _greedy_merge(
        self,
        clusters: list[Cluster],
        cluster_sim: np.ndarray,
        failures: list[FailureInstance],
        failure_map: dict[str, tuple[int, FailureInstance]],
    ) -> list[Cluster]:
        """Greedily merge clusters based on semantic similarity."""
        n = len(clusters)
        merged = [False] * n
        merge_targets = list(range(n))  # merge_targets[i] = cluster i merged into
        
        # Find merge candidates
        for i in range(n):
            if merged[i]:
                continue
                
            for j in range(i + 1, n):
                if merged[j]:
                    continue
                
                sim = cluster_sim[i, j]
                
                if sim >= self.merge_threshold:
                    # Check if same exception type (hard constraint)
                    types_i = set(clusters[i].exception_types)
                    types_j = set(clusters[j].exception_types)
                    
                    should_merge = True
                    reason = f"Semantic similarity {sim:.2f} >= {self.merge_threshold}"
                    
                    # Validate with LLM if enabled
                    if self.use_llm_validation and self.openai_client:
                        should_merge, reason = self._llm_validate_merge(
                            clusters[i], clusters[j], failures, failure_map
                        )
                    
                    if should_merge:
                        # Merge j into i
                        merged[j] = True
                        merge_targets[j] = merge_targets[i]
                        
                        self.merge_history.append(MergeDecision(
                            cluster_a_id=clusters[i].cluster_id,
                            cluster_b_id=clusters[j].cluster_id,
                            merged=True,
                            reason=reason,
                            confidence=sim,
                        ))
                    else:
                        self.merge_history.append(MergeDecision(
                            cluster_a_id=clusters[i].cluster_id,
                            cluster_b_id=clusters[j].cluster_id,
                            merged=False,
                            reason=reason,
                            confidence=sim,
                        ))
        
        # Build final clusters
        final_cluster_failures: dict[int, list[FailureInstance]] = {}
        
        for i, cluster in enumerate(clusters):
            target = merge_targets[i]
            # Follow merge chain
            while merge_targets[target] != target:
                target = merge_targets[target]
            
            if target not in final_cluster_failures:
                final_cluster_failures[target] = []
            
            # Add failures from this cluster
            for fid in cluster.failure_ids:
                if fid in failure_map:
                    final_cluster_failures[target].append(failure_map[fid][1])
        
        # Create final Cluster objects
        final_clusters = []
        for failures_list in final_cluster_failures.values():
            if failures_list:
                final_clusters.append(self._create_cluster(failures_list))
        
        return final_clusters

    def _llm_validate_merge(
        self,
        cluster_a: Cluster,
        cluster_b: Cluster,
        failures: list[FailureInstance],
        failure_map: dict[str, tuple[int, FailureInstance]],
    ) -> tuple[bool, str]:
        """Use LLM to validate if two clusters should be merged."""
        if not self.openai_client or not self.openai_client.is_available():
            return True, "LLM unavailable, defaulting to merge"
        
        # Get representative failures from each cluster
        rep_a = self._get_cluster_summary(cluster_a, failure_map)
        rep_b = self._get_cluster_summary(cluster_b, failure_map)
        
        prompt = f"""Analyze if these two groups of test failures should be merged (same root cause).

GROUP A:
- Exception types: {', '.join(cluster_a.exception_types)}
- Top frames: {', '.join(cluster_a.top_frames[:3])}
- Sample: {rep_a[:500]}

GROUP B:
- Exception types: {', '.join(cluster_b.exception_types)}
- Top frames: {', '.join(cluster_b.top_frames[:3])}
- Sample: {rep_b[:500]}

Answer in this format:
MERGE: YES or NO
REASON: Brief explanation (one sentence)"""

        try:
            response = self.openai_client.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            
            answer = response.choices[0].message.content.strip()
            
            should_merge = "MERGE: YES" in answer.upper()
            reason = answer.split("REASON:")[-1].strip() if "REASON:" in answer else answer
            
            return should_merge, reason
            
        except Exception as e:
            return True, f"LLM error: {e}, defaulting to merge"

    def _get_cluster_summary(
        self,
        cluster: Cluster,
        failure_map: dict[str, tuple[int, FailureInstance]],
    ) -> str:
        """Get a text summary of a cluster for LLM validation."""
        summaries = []
        for fid in cluster.failure_ids[:2]:  # Just first 2
            if fid in failure_map:
                f = failure_map[fid][1]
                summaries.append(f"{f.exception_type}: {f.exception_message[:100]}")
        return " | ".join(summaries)

    def _labels_to_clusters(
        self,
        labels: np.ndarray,
        failures: list[FailureInstance],
    ) -> list[Cluster]:
        """Convert numeric labels to Cluster objects."""
        from collections import defaultdict
        
        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)
        
        clusters = []
        for group in groups.values():
            if len(group) >= self.min_cluster_size:
                clusters.append(self._create_cluster(group))
        
        return clusters

    def _create_cluster(self, failures: list[FailureInstance]) -> Cluster:
        """Create a Cluster object from failures."""
        failure_ids = [f.failure_id for f in failures]
        exception_types = list(set(f.exception_type for f in failures if f.exception_type))
        
        frame_counts: Counter = Counter()
        for f in failures:
            for frame in f.frames:
                frame_counts[f"{frame.file}:{frame.func}"] += 1
        top_frames = [frame for frame, _ in frame_counts.most_common(5)]
        
        unique_tests = set(f.test_name for f in failures if f.test_name)
        unique_modules = set()
        for f in failures:
            for frame in f.frames:
                unique_modules.add(frame.file)

        return Cluster(
            cluster_id=str(uuid.uuid4())[:8],
            failure_ids=failure_ids,
            representative_ids=failure_ids[:5],
            exception_types=exception_types,
            top_frames=top_frames,
            num_unique_tests=len(unique_tests),
            num_unique_modules=len(unique_modules),
        )


