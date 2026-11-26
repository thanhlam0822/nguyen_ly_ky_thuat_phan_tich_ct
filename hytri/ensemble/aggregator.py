"""Aggregation-Based Ensemble Clustering.

Implements the aggregation approach from Karasov et al. 2022:
"Aggregation of Stack Trace Similarities for Crash Report Deduplication"

Key insight: Instead of comparing a failure to the single most similar
member of a group, we aggregate similarities across ALL group members.

Features used for aggregation:
- max_sim: Maximum similarity to any group member
- mean_sim: Mean similarity across group
- min_sim: Minimum (for single-linkage behavior)
- std_sim: Variance (high = uncertain)

The ensemble votes across multiple signals with confidence weighting.
Supports per-pair weight adaptation based on failure type (exception vs
assertion) so that mixed-type failures can share a cluster.
"""

from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from hytri.models import FailureInstance, Cluster
from hytri.ensemble.signals import SimilaritySignals


@dataclass
class GroupAffinity:
    """Affinity of a failure to a candidate group."""
    
    failure_id: str
    group_id: int
    
    # Aggregated features
    max_sim: float = 0.0
    mean_sim: float = 0.0
    min_sim: float = 0.0
    std_sim: float = 0.0
    
    # Per-signal affinities
    signal_affinities: dict[str, float] = field(default_factory=dict)
    
    # Final ensemble score
    ensemble_score: float = 0.0


@dataclass
class ClusteringResult:
    """Result of ensemble clustering."""
    
    clusters: list[Cluster]
    noise_ids: list[str]
    
    # Diagnostics
    num_signals_used: int = 0
    signal_agreements: dict[str, float] = field(default_factory=dict)
    threshold_used: float = 0.0


class AggregationEnsemble:
    """Combine signals using group-level aggregation and confidence-weighted voting.
    
    This implements the key insight from JetBrains research:
    comparing to the entire group improves Recall@1 by ~15%.
    
    The algorithm:
    1. Compute aggregated features for each failure to each candidate group
    2. Vote across signals with confidence weighting
    3. Handle disagreements conservatively (prefer splitting)
    
    Supports per-pair weight adaptation: signal weights are adjusted based
    on the failure-type pair (exc-exc, asr-asr, or mixed) so that
    type-appropriate signals are emphasized without creating hard partitions.
    """
    
    # Default signal weights (can be learned or tuned)
    # These are base weights before confidence adjustment
    DEFAULT_WEIGHTS = {
        'pdm': 0.20,       # Stack trace position-dependent model
        'overlap': 0.20,   # Direct file overlap is strong evidence
        'semantic': 0.12,  # LLM embeddings (reduced: can over-merge)
        'message': 0.12,   # Exception message similarity
        'hypothesis': 0.10,# LLM root cause hypotheses
        'test_name': 0.08, # Test name similarity
        'diff_line': 0.18, # NEW: Direct stack-to-diff correlation (strong when available)
    }
    
    # Per-pair multipliers on base weights, keyed by (type_i, type_j).
    # Encodes BuildSheriff's insight that different failure types benefit
    # from different signals, without a hard partition.
    TYPE_PAIR_MULTIPLIERS = {
        ('exception', 'exception'): {
            'pdm': 1.5, 'overlap': 1.0, 'diff_line': 1.0,
            'semantic': 1.0, 'message': 0.8, 'hypothesis': 1.0, 'test_name': 0.5,
        },
        ('assertion', 'assertion'): {
            'pdm': 0.3, 'overlap': 1.0, 'diff_line': 1.0,
            'semantic': 1.0, 'message': 1.5, 'hypothesis': 1.0, 'test_name': 1.5,
        },
        ('exception', 'assertion'): {
            'pdm': 0.5, 'overlap': 1.0, 'diff_line': 1.0,
            'semantic': 1.5, 'message': 1.0, 'hypothesis': 1.5, 'test_name': 0.8,
        },
        ('assertion', 'exception'): {
            'pdm': 0.5, 'overlap': 1.0, 'diff_line': 1.0,
            'semantic': 1.5, 'message': 1.0, 'hypothesis': 1.5, 'test_name': 0.8,
        },
    }
    
    # Assertion exception types (mirrors FailureStratifier for classification)
    _ASSERTION_INDICATORS = {
        'assertionerror', 'assertionfailederror', 'comparionfailure',
        'comparisonfailure', 'pytest.fail', 'assert',
    }
    
    # When PDM confidence is high, boost its weight significantly
    # This helps on datasets like BugSwarm with high-quality stack traces
    HIGH_CONFIDENCE_PDM_BOOST = 0.15  # Add this to PDM weight when confidence > 0.5
    
    # Linkage methods
    LINKAGE_METHOD = 'average'  # 'single', 'complete', 'average'
    
    def __init__(
        self,
        signal_weights: dict[str, float] | None = None,
        use_confidence_weighting: bool = True,
        disagreement_penalty: float = 0.2,
        min_cluster_size: int = 1,
        use_type_adaptation: bool = True,
        pdm_boost: float | None = None,
    ):
        """Initialize the aggregation ensemble.
        
        Args:
            signal_weights: Optional custom weights for each signal.
            use_confidence_weighting: Whether to weight by signal confidence.
            disagreement_penalty: Penalty when signals disagree strongly.
            min_cluster_size: Minimum failures per cluster.
            use_type_adaptation: Whether to adapt weights per failure-type pair.
            pdm_boost: Override for HIGH_CONFIDENCE_PDM_BOOST. None uses class default.
        """
        self.signal_weights = signal_weights or self.DEFAULT_WEIGHTS.copy()
        self.use_confidence_weighting = use_confidence_weighting
        self.disagreement_penalty = disagreement_penalty
        self.min_cluster_size = min_cluster_size
        self.use_type_adaptation = use_type_adaptation
        self._pdm_boost_override = pdm_boost
    
    def cluster(
        self,
        failures: list[FailureInstance],
        signals: SimilaritySignals,
        distance_threshold: float = 0.5,
    ) -> ClusteringResult:
        """Cluster failures using ensemble of signals.
        
        Args:
            failures: List of failure instances.
            signals: Pre-computed similarity signals.
            distance_threshold: Clustering distance threshold.
            
        Returns:
            ClusteringResult with clusters and diagnostics.
        """
        n = len(failures)
        if n == 0:
            return ClusteringResult(clusters=[], noise_ids=[])
        
        if n == 1:
            cluster = Cluster(
                cluster_id="cluster_0",
                failure_ids=[failures[0].failure_id],
                representative_failure_id=failures[0].failure_id,
            )
            return ClusteringResult(clusters=[cluster], noise_ids=[])
        
        # Step 1: Compute ensemble similarity matrix
        ensemble_sim = self._compute_ensemble_similarity(signals, failures)
        
        # Step 2: Convert to distance and cluster
        distance_matrix = 1.0 - ensemble_sim
        np.fill_diagonal(distance_matrix, 0.0)
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        
        # Ensure symmetry
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        
        # Step 3: Hierarchical clustering
        condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed, method=self.LINKAGE_METHOD)
        
        # Step 4: Cut tree at threshold
        labels = fcluster(linkage_matrix, t=distance_threshold, criterion='distance')
        labels = labels - 1  # Make 0-indexed
        
        # Step 5: Convert to Cluster objects
        clusters = self._labels_to_clusters(failures, labels)
        
        # Step 6: Compute diagnostics
        signal_agreements = self._compute_signal_agreements(signals, labels)
        
        return ClusteringResult(
            clusters=clusters,
            noise_ids=[],  # No noise with hierarchical clustering
            num_signals_used=len(signals.get_all_signals()),
            signal_agreements=signal_agreements,
            threshold_used=distance_threshold,
        )
    
    @classmethod
    def _classify_failure(cls, failure: FailureInstance) -> str:
        """Classify a failure as 'exception' or 'assertion'."""
        exc_type = (failure.exception_type or '').lower()
        exc_msg = (failure.exception_message or '').lower()
        for indicator in cls._ASSERTION_INDICATORS:
            if indicator in exc_type:
                return 'assertion'
        if 'expected' in exc_msg or 'but was' in exc_msg:
            return 'assertion'
        return 'exception'

    def _compute_ensemble_similarity(
        self,
        signals: SimilaritySignals,
        failures: list[FailureInstance] | None = None,
    ) -> np.ndarray:
        """Compute ensemble similarity matrix from all signals.
        
        Uses confidence-weighted voting with disagreement penalty.
        Dynamically boosts PDM weight when stack traces are high quality.
        When failures are provided and type adaptation is enabled, signal
        weights are adjusted per pair based on the failure-type combination.
        """
        all_signals = signals.get_all_signals()
        if not all_signals:
            return np.eye(signals.n_failures)
        
        n = signals.n_failures
        ensemble_sim = np.zeros((n, n))
        
        # Check if PDM has high confidence (good stack traces)
        pdm_conf = signals.get_confidence('pdm')
        max_boost = self._pdm_boost_override if self._pdm_boost_override is not None else self.HIGH_CONFIDENCE_PDM_BOOST
        pdm_boost = max_boost if pdm_conf > 0.5 else 0.0
        
        # Compute base scalar weights per signal (with PDM boost + confidence)
        base_weights = {}
        for name in all_signals.keys():
            w = self.signal_weights.get(name, 0.1)
            if w == 0.0:
                base_weights[name] = 0.0
                continue
            if name == 'pdm' and pdm_boost > 0:
                w += pdm_boost
            elif name in ('semantic', 'hypothesis') and pdm_boost > 0:
                w = max(0.05, w - pdm_boost / 2)
            if self.use_confidence_weighting:
                w *= signals.get_confidence(name)
            base_weights[name] = w
        
        # Per-pair weight adaptation
        if self.use_type_adaptation and failures and len(failures) == n:
            types = [self._classify_failure(f) for f in failures]
            
            # Build per-signal NxN weight matrices
            weight_matrices = {}
            for name in all_signals.keys():
                W = np.full((n, n), base_weights[name])
                for i in range(n):
                    for j in range(i + 1, n):
                        pair = (types[i], types[j])
                        mult = self.TYPE_PAIR_MULTIPLIERS.get(pair, {}).get(name, 1.0)
                        W[i, j] = base_weights[name] * mult
                        W[j, i] = W[i, j]
                weight_matrices[name] = W
            
            # Normalize weights per pair so they sum to 1
            total_W = sum(weight_matrices.values())
            total_W = np.where(total_W > 0, total_W, 1.0)
            
            for name, matrix in all_signals.items():
                ensemble_sim += (weight_matrices[name] / total_W) * matrix
        else:
            # Uniform scalar weights (original behavior)
            total_weight = sum(base_weights.values())
            if total_weight > 0:
                weights = {k: v / total_weight for k, v in base_weights.items()}
            else:
                weights = {k: 1.0 / len(all_signals) for k in all_signals.keys()}
            
            for name, matrix in all_signals.items():
                ensemble_sim += weights[name] * matrix
        
        # Apply disagreement penalty
        ensemble_sim = self._apply_disagreement_penalty(ensemble_sim, all_signals)
        
        # Ensure valid range
        ensemble_sim = np.clip(ensemble_sim, 0.0, 1.0)
        np.fill_diagonal(ensemble_sim, 1.0)
        
        return ensemble_sim
    
    def _apply_disagreement_penalty(
        self,
        ensemble_sim: np.ndarray,
        all_signals: dict[str, np.ndarray],
    ) -> np.ndarray:
        """Apply penalty when signals disagree strongly.
        
        If signals have high variance for a pair, reduce confidence.
        This encourages conservative clustering (prefer splitting).
        """
        if len(all_signals) < 2:
            return ensemble_sim
        
        n = ensemble_sim.shape[0]
        
        # Stack all signal matrices
        signal_stack = np.stack(list(all_signals.values()), axis=0)
        
        # Compute variance across signals for each pair
        variance = np.var(signal_stack, axis=0)
        
        # Apply penalty proportional to variance
        # High variance = signals disagree = reduce similarity
        penalty = self.disagreement_penalty * variance
        
        result = ensemble_sim - penalty
        return np.clip(result, 0.0, 1.0)
    
    def _labels_to_clusters(
        self,
        failures: list[FailureInstance],
        labels: np.ndarray,
    ) -> list[Cluster]:
        """Convert cluster labels to Cluster objects."""
        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for failure, label in zip(failures, labels):
            groups[label].append(failure)
        
        clusters = []
        for label, group_failures in sorted(groups.items()):
            failure_ids = [f.failure_id for f in group_failures]
            
            # Choose representative (first failure in group)
            representative_id = failure_ids[0]
            
            # Collect exception types
            exception_types = list(set(
                f.exception_type for f in group_failures 
                if f.exception_type
            ))
            
            cluster = Cluster(
                cluster_id=f"cluster_{label}",
                failure_ids=failure_ids,
                representative_failure_id=representative_id,
                exception_types=exception_types,
            )
            
            clusters.append(cluster)
        
        return clusters
    
    def _compute_signal_agreements(
        self,
        signals: SimilaritySignals,
        labels: np.ndarray,
    ) -> dict[str, float]:
        """Compute how well each signal agrees with final clustering.
        
        Higher agreement = signal aligned with consensus.
        """
        all_signals = signals.get_all_signals()
        if not all_signals:
            return {}
        
        n = len(labels)
        agreements = {}
        
        # Convert labels to binary same-cluster matrix
        same_cluster = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                same_cluster[i, j] = 1.0 if labels[i] == labels[j] else 0.0
        
        for name, matrix in all_signals.items():
            # Agreement = correlation between signal and clustering
            # High similarity for same cluster, low for different
            signal_flat = matrix.flatten()
            cluster_flat = same_cluster.flatten()
            
            if np.std(signal_flat) > 0 and np.std(cluster_flat) > 0:
                correlation = np.corrcoef(signal_flat, cluster_flat)[0, 1]
                agreements[name] = max(0.0, correlation)  # Only positive correlations
            else:
                agreements[name] = 0.5  # Neutral if no variance
        
        return agreements
    
    def compute_group_affinity(
        self,
        failure_idx: int,
        group_indices: list[int],
        signals: SimilaritySignals,
    ) -> GroupAffinity:
        """Compute affinity of a failure to a candidate group.
        
        Uses aggregation features across ALL group members.
        
        Args:
            failure_idx: Index of the failure.
            group_indices: Indices of failures in the candidate group.
            signals: Similarity signals.
            
        Returns:
            GroupAffinity with aggregated features.
        """
        all_signals = signals.get_all_signals()
        
        # Collect similarities to all group members
        similarities = defaultdict(list)
        
        for name, matrix in all_signals.items():
            for group_idx in group_indices:
                if group_idx != failure_idx:
                    similarities[name].append(matrix[failure_idx, group_idx])
        
        # Compute aggregated features per signal
        signal_affinities = {}
        all_sims = []
        
        for name, sims in similarities.items():
            if sims:
                signal_affinities[name] = np.mean(sims)
                all_sims.extend(sims)
        
        # Compute overall aggregated features
        if all_sims:
            max_sim = max(all_sims)
            mean_sim = np.mean(all_sims)
            min_sim = min(all_sims)
            std_sim = np.std(all_sims)
        else:
            max_sim = mean_sim = min_sim = std_sim = 0.0
        
        # Compute ensemble score with confidence weighting
        ensemble_score = 0.0
        total_weight = 0.0
        
        for name, affinity in signal_affinities.items():
            weight = self.signal_weights.get(name, 0.1)
            if self.use_confidence_weighting:
                weight *= signals.get_confidence(name)
            ensemble_score += weight * affinity
            total_weight += weight
        
        if total_weight > 0:
            ensemble_score /= total_weight
        
        return GroupAffinity(
            failure_id=str(failure_idx),
            group_id=0,
            max_sim=max_sim,
            mean_sim=mean_sim,
            min_sim=min_sim,
            std_sim=std_sim,
            signal_affinities=signal_affinities,
            ensemble_score=ensemble_score,
        )


class IncrementalAggregator(AggregationEnsemble):
    """Incremental version for online clustering.
    
    Can add failures one at a time to existing clusters.
    Useful for real-time triage during CI runs.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.existing_clusters: list[list[int]] = []
        self._cached_signals: SimilaritySignals | None = None
    
    def add_failure(
        self,
        failure_idx: int,
        signals: SimilaritySignals,
        similarity_threshold: float = 0.5,
    ) -> int:
        """Add a failure to the best matching cluster or create new.
        
        Args:
            failure_idx: Index of the new failure.
            signals: Similarity signals (must include the new failure).
            similarity_threshold: Minimum affinity to join existing cluster.
            
        Returns:
            Index of the cluster the failure was added to.
        """
        if not self.existing_clusters:
            # First failure starts a new cluster
            self.existing_clusters.append([failure_idx])
            return 0
        
        # Compute affinity to each existing cluster
        best_cluster = -1
        best_affinity = 0.0
        
        for cluster_idx, cluster_members in enumerate(self.existing_clusters):
            affinity = self.compute_group_affinity(
                failure_idx, cluster_members, signals
            )
            
            if affinity.ensemble_score > best_affinity:
                best_affinity = affinity.ensemble_score
                best_cluster = cluster_idx
        
        # Join best cluster if above threshold, otherwise create new
        if best_affinity >= similarity_threshold:
            self.existing_clusters[best_cluster].append(failure_idx)
            return best_cluster
        else:
            new_cluster_idx = len(self.existing_clusters)
            self.existing_clusters.append([failure_idx])
            return new_cluster_idx
    
    def reset(self) -> None:
        """Reset for a new clustering session."""
        self.existing_clusters = []
        self._cached_signals = None
