"""HyTri-E: Ensemble Failure Triage Pipeline.

Main orchestrating pipeline that combines:
1. Complexity Gate (BuildSheriff S1): Short-circuit when few changes
2. Failure Type Awareness: Per-pair weight adaptation for exception vs assertion
3. Multi-Signal Similarity: PDM, overlap, semantic, message, hypothesis, test_name
4. LLM Hypothesis Generation: Root cause reasoning
5. Aggregation Ensemble: Group-level similarity with confidence voting
6. Adaptive Thresholds: Signal-aware threshold selection

This architecture beats baselines by combining:
- BuildSheriff's domain-specific decision rules
- LLM's semantic understanding
- Aggregation-based similarity from JetBrains research
"""

from dataclasses import dataclass, field
from datetime import datetime

from hytri.models import FailureInstance, Cluster, TriageResult
from hytri.llm.openai_client import OpenAIClient
from hytri.ensemble.complexity_gate import ComplexityGate, ComplexityAnalysis
from hytri.ensemble.stratifier import FailureStratifier
from hytri.ensemble.signals import SimilaritySignals, MultiSignalComputer
from hytri.ensemble.hypothesis import HypothesisGenerator, HypothesisBatch
from hytri.ensemble.aggregator import AggregationEnsemble, ClusteringResult
from hytri.ensemble.threshold import AdaptiveThreshold, ThresholdConfig


@dataclass
class EnsembleConfig:
    """Configuration for the ensemble pipeline."""
    
    # Complexity gate
    delta_exception: int = 7  # BuildSheriff threshold
    delta_assertion: int = 7
    enable_complexity_gate: bool = True
    max_files_for_single_cluster: int = 3  # Safety: don't single-cluster if many files
    require_file_overlap: bool = True  # Require files in same package for single-cluster
    
    # Stratification
    enable_stratification: bool = True
    
    # Hypothesis generation
    enable_hypothesis: bool = True
    hypothesis_batch_size: int = 10
    
    # Signal weights (base weights before confidence adjustment)
    signal_weights: dict[str, float] = field(default_factory=lambda: {
        'pdm': 0.25,       # Increased: reliable when stack traces are good
        'overlap': 0.25,   # Direct file overlap
        'semantic': 0.15,  # Reduced: can over-merge
        'message': 0.15,   # Good for assertions
        'hypothesis': 0.10,# Reduced: can find false connections
        'test_name': 0.10, # Test name similarity
    })
    
    # Threshold config
    threshold_config: ThresholdConfig = field(default_factory=ThresholdConfig)
    
    # Clustering
    min_cluster_size: int = 1
    use_confidence_weighting: bool = True
    disagreement_penalty: float = 0.2


@dataclass
class EnsembleMetrics:
    """Metrics from ensemble triage."""
    
    complexity_analysis: ComplexityAnalysis | None = None
    short_circuited: bool = False
    
    n_exception_failures: int = 0
    n_assertion_failures: int = 0
    
    signals_computed: list[str] = field(default_factory=list)
    signal_confidences: dict[str, float] = field(default_factory=dict)
    
    thresholds_used: dict[str, float] = field(default_factory=dict)
    signal_agreements: dict[str, float] = field(default_factory=dict)


class HyTriEnsemble:
    """HyTri-E: Ensemble Failure Triage Pipeline.
    
    This is the main entry point for the ensemble architecture.
    
    Usage:
        ensemble = HyTriEnsemble(openai_api_key="sk-...")
        result = ensemble.triage(failures, changed_files, file_diffs)
    """
    
    def __init__(
        self,
        openai_api_key: str | None = None,
        config: EnsembleConfig | None = None,
    ):
        """Initialize the ensemble pipeline.
        
        Args:
            openai_api_key: OpenAI API key for LLM operations.
            config: Ensemble configuration.
        """
        self.config = config or EnsembleConfig()
        
        # Initialize LLM client
        self.openai_client = OpenAIClient(api_key=openai_api_key)
        
        # Initialize components
        self.complexity_gate = ComplexityGate(
            delta_exception=self.config.delta_exception,
            delta_assertion=self.config.delta_assertion,
            max_files_for_single_cluster=self.config.max_files_for_single_cluster,
            require_file_overlap=self.config.require_file_overlap,
        )
        
        self.stratifier = FailureStratifier()
        
        self.signal_computer = MultiSignalComputer(
            openai_client=self.openai_client,
            use_semantic=True,
        )
        
        self.hypothesis_generator = HypothesisGenerator(
            openai_client=self.openai_client,
            batch_size=self.config.hypothesis_batch_size,
        )
        
        self.aggregator = AggregationEnsemble(
            signal_weights=self.config.signal_weights,
            use_confidence_weighting=self.config.use_confidence_weighting,
            disagreement_penalty=self.config.disagreement_penalty,
            min_cluster_size=self.config.min_cluster_size,
            use_type_adaptation=self.config.enable_stratification,
        )
        
        self.threshold_selector = AdaptiveThreshold(
            config=self.config.threshold_config
        )
        
        # State
        self._last_metrics: EnsembleMetrics | None = None
    
    def triage(
        self,
        failures: list[FailureInstance],
        changed_files: list[str] | None = None,
        file_diffs: dict[str, str] | None = None,
        build_id: str | None = None,
    ) -> TriageResult:
        """Run ensemble triage on failures.
        
        Args:
            failures: List of failure instances.
            changed_files: List of changed file paths.
            file_diffs: Optional dict mapping file path to diff content.
            build_id: Optional build identifier.
            
        Returns:
            TriageResult with clusters and metadata.
        """
        metrics = EnsembleMetrics()
        changed_files = changed_files or []
        file_diffs = file_diffs or {}
        
        if not failures:
            return TriageResult(
                build_id=build_id or "unknown",
                clusters=[],
                priority_queue=[],
                num_failures_total=0,
            )
        
        # Stage 1: Complexity Gate
        if self.config.enable_complexity_gate and changed_files:
            metrics.complexity_analysis = self.complexity_gate.analyze(
                changed_files, file_diffs
            )
            
            if metrics.complexity_analysis.should_single_cluster:
                metrics.short_circuited = True
                return self._create_single_cluster_result(
                    failures, build_id, metrics
                )
        
        # Stage 2: Classify failures for metrics (no hard partition)
        stratified = self.stratifier.stratify(failures)
        metrics.n_exception_failures = len(stratified.exception_failures)
        metrics.n_assertion_failures = len(stratified.assertion_failures)
        
        # Stage 3-4: Single clustering pass on ALL failures.
        # Per-pair weight adaptation in the aggregator handles type-aware
        # signal emphasis without splitting failures into separate pools.
        all_clusters, stratum_metrics = self._process_stratum(
            failures,
            changed_files,
            file_diffs,
            failure_type="mixed",
        )
        self._merge_metrics(metrics, stratum_metrics, "mixed")
        
        # Renumber clusters
        for i, cluster in enumerate(all_clusters):
            cluster.cluster_id = f"cluster_{i}"
        
        self._last_metrics = metrics
        
        return TriageResult(
            build_id=build_id or (failures[0].build_id if failures else "unknown"),
            clusters=all_clusters,
            priority_queue=[c.cluster_id for c in all_clusters],
            num_failures_total=len(failures),
            num_noise_failures=0,
            noise_failure_ids=[],
            metadata=self._build_metadata(metrics),
        )
    
    def _process_stratum(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
        file_diffs: dict[str, str],
        failure_type: str,
    ) -> tuple[list[Cluster], dict]:
        """Process a single stratum (exception or assertion failures).
        
        Args:
            failures: Failures in this stratum.
            changed_files: Changed files.
            file_diffs: File diffs.
            failure_type: "exception" or "assertion"
            
        Returns:
            Tuple of (clusters, stratum_metrics).
        """
        stratum_metrics = {}
        
        if not failures:
            return [], stratum_metrics
        
        if len(failures) == 1:
            cluster = Cluster(
                cluster_id="cluster_0",
                failure_ids=[failures[0].failure_id],
                representative_failure_id=failures[0].failure_id,
            )
            return [cluster], stratum_metrics
        
        # Generate hypotheses (if enabled)
        hypotheses = None
        if self.config.enable_hypothesis and changed_files:
            hypothesis_batch = self.hypothesis_generator.generate(
                failures, changed_files, file_diffs
            )
            hypotheses = hypothesis_batch.hypotheses
        
        # Compute all signals (now including diff-line correlation)
        signals = self.signal_computer.compute(
            failures,
            changed_files=changed_files,
            file_diffs=file_diffs,
            hypotheses=hypotheses,
        )
        
        stratum_metrics['signals'] = list(signals.get_all_signals().keys())
        stratum_metrics['confidences'] = signals.confidences.copy()
        
        # Select adaptive threshold
        threshold = self.threshold_selector.select_threshold(
            signals,
            failure_type=failure_type,
        )
        stratum_metrics['threshold'] = threshold
        
        # Cluster using aggregation ensemble
        result = self.aggregator.cluster(
            failures,
            signals,
            distance_threshold=threshold,
        )
        
        stratum_metrics['signal_agreements'] = result.signal_agreements
        
        # Enrich clusters with hypothesis info
        if hypotheses:
            self._enrich_clusters_with_hypotheses(result.clusters, failures, hypotheses)
        
        return result.clusters, stratum_metrics
    
    def _enrich_clusters_with_hypotheses(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
        hypotheses: list,
    ) -> None:
        """Add hypothesis information to clusters."""
        # Map failure_id to hypothesis
        hypothesis_map = {h.failure_id: h for h in hypotheses}
        failure_map = {f.failure_id: f for f in failures}
        
        for cluster in clusters:
            cluster_hypotheses = []
            suspected_changes = []
            
            for fid in cluster.failure_ids:
                if fid in hypothesis_map:
                    h = hypothesis_map[fid]
                    cluster_hypotheses.append(h)
                    if h.suspected_change:
                        suspected_changes.append(h.suspected_change)
            
            # Find most common suspected change
            if suspected_changes:
                from collections import Counter
                most_common = Counter(suspected_changes).most_common(1)[0][0]
                
                # Add to cluster metadata
                if not cluster.metadata:
                    cluster.metadata = {}
                cluster.metadata['suspected_change'] = most_common
                cluster.metadata['hypothesis_count'] = len(cluster_hypotheses)
                
                # Generate explanation from hypotheses
                mechanisms = [h.mechanism for h in cluster_hypotheses if h.mechanism]
                if mechanisms:
                    cluster.metadata['mechanisms'] = mechanisms[:3]
    
    def _create_single_cluster_result(
        self,
        failures: list[FailureInstance],
        build_id: str | None,
        metrics: EnsembleMetrics,
    ) -> TriageResult:
        """Create result when complexity gate triggers single cluster."""
        cluster = Cluster(
            cluster_id="cluster_0",
            failure_ids=[f.failure_id for f in failures],
            representative_failure_id=failures[0].failure_id,
            exception_types=list(set(f.exception_type for f in failures if f.exception_type)),
        )
        
        # Add explanation about why single cluster
        cluster.metadata = {
            'reason': 'complexity_gate',
            'explanation': metrics.complexity_analysis.reason if metrics.complexity_analysis else "Few changes detected",
        }
        
        self._last_metrics = metrics
        
        return TriageResult(
            build_id=build_id or (failures[0].build_id if failures else "unknown"),
            clusters=[cluster],
            priority_queue=["cluster_0"],
            num_failures_total=len(failures),
            num_noise_failures=0,
            noise_failure_ids=[],
            metadata=self._build_metadata(metrics),
        )
    
    def _merge_metrics(
        self,
        metrics: EnsembleMetrics,
        stratum_metrics: dict,
        stratum_name: str,
    ) -> None:
        """Merge stratum metrics into overall metrics."""
        if 'signals' in stratum_metrics:
            metrics.signals_computed.extend(stratum_metrics['signals'])
        
        if 'confidences' in stratum_metrics:
            for k, v in stratum_metrics['confidences'].items():
                key = f"{stratum_name}_{k}"
                metrics.signal_confidences[key] = v
        
        if 'threshold' in stratum_metrics:
            metrics.thresholds_used[stratum_name] = stratum_metrics['threshold']
        
        if 'signal_agreements' in stratum_metrics:
            for k, v in stratum_metrics['signal_agreements'].items():
                key = f"{stratum_name}_{k}"
                metrics.signal_agreements[key] = v
    
    def _build_metadata(self, metrics: EnsembleMetrics) -> dict:
        """Build metadata dict from metrics."""
        return {
            'pipeline': 'ensemble',
            'short_circuited': metrics.short_circuited,
            'complexity_analysis': {
                'changed_files': metrics.complexity_analysis.changed_files_count,
                'estimated_methods': metrics.complexity_analysis.estimated_changed_methods,
                'should_single_cluster': metrics.complexity_analysis.should_single_cluster,
            } if metrics.complexity_analysis else None,
            'n_exception_failures': metrics.n_exception_failures,
            'n_assertion_failures': metrics.n_assertion_failures,
            'signals_computed': list(set(metrics.signals_computed)),
            'signal_confidences': metrics.signal_confidences,
            'thresholds_used': metrics.thresholds_used,
            'signal_agreements': metrics.signal_agreements,
        }
    
    def get_last_metrics(self) -> EnsembleMetrics | None:
        """Get metrics from the last triage run."""
        return self._last_metrics


class HyTriEnsembleBuilder:
    """Builder for configuring HyTriEnsemble."""
    
    def __init__(self):
        self._openai_api_key: str | None = None
        self._config = EnsembleConfig()
    
    def with_openai_key(self, key: str) -> "HyTriEnsembleBuilder":
        """Set OpenAI API key."""
        self._openai_api_key = key
        return self
    
    def with_complexity_gate(self, enabled: bool = True, delta: int = 7) -> "HyTriEnsembleBuilder":
        """Configure complexity gate."""
        self._config.enable_complexity_gate = enabled
        self._config.delta_exception = delta
        self._config.delta_assertion = delta
        return self
    
    def with_stratification(self, enabled: bool = True) -> "HyTriEnsembleBuilder":
        """Enable/disable failure stratification."""
        self._config.enable_stratification = enabled
        return self
    
    def with_hypothesis(self, enabled: bool = True, batch_size: int = 10) -> "HyTriEnsembleBuilder":
        """Configure hypothesis generation."""
        self._config.enable_hypothesis = enabled
        self._config.hypothesis_batch_size = batch_size
        return self
    
    def with_signal_weights(self, weights: dict[str, float]) -> "HyTriEnsembleBuilder":
        """Set custom signal weights."""
        self._config.signal_weights.update(weights)
        return self
    
    def with_thresholds(
        self,
        dt_stack: float = 0.57,
        dt_message: float = 0.73,
        dt_test: float = 0.50,
    ) -> "HyTriEnsembleBuilder":
        """Set clustering thresholds."""
        self._config.threshold_config.dt_stack = dt_stack
        self._config.threshold_config.dt_message = dt_message
        self._config.threshold_config.dt_test = dt_test
        return self
    
    def build(self) -> HyTriEnsemble:
        """Build the configured ensemble."""
        return HyTriEnsemble(
            openai_api_key=self._openai_api_key,
            config=self._config,
        )
