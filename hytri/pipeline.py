"""HyTri Pipeline - End-to-end failure triage orchestrator.

This module ties together all components (A through E) into a cohesive pipeline
that takes raw failure instances and produces prioritized, explained clusters.

Supports five clustering modes:
- "ensemble": NEW. HyTri-E ensemble combining BuildSheriff decision rules + LLM semantics (recommended)
- "change_aware": Uses LLM + direct file-frame overlap for robust clustering
- "category_guided": Novel approach using LLM categories/components as constraints
- "fused": Traditional alpha-weighted fusion of lexical + semantic similarity
- "two_stage": Two-stage approach (lexical over-cluster → semantic merge)
"""

from datetime import datetime
from typing import Literal

from hytri.clustering.category_guided import CategoryGuidedClusterer, CategoryGuidedConfig
from hytri.ensemble.pipeline import HyTriEnsemble, EnsembleConfig
from hytri.clustering.change_aware import ChangeAwareClusterer, ChangeAwareConfig
from hytri.clustering.clusterer import FailureClusterer
from hytri.clustering.fusion import SimilarityFusion
from hytri.clustering.two_stage import TwoStageClusterer
from hytri.explanation.summarizer import ClusterSummarizer, FallbackSummarizer
from hytri.explanation.verifier import EvidenceVerifier
from hytri.llm.openai_client import OpenAIClient
from hytri.models import Cluster, FailureInstance, TriageResult
from hytri.parsing.log_templates import extract_log_templates
from hytri.parsing.signature import SignatureTokenizer
from hytri.parsing.stack_trace import StackTraceParser
from hytri.prioritization.scorer import PriorityScorer
from hytri.similarity.lexical import LexicalSimilarity
from hytri.similarity.semantic import FallbackSemanticSimilarity, SemanticSimilarity


class HyTriPipeline:
    """End-to-end pipeline for hybrid failure triage.

    Usage:
        pipeline = HyTriPipeline(openai_api_key="sk-...")
        results = pipeline.triage(failures)

    The pipeline executes these steps:
    1. Parse stack traces and extract signatures (Component A)
    2. Compute lexical similarity (Component B1)
    3. Compute semantic similarity via LLM (Component B2)
    4. Fuse similarities and cluster (Component C)
    5. Generate explanations for clusters (Component D)
    6. Score and prioritize clusters (Component E)
    """

    def __init__(
        self,
        openai_api_key: str | None = None,
        clustering_mode: Literal["ensemble", "change_aware", "category_guided", "fused", "two_stage"] = "ensemble",
        alpha: float = 0.4,
        min_cluster_size: int = 2,
        use_semantic: bool = True,
        generate_explanations: bool = True,
        verify_explanations: bool = True,
        # Two-stage specific parameters
        stage1_threshold: float = 0.4,
        merge_threshold: float = 0.85,
        use_llm_merge_validation: bool = False,
        # Category-guided specific parameters
        same_category_boost: float = 0.15,
        diff_category_penalty: float = 0.10,
        same_component_boost: float = 0.10,
        hypothesis_weight: float = 0.05,
        distance_threshold: float = 0.1,  # Tuned: 0.1 achieves ARI=0.838 (beats Rebucket 0.703)
        # Change-aware specific parameters
        direct_overlap_weight: float = 0.5,  # Weight for direct file-frame overlap boost
        # Ensemble specific parameters
        enable_complexity_gate: bool = True,
        enable_stratification: bool = True,
        enable_hypothesis: bool = True,
    ):
        """Initialize the pipeline.

        Args:
            openai_api_key: OpenAI API key for LLM operations.
            clustering_mode: Clustering approach to use:
                - "ensemble": NEW. HyTri-E combining BuildSheriff rules + LLM semantics (recommended)
                - "change_aware": Combines LLM semantic analysis with direct file-frame overlap
                - "category_guided": Novel approach using LLM categories as constraints
                - "fused": Traditional alpha-weighted fusion
                - "two_stage": Lexical over-clustering + semantic merging
            alpha: Weight for lexical vs semantic similarity (0=semantic, 1=lexical).
                   Used in "fused" and "category_guided" modes.
            min_cluster_size: Minimum cluster size for HDBSCAN.
            use_semantic: Whether to use LLM-based semantic similarity.
            generate_explanations: Whether to generate cluster explanations.
            verify_explanations: Whether to verify explanations for groundedness.
            stage1_threshold: Distance threshold for Stage 1 (lower = more clusters).
                             Only used in "two_stage" mode.
            merge_threshold: Semantic similarity threshold for merging in Stage 2.
                            Only used in "two_stage" mode.
            use_llm_merge_validation: Whether to use LLM to validate merge decisions.
                                     Only used in "two_stage" mode.
            same_category_boost: Boost for same-category pairs (category_guided mode).
            diff_category_penalty: Penalty for different-category pairs (category_guided mode).
            same_component_boost: Boost for same-component pairs (category_guided mode).
            hypothesis_weight: Weight for hypothesis similarity (category_guided mode).
            distance_threshold: Clustering distance threshold (category_guided/change_aware mode).
                               Default 0.1 tuned on Defects4J (ARI=0.838, beats Rebucket 0.703).
            direct_overlap_weight: Weight for direct file-frame overlap boost (change_aware mode).
                                  Higher values give more weight to BuildSheriff-style matching.
            enable_complexity_gate: Enable BuildSheriff S1 complexity check (ensemble mode).
            enable_stratification: Enable exception/assertion stratification (ensemble mode).
            enable_hypothesis: Enable LLM hypothesis generation (ensemble mode).
        """
        self.openai_api_key = openai_api_key
        self.clustering_mode = clustering_mode
        self.alpha = alpha
        self.min_cluster_size = min_cluster_size
        self.use_semantic = use_semantic
        self.generate_explanations = generate_explanations
        self.verify_explanations = verify_explanations
        
        # Two-stage parameters
        self.stage1_threshold = stage1_threshold
        self.merge_threshold = merge_threshold
        self.use_llm_merge_validation = use_llm_merge_validation
        
        # Category-guided parameters
        self.same_category_boost = same_category_boost
        self.diff_category_penalty = diff_category_penalty
        self.same_component_boost = same_component_boost
        self.hypothesis_weight = hypothesis_weight
        self.distance_threshold = distance_threshold
        
        # Change-aware parameters
        self.direct_overlap_weight = direct_overlap_weight
        
        # Ensemble parameters
        self.enable_complexity_gate = enable_complexity_gate
        self.enable_stratification = enable_stratification
        self.enable_hypothesis = enable_hypothesis
        
        # Ensemble pipeline (initialized if mode is "ensemble")
        self.ensemble_pipeline: HyTriEnsemble | None = None
        
        # Store changed files for ensemble mode
        self._changed_files: list[str] = []
        self._file_diffs: dict[str, str] = {}

        # Initialize components
        self._init_components()

    def _init_components(self) -> None:
        """Initialize all pipeline components."""
        # Component A: Parsing
        self.stack_parser = StackTraceParser()
        self.signature_tokenizer = SignatureTokenizer()

        # Component B: Similarity
        self.lexical_sim = LexicalSimilarity()

        # LLM client (shared)
        self.llm_client = OpenAIClient(api_key=self.openai_api_key)

        if self.use_semantic and self.llm_client.is_available():
            self.semantic_sim = SemanticSimilarity(openai_client=self.llm_client)
        else:
            self.semantic_sim = FallbackSemanticSimilarity()  # type: ignore

        # Component C: Clustering (mode-dependent)
        self.change_aware_clusterer = None
        self.category_guided_clusterer = None
        self.two_stage_clusterer = None
        self.clusterer = None
        self.fusion = None
        
        if self.clustering_mode == "ensemble":
            # Ensemble mode: HyTri-E with BuildSheriff rules + LLM semantics
            config = EnsembleConfig(
                enable_complexity_gate=self.enable_complexity_gate,
                enable_stratification=self.enable_stratification,
                enable_hypothesis=self.enable_hypothesis,
                min_cluster_size=self.min_cluster_size,
            )
            self.ensemble_pipeline = HyTriEnsemble(
                openai_api_key=self.openai_api_key,
                config=config,
            )
        elif self.clustering_mode == "change_aware":
            # Change-aware: combines LLM semantic analysis with direct file-frame overlap
            # This is the most robust mode, working well even when stack frames lack file info
            config = ChangeAwareConfig(
                alpha=self.alpha,
                same_category_boost=self.same_category_boost,
                diff_category_penalty=self.diff_category_penalty,
                same_component_boost=self.same_component_boost,
                hypothesis_weight=self.hypothesis_weight,
                distance_threshold=self.distance_threshold,
                min_cluster_size=self.min_cluster_size,
            )
            self.change_aware_clusterer = ChangeAwareClusterer(
                config=config,
                openai_client=self.llm_client,
                direct_overlap_weight=self.direct_overlap_weight,
            )
        elif self.clustering_mode == "category_guided":
            # Category-guided: uses LLM categories/components as constraints
            config = CategoryGuidedConfig(
                alpha=self.alpha,
                same_category_boost=self.same_category_boost,
                diff_category_penalty=self.diff_category_penalty,
                same_component_boost=self.same_component_boost,
                hypothesis_weight=self.hypothesis_weight,
                distance_threshold=self.distance_threshold,
                min_cluster_size=self.min_cluster_size,
            )
            self.category_guided_clusterer = CategoryGuidedClusterer(
                config=config,
                openai_client=self.llm_client,
            )
        elif self.clustering_mode == "two_stage":
            # Two-stage: lexical over-cluster + semantic merge
            self.two_stage_clusterer = TwoStageClusterer(
                stage1_distance_threshold=self.stage1_threshold,
                semantic_merge_threshold=self.merge_threshold,
                use_llm_validation=self.use_llm_merge_validation,
                openai_client=self.llm_client if self.use_llm_merge_validation else None,
                min_cluster_size=self.min_cluster_size,
            )
        else:
            # Fused mode: traditional alpha-weighted fusion
            self.fusion = SimilarityFusion(alpha=self.alpha)
            self.clusterer = FailureClusterer(min_cluster_size=self.min_cluster_size)

        # Component D: Explanation
        if self.generate_explanations and self.llm_client.is_available():
            self.summarizer = ClusterSummarizer(openai_client=self.llm_client)
        else:
            self.summarizer = FallbackSummarizer()  # type: ignore

        self.verifier = EvidenceVerifier() if self.verify_explanations else None

        # Component E: Prioritization
        self.scorer = PriorityScorer()

    def triage(
        self,
        failures: list[FailureInstance],
        build_id: str | None = None,
        reference_time: datetime | None = None,
    ) -> TriageResult:
        """Run full triage pipeline on failures.

        Args:
            failures: List of failure instances to triage.
            build_id: Optional build ID for the result.
            reference_time: Reference time for recency scoring.

        Returns:
            TriageResult with clusters and priority queue.
        """
        if not failures:
            return TriageResult(
                build_id=build_id or "unknown",
                clusters=[],
                priority_queue=[],
                num_failures_total=0,
            )

        # Step 1: Parse and extract features (Component A)
        failures = self._parse_failures(failures)

        # Special case: Ensemble mode delegates to HyTriEnsemble
        if self.clustering_mode == "ensemble" and self.ensemble_pipeline:
            result = self.ensemble_pipeline.triage(
                failures,
                changed_files=self._changed_files,
                file_diffs=self._file_diffs,
                build_id=build_id,
            )
            # Generate explanations if enabled (ensemble doesn't do this by default)
            if self.generate_explanations:
                result.clusters = self._generate_explanations(result.clusters, failures)
            return result

        # Step 2: Compute lexical similarity (Component B1)
        s_lex = self.lexical_sim.compute_similarity_matrix(failures)

        # Step 3-4: Cluster (Component C) - mode-dependent
        merge_history = []
        
        if self.clustering_mode == "change_aware":
            # Change-aware: combines LLM semantic analysis with direct file-frame overlap
            # The clusterer automatically boosts similarity for failures touching same changed files
            clusters, noise_ids, normalizations = self.change_aware_clusterer.cluster(
                failures, s_lex
            )
        elif self.clustering_mode == "category_guided":
            # Category-guided: uses LLM categories/components as constraints
            # (semantic similarity computed internally with structured output)
            clusters, noise_ids, normalizations = self.category_guided_clusterer.cluster(
                failures, s_lex
            )
        elif self.clustering_mode == "two_stage":
            # Compute semantic similarity for two-stage
            s_sem = self.semantic_sim.compute_similarity_matrix(failures)
            # Two-stage: lexical over-cluster → semantic merge
            clusters, noise_ids, merge_history = self.two_stage_clusterer.cluster(
                failures, s_lex, s_sem
            )
        else:
            # Fused mode: traditional alpha-weighted fusion
            s_sem = self.semantic_sim.compute_similarity_matrix(failures)
            exception_types = [f.exception_type for f in failures]
            s_fused = self.fusion.fuse(s_lex, s_sem, exception_types)
            clusters, noise_ids = self.clusterer.cluster(s_fused, failures)

        # Step 5: Generate explanations (Component D)
        if self.generate_explanations:
            clusters = self._generate_explanations(clusters, failures)

        # Step 6: Prioritize (Component E)
        priority_queue = self.scorer.get_priority_queue(
            clusters, failures, reference_time
        )

        # Build metadata
        metadata = {
            "clustering_mode": self.clustering_mode,
            "min_cluster_size": self.min_cluster_size,
            "use_semantic": self.use_semantic,
            "num_clusters": len(clusters),
        }
        
        if self.clustering_mode == "change_aware":
            metadata["alpha"] = self.alpha
            metadata["same_category_boost"] = self.same_category_boost
            metadata["diff_category_penalty"] = self.diff_category_penalty
            metadata["distance_threshold"] = self.distance_threshold
            metadata["direct_overlap_weight"] = self.direct_overlap_weight
        elif self.clustering_mode == "category_guided":
            metadata["alpha"] = self.alpha
            metadata["same_category_boost"] = self.same_category_boost
            metadata["diff_category_penalty"] = self.diff_category_penalty
            metadata["distance_threshold"] = self.distance_threshold
        elif self.clustering_mode == "two_stage":
            metadata["stage1_threshold"] = self.stage1_threshold
            metadata["merge_threshold"] = self.merge_threshold
            metadata["num_merges"] = len([m for m in merge_history if m.merged])
        else:
            metadata["alpha"] = self.alpha

        # Build result
        return TriageResult(
            build_id=build_id or failures[0].build_id if failures else "unknown",
            clusters=clusters,
            priority_queue=priority_queue,
            num_failures_total=len(failures),
            num_noise_failures=len(noise_ids),
            noise_failure_ids=noise_ids,
            metadata=metadata,
        )

    def set_build_changes(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> None:
        """Set the code changes for change-aware or ensemble clustering.
        
        This provides the change context that enables direct file-frame overlap
        matching (BuildSheriff-style) and LLM-based change attribution.
        
        Must be called before triage() when using change_aware or ensemble mode.
        
        Args:
            changed_files: List of changed file paths in this build.
            file_diffs: Optional dict mapping file path to diff content.
        """
        # Store for ensemble mode
        self._changed_files = changed_files
        self._file_diffs = file_diffs or {}
        
        # Also set for change_aware mode
        if self.clustering_mode == "change_aware" and self.change_aware_clusterer:
            self.change_aware_clusterer.set_changes_from_build(
                changed_files,
                file_diffs or {},
            )
    
    def _parse_failures(
        self, failures: list[FailureInstance]
    ) -> list[FailureInstance]:
        """Parse and enrich failure instances.

        Populates frames, signature_tokens, and log_templates.
        """
        for failure in failures:
            # Parse stack trace if frames not already populated
            if not failure.frames:
                parsed = self.stack_parser.parse(failure.stack_trace)
                failure.frames = parsed.frames

            # Generate signature tokens
            self.signature_tokenizer.update_failure_signature(failure)

            # Extract log templates
            if failure.stderr or failure.stdout:
                failure.log_templates = extract_log_templates(
                    failure.stderr, failure.stdout
                )

        return failures

    def _generate_explanations(
        self,
        clusters: list[Cluster],
        failures: list[FailureInstance],
    ) -> list[Cluster]:
        """Generate and verify explanations for clusters."""
        for cluster in clusters:
            # Generate explanation
            explanation = self.summarizer.summarize_cluster(cluster, failures)
            cluster.explanation = explanation

            # Verify if enabled
            if self.verifier and cluster.explanation:
                self.verifier.verify_and_update(cluster, failures)

        return clusters

    def triage_batch(
        self,
        builds: list[tuple[str, list[FailureInstance]]],
    ) -> list[TriageResult]:
        """Triage multiple builds.

        Args:
            builds: List of (build_id, failures) tuples.

        Returns:
            List of TriageResult objects.
        """
        return [
            self.triage(failures, build_id)
            for build_id, failures in builds
        ]

    def get_cluster_by_id(
        self,
        result: TriageResult,
        cluster_id: str,
    ) -> Cluster | None:
        """Get a cluster by its ID from a triage result.

        Args:
            result: Triage result to search.
            cluster_id: ID of cluster to find.

        Returns:
            Cluster if found, None otherwise.
        """
        for cluster in result.clusters:
            if cluster.cluster_id == cluster_id:
                return cluster
        return None


class HyTriPipelineBuilder:
    """Builder for configuring HyTriPipeline with custom components."""

    def __init__(self):
        """Initialize the builder with defaults."""
        self._openai_api_key: str | None = None
        self._clustering_mode: Literal["ensemble", "change_aware", "category_guided", "fused", "two_stage"] = "ensemble"
        self._alpha: float = 0.4
        self._min_cluster_size: int = 2
        self._use_semantic: bool = True
        self._generate_explanations: bool = True
        self._verify_explanations: bool = True
        # Two-stage defaults
        self._stage1_threshold: float = 0.4
        self._merge_threshold: float = 0.85
        self._use_llm_merge_validation: bool = False
        # Category-guided defaults (tuned on Defects4J)
        self._same_category_boost: float = 0.15
        self._diff_category_penalty: float = 0.10
        self._same_component_boost: float = 0.10
        self._hypothesis_weight: float = 0.05
        self._distance_threshold: float = 0.1  # Achieves ARI=0.838
        # Change-aware defaults
        self._direct_overlap_weight: float = 0.5
        # Ensemble defaults
        self._enable_complexity_gate: bool = True
        self._enable_stratification: bool = True
        self._enable_hypothesis: bool = True

    def with_openai_key(self, key: str) -> "HyTriPipelineBuilder":
        """Set OpenAI API key."""
        self._openai_api_key = key
        return self

    def with_clustering_mode(
        self, mode: Literal["ensemble", "change_aware", "category_guided", "fused", "two_stage"]
    ) -> "HyTriPipelineBuilder":
        """Set clustering mode ('ensemble', 'change_aware', 'category_guided', 'fused', or 'two_stage')."""
        self._clustering_mode = mode
        return self
    
    def with_ensemble_params(
        self,
        enable_complexity_gate: bool = True,
        enable_stratification: bool = True,
        enable_hypothesis: bool = True,
    ) -> "HyTriPipelineBuilder":
        """Configure ensemble mode parameters.
        
        Args:
            enable_complexity_gate: Enable BuildSheriff S1 complexity check.
            enable_stratification: Enable exception/assertion stratification.
            enable_hypothesis: Enable LLM hypothesis generation.
        """
        self._enable_complexity_gate = enable_complexity_gate
        self._enable_stratification = enable_stratification
        self._enable_hypothesis = enable_hypothesis
        return self

    def with_alpha(self, alpha: float) -> "HyTriPipelineBuilder":
        """Set similarity fusion alpha (for 'fused' and 'category_guided' modes)."""
        self._alpha = alpha
        return self

    def with_two_stage_params(
        self,
        stage1_threshold: float = 0.4,
        merge_threshold: float = 0.85,
        use_llm_validation: bool = False,
    ) -> "HyTriPipelineBuilder":
        """Configure two-stage clustering parameters.
        
        Args:
            stage1_threshold: Distance threshold for Stage 1 (lower = more clusters).
            merge_threshold: Semantic similarity threshold for merging.
            use_llm_validation: Whether to use LLM to validate merges.
        """
        self._stage1_threshold = stage1_threshold
        self._merge_threshold = merge_threshold
        self._use_llm_merge_validation = use_llm_validation
        return self

    def with_category_guided_params(
        self,
        same_category_boost: float = 0.15,
        diff_category_penalty: float = 0.10,
        same_component_boost: float = 0.10,
        hypothesis_weight: float = 0.05,
        distance_threshold: float = 0.1,
    ) -> "HyTriPipelineBuilder":
        """Configure category-guided clustering parameters.
        
        Args:
            same_category_boost: Boost for pairs with same LLM category.
            diff_category_penalty: Penalty for pairs with different categories.
            same_component_boost: Boost for pairs with same suspected component.
            hypothesis_weight: Weight for hypothesis similarity.
            distance_threshold: Clustering distance threshold.
        """
        self._same_category_boost = same_category_boost
        self._diff_category_penalty = diff_category_penalty
        self._same_component_boost = same_component_boost
        self._hypothesis_weight = hypothesis_weight
        self._distance_threshold = distance_threshold
        return self
    
    def with_change_aware_params(
        self,
        direct_overlap_weight: float = 0.5,
        same_category_boost: float = 0.15,
        diff_category_penalty: float = 0.10,
        distance_threshold: float = 0.1,
    ) -> "HyTriPipelineBuilder":
        """Configure change-aware clustering parameters.
        
        Args:
            direct_overlap_weight: Weight for direct file-frame overlap boost (0-1).
                                  Higher values give more weight to BuildSheriff-style matching.
            same_category_boost: Boost for pairs with same LLM category.
            diff_category_penalty: Penalty for pairs with different categories.
            distance_threshold: Clustering distance threshold.
        """
        self._direct_overlap_weight = direct_overlap_weight
        self._same_category_boost = same_category_boost
        self._diff_category_penalty = diff_category_penalty
        self._distance_threshold = distance_threshold
        return self

    def with_min_cluster_size(self, size: int) -> "HyTriPipelineBuilder":
        """Set minimum cluster size."""
        self._min_cluster_size = size
        return self

    def without_semantic(self) -> "HyTriPipelineBuilder":
        """Disable semantic similarity (LLM-free mode)."""
        self._use_semantic = False
        return self

    def without_explanations(self) -> "HyTriPipelineBuilder":
        """Disable explanation generation."""
        self._generate_explanations = False
        return self

    def without_verification(self) -> "HyTriPipelineBuilder":
        """Disable explanation verification."""
        self._verify_explanations = False
        return self

    def build(self) -> HyTriPipeline:
        """Build the configured pipeline."""
        return HyTriPipeline(
            openai_api_key=self._openai_api_key,
            clustering_mode=self._clustering_mode,
            alpha=self._alpha,
            min_cluster_size=self._min_cluster_size,
            use_semantic=self._use_semantic,
            generate_explanations=self._generate_explanations,
            verify_explanations=self._verify_explanations,
            stage1_threshold=self._stage1_threshold,
            merge_threshold=self._merge_threshold,
            use_llm_merge_validation=self._use_llm_merge_validation,
            same_category_boost=self._same_category_boost,
            diff_category_penalty=self._diff_category_penalty,
            same_component_boost=self._same_component_boost,
            hypothesis_weight=self._hypothesis_weight,
            distance_threshold=self._distance_threshold,
            direct_overlap_weight=self._direct_overlap_weight,
            enable_complexity_gate=self._enable_complexity_gate,
            enable_stratification=self._enable_stratification,
            enable_hypothesis=self._enable_hypothesis,
        )


