"""Category-Guided Constrained Clustering.

A novel hybrid clustering approach that integrates structured LLM understanding
(categories, components, hypotheses) directly into similarity computation,
enabling constraint-guided clustering.

Unlike prior work that uses LLMs as post-processors, this approach fuses
lexical similarity, semantic embeddings, AND categorical constraints from
the LLM's structured output.
"""

import numpy as np
from collections import defaultdict
from dataclasses import dataclass

from hytri.llm.openai_client import OpenAIClient
from hytri.models import Cluster, FailureInstance, NormalizedFailure
from hytri.similarity.semantic import SemanticSimilarity


@dataclass
class CategoryGuidedConfig:
    """Configuration for category-guided clustering."""
    
    # Weight for lexical similarity (rest goes to semantic)
    alpha: float = 0.4
    
    # Category constraint weights
    same_category_boost: float = 0.15      # Boost when categories match
    diff_category_penalty: float = 0.10    # Penalty when categories differ
    
    # Component affinity
    same_component_boost: float = 0.10     # Boost when suspected_component matches
    
    # Hypothesis similarity weight (added to semantic)
    hypothesis_weight: float = 0.05
    
    # Clustering parameters
    distance_threshold: float = 0.1        # Tuned on Defects4J (ARI=0.838)
    min_cluster_size: int = 1
    
    # Whether to use adaptive weights per category
    adaptive_weights: bool = False
    
    # Adaptive threshold based on exception diversity
    adaptive_threshold: bool = True        # Enable adaptive threshold by default
    threshold_low_diversity: float = 0.5   # Threshold when diversity < 0.1
    threshold_medium_diversity: float = 0.3 # Threshold when diversity 0.1-0.3
    threshold_high_diversity: float = 0.1  # Threshold when diversity > 0.3
    
    # === NEW: Cost-free signal weights ===
    # Exception type matching (FREE - no LLM needed)
    same_exception_boost: float = 0.12     # Boost when exception types match exactly
    exception_family_boost: float = 0.06   # Boost for same exception family (e.g., both ValueError)
    
    # Test name similarity (FREE - no LLM needed)
    same_test_class_boost: float = 0.08    # Boost when tests from same class
    same_test_module_boost: float = 0.04   # Boost when tests from same module
    
    # Identical message must-link (FREE - very strong signal)
    identical_message_boost: float = 0.25  # Strong boost for identical exception messages
    
    # Linkage method for hierarchical clustering
    linkage_method: str = "ward"           # 'ward' often better than 'average'
    
    # === NEW: Two-Stage Clustering with Must-Link Constraints ===
    use_two_stage: bool = True             # Enable two-stage clustering
    
    # Must-link threshold: pairs with similarity above this are FORCED together
    must_link_threshold: float = 0.95      # Very high = only near-identical failures
    
    # Pre-cluster by exception signature (type + normalized message)
    precluster_by_signature: bool = True   # Group identical signatures first
    
    # Single root cause heuristic (like BuildSheriff)
    # If very few distinct exception types, assume single root cause
    single_cause_exception_threshold: int = 1  # If ≤1 unique exception types → 1 cluster
    single_cause_enabled: bool = True


class CategoryGuidedFusion:
    """Fuses lexical, semantic, and categorical signals for clustering.
    
    This is the core novel contribution: instead of simple weighted fusion,
    we use the LLM's structured understanding to create soft constraints
    that guide the clustering process.
    
    Signals used:
    1. Lexical similarity (BM25, frame overlap)
    2. Semantic embeddings (from normalized_signature)
    3. Category matching (from LLM's category field)
    4. Component affinity (from LLM's suspected_component)
    5. Hypothesis similarity (from LLM's likely_root_cause_hypothesis)
    """
    
    def __init__(
        self,
        config: CategoryGuidedConfig | None = None,
        openai_client: OpenAIClient | None = None,
    ):
        """Initialize category-guided fusion.
        
        Args:
            config: Configuration parameters.
            openai_client: OpenAI client for semantic similarity.
        """
        self.config = config or CategoryGuidedConfig()
        self.openai_client = openai_client or OpenAIClient()
        self.semantic_sim = SemanticSimilarity(openai_client=self.openai_client)
        
        # Cache for normalizations
        self._norm_cache: dict[str, NormalizedFailure] = {}
    
    def compute_fused_similarity(
        self,
        failures: list[FailureInstance],
        s_lex: np.ndarray,
    ) -> tuple[np.ndarray, list[NormalizedFailure]]:
        """Compute category-guided fused similarity matrix.
        
        Args:
            failures: List of failure instances.
            s_lex: Pre-computed lexical similarity matrix.
            
        Returns:
            Tuple of (fused similarity matrix, list of normalizations).
        """
        n = len(failures)
        if n == 0:
            return np.array([]), []
        
        # Step 1: Get LLM normalizations (categories, components, hypotheses)
        normalizations = self._get_normalizations(failures)
        
        # Step 2: Compute semantic embedding similarity
        s_sem = self._compute_semantic_similarity(failures, normalizations)
        
        # Step 3: Compute category constraint matrix
        s_cat = self._compute_category_matrix(normalizations)
        
        # Step 4: Compute component affinity matrix
        s_comp = self._compute_component_matrix(normalizations)
        
        # Step 5: Compute hypothesis similarity matrix
        s_hyp = self._compute_hypothesis_similarity(normalizations)
        
        # Step 6: NEW - Compute cost-free signals (no LLM needed!)
        s_exc = self._compute_exception_type_matrix(failures)
        s_test = self._compute_test_name_matrix(failures)
        s_msg = self._compute_identical_message_matrix(failures)
        
        # Step 7: Fuse all signals
        fused = self._fuse_signals(
            s_lex, s_sem, s_cat, s_comp, s_hyp, 
            s_exc, s_test, s_msg,
            normalizations
        )
        
        return fused, normalizations
    
    def _get_normalizations(
        self, failures: list[FailureInstance]
    ) -> list[NormalizedFailure]:
        """Get LLM normalizations for all failures."""
        normalizations = []
        
        for failure in failures:
            if failure.failure_id in self._norm_cache:
                normalizations.append(self._norm_cache[failure.failure_id])
            else:
                norm = self.openai_client.normalize_failure(
                    exception_type=failure.exception_type,
                    exception_message=failure.exception_message,
                    stack_trace=failure.stack_trace,
                    test_name=failure.test_name,
                )
                self._norm_cache[failure.failure_id] = norm
                normalizations.append(norm)
        
        return normalizations
    
    def _compute_semantic_similarity(
        self,
        failures: list[FailureInstance],
        normalizations: list[NormalizedFailure],
    ) -> np.ndarray:
        """Compute semantic similarity from normalized signatures."""
        n = len(failures)
        
        # Get signatures
        signatures = [n.normalized_signature for n in normalizations]
        
        # Get embeddings
        embeddings = self.openai_client.get_embeddings(signatures)
        embeddings_matrix = np.array(embeddings)
        
        # Normalize for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_embeddings = embeddings_matrix / norms
        
        # Cosine similarity
        sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)
        sim_matrix = (sim_matrix + 1) / 2  # Scale to [0, 1]
        np.fill_diagonal(sim_matrix, 1.0)
        
        return sim_matrix
    
    def _compute_category_matrix(
        self, normalizations: list[NormalizedFailure]
    ) -> np.ndarray:
        """Compute category constraint matrix.
        
        Returns a matrix where:
        - Same category pairs get +1
        - Different category pairs get -1
        - Will be scaled by boost/penalty weights during fusion
        """
        n = len(normalizations)
        categories = [n.category.lower().strip() for n in normalizations]
        
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if categories[i] == categories[j]:
                    matrix[i, j] = 1.0
                    matrix[j, i] = 1.0
                else:
                    matrix[i, j] = -1.0
                    matrix[j, i] = -1.0
        
        np.fill_diagonal(matrix, 0.0)  # No self-constraint
        return matrix
    
    def _compute_component_matrix(
        self, normalizations: list[NormalizedFailure]
    ) -> np.ndarray:
        """Compute component affinity matrix.
        
        Failures in the same suspected_component are more likely to be related.
        """
        n = len(normalizations)
        components = [n.suspected_component.lower().strip() for n in normalizations]
        
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                # Check for exact match or substring containment
                comp_i, comp_j = components[i], components[j]
                if comp_i and comp_j:
                    if comp_i == comp_j:
                        matrix[i, j] = 1.0
                    elif comp_i in comp_j or comp_j in comp_i:
                        matrix[i, j] = 0.5
                    else:
                        # Check for shared prefix (e.g., "auth.login" and "auth.session")
                        parts_i = comp_i.split('.')
                        parts_j = comp_j.split('.')
                        if parts_i[0] == parts_j[0] and parts_i[0]:
                            matrix[i, j] = 0.3
                    
                    matrix[j, i] = matrix[i, j]
        
        return matrix
    
    def _compute_hypothesis_similarity(
        self, normalizations: list[NormalizedFailure]
    ) -> np.ndarray:
        """Compute similarity between root cause hypotheses.
        
        Uses Jaccard similarity on hypothesis words.
        """
        n = len(normalizations)
        
        # Tokenize hypotheses
        hypothesis_tokens = []
        for norm in normalizations:
            tokens = set(norm.likely_root_cause_hypothesis.lower().split())
            # Remove common stop words
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'to', 'in', 'on', 'of', 'for', 'and', 'or'}
            tokens = tokens - stop_words
            hypothesis_tokens.append(tokens)
        
        matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if hypothesis_tokens[i] and hypothesis_tokens[j]:
                    intersection = len(hypothesis_tokens[i] & hypothesis_tokens[j])
                    union = len(hypothesis_tokens[i] | hypothesis_tokens[j])
                    sim = intersection / union if union > 0 else 0.0
                else:
                    sim = 0.0
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        np.fill_diagonal(matrix, 1.0)
        return matrix
    
    def _compute_exception_type_matrix(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute exception type similarity matrix (FREE - no LLM needed).
        
        Exact match → 1.0
        Same family (e.g., both *Error or both *Exception) → 0.5
        Different → 0.0
        """
        n = len(failures)
        matrix = np.zeros((n, n))
        
        # Extract and normalize exception types
        exc_types = []
        for f in failures:
            exc = f.exception_type or ""
            # Normalize: remove package prefixes, lowercase
            exc = exc.split('.')[-1].lower()
            exc_types.append(exc)
        
        # Define exception families
        def get_family(exc: str) -> str:
            if 'assertion' in exc:
                return 'assertion'
            elif 'value' in exc or 'type' in exc or 'attribute' in exc:
                return 'type_value'
            elif 'key' in exc or 'index' in exc or 'bounds' in exc:
                return 'access'
            elif 'null' in exc or 'none' in exc or 'nil' in exc:
                return 'null'
            elif 'io' in exc or 'file' in exc or 'path' in exc:
                return 'io'
            elif 'timeout' in exc or 'connection' in exc or 'network' in exc:
                return 'network'
            elif 'parse' in exc or 'format' in exc or 'syntax' in exc:
                return 'parse'
            else:
                return 'other'
        
        families = [get_family(exc) for exc in exc_types]
        
        for i in range(n):
            for j in range(i + 1, n):
                if exc_types[i] == exc_types[j]:
                    # Exact match
                    sim = 1.0
                elif families[i] == families[j]:
                    # Same family
                    sim = 0.5
                else:
                    sim = 0.0
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        np.fill_diagonal(matrix, 1.0)
        return matrix
    
    def _compute_test_name_matrix(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute test name similarity matrix (FREE - no LLM needed).
        
        Tests from same class → higher similarity
        Tests from same module → medium similarity
        """
        n = len(failures)
        matrix = np.zeros((n, n))
        
        # Parse test names to extract module/class/method
        test_parts = []
        for f in failures:
            test = f.test_name or ""
            # Handle different formats: 
            # - Python: module::class::method or module.class.method
            # - Java: com.package.Class::method or com.package.Class.method
            parts = test.replace('::', '.').split('.')
            
            if len(parts) >= 3:
                module = '.'.join(parts[:-2])
                cls = parts[-2]
                method = parts[-1]
            elif len(parts) == 2:
                module = ''
                cls = parts[0]
                method = parts[1]
            else:
                module = ''
                cls = ''
                method = test
            
            test_parts.append({'module': module, 'class': cls, 'method': method})
        
        for i in range(n):
            for j in range(i + 1, n):
                p_i, p_j = test_parts[i], test_parts[j]
                
                if p_i['class'] and p_i['class'] == p_j['class']:
                    # Same test class
                    sim = 1.0
                elif p_i['module'] and p_i['module'] == p_j['module']:
                    # Same module
                    sim = 0.5
                else:
                    sim = 0.0
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        np.fill_diagonal(matrix, 1.0)
        return matrix
    
    def _compute_identical_message_matrix(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute identical message matrix (FREE - very strong signal).
        
        If two failures have IDENTICAL exception messages, they're almost
        certainly from the same root cause. This is a must-link constraint.
        """
        n = len(failures)
        matrix = np.zeros((n, n))
        
        # Normalize messages (strip whitespace, lowercase)
        messages = []
        for f in failures:
            msg = (f.exception_message or "").strip().lower()
            # Remove variable parts like line numbers, memory addresses
            import re
            msg = re.sub(r'0x[0-9a-f]+', '<addr>', msg)
            msg = re.sub(r'line \d+', 'line <n>', msg)
            msg = re.sub(r':\d+:', ':<n>:', msg)
            messages.append(msg)
        
        for i in range(n):
            for j in range(i + 1, n):
                if messages[i] and messages[j]:
                    if messages[i] == messages[j]:
                        # Identical messages - very strong signal!
                        sim = 1.0
                    elif len(messages[i]) > 20 and len(messages[j]) > 20:
                        # For longer messages, check substring containment
                        if messages[i] in messages[j] or messages[j] in messages[i]:
                            sim = 0.7
                        else:
                            sim = 0.0
                    else:
                        sim = 0.0
                else:
                    sim = 0.0
                
                matrix[i, j] = sim
                matrix[j, i] = sim
        
        np.fill_diagonal(matrix, 1.0)
        return matrix
    
    def _fuse_signals(
        self,
        s_lex: np.ndarray,
        s_sem: np.ndarray,
        s_cat: np.ndarray,
        s_comp: np.ndarray,
        s_hyp: np.ndarray,
        s_exc: np.ndarray,
        s_test: np.ndarray,
        s_msg: np.ndarray,
        normalizations: list[NormalizedFailure],
    ) -> np.ndarray:
        """Fuse all similarity signals with category constraints.
        
        Enhanced formula with cost-free signals:
        S_fused = α·S_lex + (1-α)·S_sem 
                  + boost·(S_cat > 0)·S_cat 
                  - penalty·(S_cat < 0)·|S_cat|
                  + comp_boost·S_comp
                  + hyp_weight·S_hyp
                  + exc_boost·S_exc          # NEW: Exception type matching
                  + test_boost·S_test        # NEW: Test name similarity  
                  + msg_boost·S_msg          # NEW: Identical message (must-link)
        """
        cfg = self.config
        
        # Base fusion of lexical and semantic
        base = cfg.alpha * s_lex + (1 - cfg.alpha) * s_sem
        
        # Apply category constraints
        # Positive constraint (same category): boost
        same_cat_mask = s_cat > 0
        category_boost = cfg.same_category_boost * same_cat_mask * s_cat
        
        # Negative constraint (different category): penalty
        diff_cat_mask = s_cat < 0
        category_penalty = cfg.diff_category_penalty * diff_cat_mask * np.abs(s_cat)
        
        # Component affinity
        component_boost = cfg.same_component_boost * s_comp
        
        # Hypothesis similarity
        hypothesis_boost = cfg.hypothesis_weight * s_hyp
        
        # NEW: Cost-free signals (no LLM needed!)
        # Exception type matching
        exc_exact_mask = s_exc == 1.0
        exc_family_mask = (s_exc > 0) & (s_exc < 1.0)
        exception_boost = (
            cfg.same_exception_boost * exc_exact_mask * s_exc +
            cfg.exception_family_boost * exc_family_mask * s_exc
        )
        
        # Test name similarity
        test_class_mask = s_test == 1.0
        test_module_mask = (s_test > 0) & (s_test < 1.0)
        test_boost = (
            cfg.same_test_class_boost * test_class_mask * s_test +
            cfg.same_test_module_boost * test_module_mask * s_test
        )
        
        # Identical message (strongest signal - almost a must-link constraint)
        message_boost = cfg.identical_message_boost * s_msg
        
        # Combine all signals
        fused = (
            base + 
            category_boost - category_penalty + 
            component_boost + 
            hypothesis_boost +
            exception_boost +
            test_boost +
            message_boost
        )
        
        # Adaptive weighting by category (optional)
        if cfg.adaptive_weights:
            fused = self._apply_adaptive_weights(fused, normalizations)
        
        # Clip to [0, 1]
        fused = np.clip(fused, 0.0, 1.0)
        np.fill_diagonal(fused, 1.0)
        
        return fused
    
    def _apply_adaptive_weights(
        self,
        similarity: np.ndarray,
        normalizations: list[NormalizedFailure],
    ) -> np.ndarray:
        """Apply adaptive weights based on category characteristics.
        
        Some categories (like "Assertion failure") may need different 
        weighting than others (like "Connection error").
        """
        # Category-specific adjustments
        # (This could be learned, but we use heuristics for now)
        category_adjustments = {
            "assertion failure": 1.1,  # Assertions often have varied messages
            "type error": 0.95,
            "import error": 0.9,       # Import errors are very specific
            "connection error": 1.0,
            "timeout": 0.95,
        }
        
        n = len(normalizations)
        adjusted = similarity.copy()
        
        for i in range(n):
            cat_i = normalizations[i].category.lower()
            for j in range(i + 1, n):
                cat_j = normalizations[j].category.lower()
                
                # Apply adjustment for same-category pairs
                if cat_i == cat_j:
                    adj = category_adjustments.get(cat_i, 1.0)
                    adjusted[i, j] *= adj
                    adjusted[j, i] *= adj
        
        return adjusted
    
    def clear_cache(self) -> None:
        """Clear normalization cache."""
        self._norm_cache.clear()


class CategoryGuidedClusterer:
    """Clusterer using category-guided fusion.
    
    Uses agglomerative clustering on the fused similarity matrix.
    """
    
    def __init__(
        self,
        config: CategoryGuidedConfig | None = None,
        openai_client: OpenAIClient | None = None,
    ):
        """Initialize the clusterer.
        
        Args:
            config: Configuration parameters.
            openai_client: OpenAI client for LLM operations.
        """
        self.config = config or CategoryGuidedConfig()
        self.fusion = CategoryGuidedFusion(config=self.config, openai_client=openai_client)
    
    def _compute_adaptive_threshold(self, failures: list[FailureInstance]) -> float:
        """Compute adaptive threshold based on exception type diversity.
        
        When exception type diversity is low (many failures, few unique types),
        failures are likely from few root causes → use looser threshold to encourage merging.
        
        Args:
            failures: List of failure instances.
            
        Returns:
            Appropriate distance threshold for clustering.
        """
        if not self.config.adaptive_threshold or len(failures) <= 1:
            return self.config.distance_threshold
        
        # Count unique exception types
        exception_types = set(f.exception_type for f in failures)
        diversity_ratio = len(exception_types) / len(failures)
        
        # Select threshold based on diversity
        if diversity_ratio < 0.1:
            # Very low diversity: likely 1-2 root causes
            # e.g., 216 failures with 1 exception type → ratio = 0.005
            threshold = self.config.threshold_low_diversity
        elif diversity_ratio < 0.3:
            # Medium diversity: moderate number of root causes
            threshold = self.config.threshold_medium_diversity
        else:
            # High diversity: many distinct exceptions
            threshold = self.config.threshold_high_diversity
        
        return threshold
    
    def cluster(
        self,
        failures: list[FailureInstance],
        s_lex: np.ndarray,
    ) -> tuple[list[Cluster], list[str], list[NormalizedFailure]]:
        """Cluster failures using category-guided fusion with two-stage approach.
        
        Enhanced clustering pipeline:
        1. Check single root cause heuristic (like BuildSheriff)
        2. Pre-cluster by exception signature (if enabled)
        3. Apply must-link constraints
        4. Hierarchical clustering on fused similarity
        
        Args:
            failures: List of failure instances.
            s_lex: Pre-computed lexical similarity matrix.
            
        Returns:
            Tuple of (clusters, noise failure IDs, normalizations).
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        
        n = len(failures)
        if n == 0:
            return [], [], []
        
        if n == 1:
            cluster = Cluster(
                cluster_id="cluster_0",
                failure_ids=[failures[0].failure_id],
                representative_failure_id=failures[0].failure_id,
            )
            norm = self.fusion._get_normalizations(failures)
            return [cluster], [], norm
        
        # === STAGE 0: Single Root Cause Heuristic (like BuildSheriff) ===
        if self.config.single_cause_enabled:
            unique_exceptions = set(f.exception_type for f in failures)
            if len(unique_exceptions) <= self.config.single_cause_exception_threshold:
                # Very few exception types → likely single root cause
                # Put all failures in one cluster
                normalizations = self.fusion._get_normalizations(failures)
                cluster = Cluster(
                    cluster_id="cluster_0",
                    failure_ids=[f.failure_id for f in failures],
                    representative_failure_id=failures[0].failure_id,
                )
                cluster.metadata = {
                    "categories": list(set(n.category for n in normalizations)),
                    "primary_category": normalizations[0].category,
                    "single_cause_heuristic": True,
                }
                return [cluster], [], normalizations
        
        # === STAGE 1: Pre-cluster by Exception Signature ===
        if self.config.use_two_stage and self.config.precluster_by_signature:
            preclusters = self._precluster_by_signature(failures)
            if len(preclusters) == 1:
                # All same signature → one cluster
                normalizations = self.fusion._get_normalizations(failures)
                cluster = Cluster(
                    cluster_id="cluster_0",
                    failure_ids=[f.failure_id for f in failures],
                    representative_failure_id=failures[0].failure_id,
                )
                return [cluster], [], normalizations
        
        # Compute fused similarity
        s_fused, normalizations = self.fusion.compute_fused_similarity(failures, s_lex)
        
        # === STAGE 2: Apply Must-Link Constraints ===
        if self.config.use_two_stage:
            s_fused = self._apply_must_link_constraints(s_fused, failures)
        
        # Convert to distance matrix
        distance_matrix = 1.0 - s_fused
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Ensure symmetry and valid values
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        
        # Perform hierarchical clustering with configurable linkage
        condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed, method=self.config.linkage_method)
        
        # Compute adaptive threshold based on exception diversity
        threshold = self._compute_adaptive_threshold(failures)
        
        # Cut tree at threshold
        labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
        
        # Convert labels to clusters
        clusters = self._labels_to_clusters(failures, labels, normalizations)
        
        # No noise in agglomerative clustering
        noise_ids = []
        
        return clusters, noise_ids, normalizations
    
    def _precluster_by_signature(
        self, failures: list[FailureInstance]
    ) -> dict[str, list[int]]:
        """Pre-cluster failures by exception signature.
        
        Signature = exception_type + normalized_message
        Failures with identical signatures are almost certainly same root cause.
        """
        import re
        
        signature_to_indices: dict[str, list[int]] = defaultdict(list)
        
        for i, f in enumerate(failures):
            # Create signature from exception type and normalized message
            exc_type = (f.exception_type or "").lower()
            msg = (f.exception_message or "").lower().strip()
            
            # Normalize message: remove variable parts
            msg = re.sub(r'0x[0-9a-f]+', '<addr>', msg)
            msg = re.sub(r'line \d+', 'line <n>', msg)
            msg = re.sub(r':\d+:', ':<n>:', msg)
            msg = re.sub(r'\d+', '<n>', msg)  # Replace all numbers
            
            # Truncate long messages
            if len(msg) > 100:
                msg = msg[:100]
            
            signature = f"{exc_type}::{msg}"
            signature_to_indices[signature].append(i)
        
        return dict(signature_to_indices)
    
    def _apply_must_link_constraints(
        self,
        similarity: np.ndarray,
        failures: list[FailureInstance],
    ) -> np.ndarray:
        """Apply must-link constraints for near-identical failures.
        
        If two failures have very high similarity on strong signals,
        set their similarity to 1.0 (forcing them into same cluster).
        """
        import re
        
        n = len(failures)
        result = similarity.copy()
        
        # Extract normalized messages
        messages = []
        for f in failures:
            msg = (f.exception_message or "").lower().strip()
            msg = re.sub(r'0x[0-9a-f]+', '<addr>', msg)
            msg = re.sub(r'line \d+', 'line <n>', msg)
            msg = re.sub(r':\d+:', ':<n>:', msg)
            messages.append(msg)
        
        # Apply must-link for identical signatures
        for i in range(n):
            for j in range(i + 1, n):
                # Check if messages are identical and non-empty
                if messages[i] and messages[j] and messages[i] == messages[j]:
                    # Also check exception types match
                    if failures[i].exception_type == failures[j].exception_type:
                        # Must-link: set similarity to 1.0
                        result[i, j] = 1.0
                        result[j, i] = 1.0
                
                # Also must-link if similarity already very high
                if result[i, j] >= self.config.must_link_threshold:
                    result[i, j] = 1.0
                    result[j, i] = 1.0
        
        return result
    
    def _labels_to_clusters(
        self,
        failures: list[FailureInstance],
        labels: np.ndarray,
        normalizations: list[NormalizedFailure],
    ) -> list[Cluster]:
        """Convert cluster labels to Cluster objects."""
        from collections import defaultdict
        
        # Group failures by label
        label_to_failures: dict[int, list[tuple[FailureInstance, NormalizedFailure]]] = defaultdict(list)
        for failure, norm, label in zip(failures, normalizations, labels):
            label_to_failures[label].append((failure, norm))
        
        clusters = []
        for label, failure_norm_pairs in sorted(label_to_failures.items()):
            failure_ids = [f.failure_id for f, _ in failure_norm_pairs]
            
            # Find representative (most common category in cluster)
            categories = [n.category for _, n in failure_norm_pairs]
            most_common_category = max(set(categories), key=categories.count)
            
            # Representative is the first failure with the most common category
            representative_id = failure_ids[0]
            for f, n in failure_norm_pairs:
                if n.category == most_common_category:
                    representative_id = f.failure_id
                    break
            
            cluster = Cluster(
                cluster_id=f"cluster_{label}",
                failure_ids=failure_ids,
                representative_failure_id=representative_id,
            )
            
            # Add category info to metadata
            unique_categories = list(set(categories))
            cluster.metadata = {
                "categories": unique_categories,
                "primary_category": most_common_category,
                "components": list(set(n.suspected_component for _, n in failure_norm_pairs)),
            }
            
            clusters.append(cluster)
        
        return clusters

