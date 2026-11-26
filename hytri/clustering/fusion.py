"""Component C1: Similarity Fusion.

Combines lexical and semantic similarity matrices into a unified
similarity graph for clustering.
"""

import numpy as np


class SimilarityFusion:
    """Fuses multiple similarity matrices into one.

    Supports:
    - Weighted linear combination
    - Learned/tuned weights
    - Exception type boosting
    """

    def __init__(
        self,
        alpha: float = 0.5,
        exception_type_boost: float = 0.1,
        min_similarity: float = 0.0,
    ):
        """Initialize the fusion module.

        Args:
            alpha: Weight for lexical similarity (1-alpha for semantic).
            exception_type_boost: Boost for same exception type pairs.
            min_similarity: Minimum similarity threshold (values below are set to 0).
        """
        if not 0 <= alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha
        self.exception_type_boost = exception_type_boost
        self.min_similarity = min_similarity

    def fuse(
        self,
        s_lex: np.ndarray,
        s_sem: np.ndarray,
        exception_types: list[str] | None = None,
    ) -> np.ndarray:
        """Fuse lexical and semantic similarity matrices.

        Args:
            s_lex: Lexical similarity matrix (NxN).
            s_sem: Semantic similarity matrix (NxN).
            exception_types: Optional list of exception types for boosting.

        Returns:
            Fused similarity matrix (NxN).
        """
        if s_lex.shape != s_sem.shape:
            raise ValueError(
                f"Matrix shapes must match: {s_lex.shape} vs {s_sem.shape}"
            )

        # Linear combination
        fused = self.alpha * s_lex + (1 - self.alpha) * s_sem

        # Apply exception type boost
        if exception_types is not None:
            boost_matrix = self._compute_exception_boost(exception_types)
            fused = fused + self.exception_type_boost * boost_matrix

        # Apply minimum threshold
        if self.min_similarity > 0:
            fused = np.where(fused >= self.min_similarity, fused, 0.0)

        # Clip to [0, 1]
        fused = np.clip(fused, 0.0, 1.0)

        # Ensure diagonal is 1
        np.fill_diagonal(fused, 1.0)

        return fused

    def _compute_exception_boost(self, exception_types: list[str]) -> np.ndarray:
        """Compute exception type match boost matrix."""
        n = len(exception_types)
        boost_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                if exception_types[i] == exception_types[j]:
                    boost_matrix[i, j] = 1.0
                    boost_matrix[j, i] = 1.0

        return boost_matrix

    def similarity_to_distance(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """Convert similarity matrix to distance matrix.

        Args:
            similarity_matrix: Similarity values in [0, 1].

        Returns:
            Distance matrix (1 - similarity).
        """
        return 1.0 - similarity_matrix

    def tune_alpha(
        self,
        s_lex: np.ndarray,
        s_sem: np.ndarray,
        ground_truth_labels: np.ndarray,
        alpha_range: tuple[float, float] = (0.0, 1.0),
        num_steps: int = 11,
    ) -> float:
        """Tune alpha parameter using grid search.

        Uses silhouette score on ground truth labels to find optimal alpha.

        Args:
            s_lex: Lexical similarity matrix.
            s_sem: Semantic similarity matrix.
            ground_truth_labels: True cluster labels.
            alpha_range: Range of alpha values to try.
            num_steps: Number of alpha values to evaluate.

        Returns:
            Optimal alpha value.
        """
        from sklearn.metrics import silhouette_score

        alphas = np.linspace(alpha_range[0], alpha_range[1], num_steps)
        best_alpha = self.alpha
        best_score = -1.0

        for alpha in alphas:
            self.alpha = alpha
            fused = self.fuse(s_lex, s_sem)
            distance = self.similarity_to_distance(fused)

            try:
                score = silhouette_score(distance, ground_truth_labels, metric="precomputed")
                if score > best_score:
                    best_score = score
                    best_alpha = alpha
            except ValueError:
                # Skip if silhouette can't be computed
                continue

        self.alpha = best_alpha
        return best_alpha


class AdaptiveFusion(SimilarityFusion):
    """Adaptive fusion that learns weights from data.

    Can adjust weights based on:
    - Confidence of each similarity source
    - Variance in similarity distributions
    """

    def __init__(
        self,
        initial_alpha: float = 0.5,
        adapt_by_variance: bool = True,
    ):
        """Initialize adaptive fusion.

        Args:
            initial_alpha: Starting alpha value.
            adapt_by_variance: Whether to adapt weights by variance.
        """
        super().__init__(alpha=initial_alpha)
        self.adapt_by_variance = adapt_by_variance

    def fuse(
        self,
        s_lex: np.ndarray,
        s_sem: np.ndarray,
        exception_types: list[str] | None = None,
    ) -> np.ndarray:
        """Fuse with adaptive weights.

        Weights are adjusted based on the variance of each similarity matrix.
        Higher variance sources get more weight (more discriminative).
        """
        if self.adapt_by_variance and s_lex.size > 0:
            # Compute variance of off-diagonal elements
            n = s_lex.shape[0]
            mask = ~np.eye(n, dtype=bool)

            var_lex = np.var(s_lex[mask]) if mask.sum() > 0 else 0.0
            var_sem = np.var(s_sem[mask]) if mask.sum() > 0 else 0.0

            total_var = var_lex + var_sem
            if total_var > 0:
                self.alpha = var_lex / total_var
            else:
                self.alpha = 0.5

        return super().fuse(s_lex, s_sem, exception_types)


