"""Component B1: Lexical/Structural Similarity.

Computes similarity between failures using:
- BM25 over signature tokens
- Jaccard similarity on (file, func) pairs
- LCS similarity on frame sequences
"""

import numpy as np
from rank_bm25 import BM25Okapi

from hytri.models import FailureInstance, StackFrame


class LexicalSimilarity:
    """Computes lexical and structural similarity between failures.

    Combines multiple signals:
    - BM25 text similarity on signature tokens
    - Jaccard overlap on stack frame (file, func) pairs
    - LCS similarity on frame sequences

    The final similarity is a weighted combination of these signals.
    """

    def __init__(
        self,
        bm25_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        lcs_weight: float = 0.3,
        same_exception_boost: float = 0.1,
    ):
        """Initialize the similarity computer.

        Args:
            bm25_weight: Weight for BM25 text similarity.
            jaccard_weight: Weight for Jaccard frame overlap.
            lcs_weight: Weight for LCS frame sequence similarity.
            same_exception_boost: Boost to add when exception types match.
        """
        self.bm25_weight = bm25_weight
        self.jaccard_weight = jaccard_weight
        self.lcs_weight = lcs_weight
        self.same_exception_boost = same_exception_boost

        # Normalize weights
        total = bm25_weight + jaccard_weight + lcs_weight
        self.bm25_weight /= total
        self.jaccard_weight /= total
        self.lcs_weight /= total

    def compute_similarity_matrix(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute pairwise similarity matrix for all failures.

        Args:
            failures: List of failure instances with signature_tokens populated.

        Returns:
            NxN similarity matrix where N = len(failures).
        """
        n = len(failures)
        if n == 0:
            return np.array([])

        # Compute individual similarity matrices
        bm25_sim = self._compute_bm25_similarity(failures)
        jaccard_sim = self._compute_jaccard_similarity(failures)
        lcs_sim = self._compute_lcs_similarity(failures)
        exception_boost = self._compute_exception_boost(failures)

        # Combine with weights
        combined = (
            self.bm25_weight * bm25_sim
            + self.jaccard_weight * jaccard_sim
            + self.lcs_weight * lcs_sim
            + self.same_exception_boost * exception_boost
        )

        # Clip to [0, 1]
        combined = np.clip(combined, 0.0, 1.0)

        # Ensure diagonal is 1
        np.fill_diagonal(combined, 1.0)

        return combined

    def _compute_bm25_similarity(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute BM25 similarity matrix over signature tokens."""
        n = len(failures)
        if n == 0:
            return np.array([])

        # Get tokenized corpus
        corpus = [f.signature_tokens if f.signature_tokens else ["<EMPTY>"] for f in failures]

        # Build BM25 index
        bm25 = BM25Okapi(corpus)

        # Compute pairwise similarities
        sim_matrix = np.zeros((n, n))

        for i, tokens in enumerate(corpus):
            scores = bm25.get_scores(tokens)
            # Normalize scores to [0, 1]
            max_score = max(scores) if max(scores) > 0 else 1.0
            sim_matrix[i] = scores / max_score

        # Make symmetric (average of both directions)
        sim_matrix = (sim_matrix + sim_matrix.T) / 2

        return sim_matrix

    def _compute_jaccard_similarity(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute Jaccard similarity on (file, func) pairs."""
        n = len(failures)
        if n == 0:
            return np.array([])

        # Extract frame sets
        frame_sets: list[set[tuple[str, str]]] = []
        for f in failures:
            frame_set = {(frame.file, frame.func) for frame in f.frames}
            frame_sets.append(frame_set)

        # Compute pairwise Jaccard
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = self._jaccard(frame_sets[i], frame_sets[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _jaccard(self, set1: set, set2: set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _compute_lcs_similarity(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute LCS similarity on frame sequences."""
        n = len(failures)
        if n == 0:
            return np.array([])

        # Extract frame sequences as strings
        frame_seqs: list[list[str]] = []
        for f in failures:
            seq = [f"{frame.file}:{frame.func}" for frame in f.frames]
            frame_seqs.append(seq)

        # Compute pairwise LCS similarity
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                sim = self._lcs_similarity(frame_seqs[i], frame_seqs[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _lcs_similarity(self, seq1: list[str], seq2: list[str]) -> float:
        """Compute LCS-based similarity between two sequences."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        lcs_len = self._lcs_length(seq1, seq2)
        max_len = max(len(seq1), len(seq2))
        return lcs_len / max_len if max_len > 0 else 0.0

    def _lcs_length(self, seq1: list[str], seq2: list[str]) -> int:
        """Compute length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    def _compute_exception_boost(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute exception type match boost matrix."""
        n = len(failures)
        if n == 0:
            return np.array([])

        exception_types = [f.exception_type for f in failures]

        boost_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if exception_types[i] == exception_types[j]:
                    boost_matrix[i, j] = 1.0
                    boost_matrix[j, i] = 1.0

        return boost_matrix

    def get_top_similar(
        self,
        query_failure: FailureInstance,
        candidates: list[FailureInstance],
        top_k: int = 5,
    ) -> list[tuple[FailureInstance, float]]:
        """Find top-k most similar failures to a query.

        Args:
            query_failure: The query failure.
            candidates: List of candidate failures to compare against.
            top_k: Number of top results to return.

        Returns:
            List of (failure, similarity_score) tuples.
        """
        all_failures = [query_failure] + candidates
        sim_matrix = self.compute_similarity_matrix(all_failures)

        # Get similarities to query (first row, excluding self)
        query_similarities = sim_matrix[0, 1:]

        # Get top-k indices
        top_indices = np.argsort(query_similarities)[::-1][:top_k]

        results = [
            (candidates[i], query_similarities[i])
            for i in top_indices
        ]

        return results


