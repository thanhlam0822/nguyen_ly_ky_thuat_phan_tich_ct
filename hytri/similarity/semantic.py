"""Component B2: Semantic Similarity.

Uses LLM normalization and embeddings to compute semantic similarity
between failures. This captures meaning-based similarity that lexical
methods might miss.
"""

import numpy as np

from hytri.llm.openai_client import OpenAIClient
from hytri.models import FailureInstance, NormalizedFailure


class SemanticSimilarity:
    """Computes semantic similarity between failures using LLM.

    Process:
    1. Normalize each failure to a structured format using LLM
    2. Extract the normalized_signature from each failure
    3. Embed signatures using OpenAI embeddings
    4. Compute cosine similarity matrix

    Falls back to simple text matching when LLM is unavailable.
    """

    def __init__(
        self,
        openai_client: OpenAIClient | None = None,
        cache_normalizations: bool = True,
    ):
        """Initialize the semantic similarity computer.

        Args:
            openai_client: OpenAI client for LLM operations.
            cache_normalizations: Whether to cache normalization results.
        """
        self.client = openai_client or OpenAIClient()
        self.cache_normalizations = cache_normalizations
        self._normalization_cache: dict[str, NormalizedFailure] = {}
        self._embedding_cache: dict[str, list[float]] = {}

    def normalize_failure(self, failure: FailureInstance) -> NormalizedFailure:
        """Normalize a failure to structured format.

        Args:
            failure: The failure instance to normalize.

        Returns:
            NormalizedFailure with structured diagnostic info.
        """
        # Check cache
        if self.cache_normalizations and failure.failure_id in self._normalization_cache:
            return self._normalization_cache[failure.failure_id]

        # Call LLM for normalization
        normalized = self.client.normalize_failure(
            exception_type=failure.exception_type,
            exception_message=failure.exception_message,
            stack_trace=failure.stack_trace,
            test_name=failure.test_name,
        )

        # Cache result
        if self.cache_normalizations:
            self._normalization_cache[failure.failure_id] = normalized

        return normalized

    def batch_normalize(
        self, failures: list[FailureInstance]
    ) -> list[NormalizedFailure]:
        """Normalize a batch of failures.

        Args:
            failures: List of failure instances.

        Returns:
            List of normalized failures.
        """
        return [self.normalize_failure(f) for f in failures]

    def compute_similarity_matrix(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute pairwise semantic similarity matrix.

        Args:
            failures: List of failure instances.

        Returns:
            NxN similarity matrix where N = len(failures).
        """
        n = len(failures)
        if n == 0:
            return np.array([])

        # Normalize all failures
        normalized = self.batch_normalize(failures)

        # Extract signatures for embedding
        signatures = [n.normalized_signature for n in normalized]

        # Get embeddings
        embeddings = self._get_embeddings(signatures, [f.failure_id for f in failures])

        # Compute cosine similarity matrix
        embeddings_matrix = np.array(embeddings)

        # Normalize vectors for cosine similarity
        norms = np.linalg.norm(embeddings_matrix, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        normalized_embeddings = embeddings_matrix / norms

        # Cosine similarity via dot product of normalized vectors
        sim_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

        # Ensure values are in [0, 1] (cosine can be negative)
        sim_matrix = (sim_matrix + 1) / 2

        # Ensure diagonal is 1
        np.fill_diagonal(sim_matrix, 1.0)

        return sim_matrix

    def _get_embeddings(
        self, texts: list[str], cache_keys: list[str]
    ) -> list[list[float]]:
        """Get embeddings for texts, using cache when available.

        Args:
            texts: Text strings to embed.
            cache_keys: Cache keys for each text.

        Returns:
            List of embedding vectors.
        """
        embeddings = []
        texts_to_embed = []
        indices_to_embed = []

        # Check cache first
        for i, (text, key) in enumerate(zip(texts, cache_keys)):
            if key in self._embedding_cache:
                embeddings.append(self._embedding_cache[key])
            else:
                embeddings.append(None)  # Placeholder
                texts_to_embed.append(text)
                indices_to_embed.append(i)

        # Get embeddings for uncached texts
        if texts_to_embed:
            new_embeddings = self.client.get_embeddings(texts_to_embed)

            for idx, emb in zip(indices_to_embed, new_embeddings):
                embeddings[idx] = emb
                self._embedding_cache[cache_keys[idx]] = emb

        return embeddings  # type: ignore

    def get_normalized_signatures(
        self, failures: list[FailureInstance]
    ) -> list[str]:
        """Get normalized signatures for a list of failures.

        Useful for debugging or display.

        Args:
            failures: List of failure instances.

        Returns:
            List of normalized signature strings.
        """
        normalized = self.batch_normalize(failures)
        return [n.normalized_signature for n in normalized]

    def get_failure_categories(
        self, failures: list[FailureInstance]
    ) -> dict[str, list[str]]:
        """Group failures by their normalized category.

        Args:
            failures: List of failure instances.

        Returns:
            Dict mapping category -> list of failure_ids.
        """
        normalized = self.batch_normalize(failures)

        categories: dict[str, list[str]] = {}
        for failure, norm in zip(failures, normalized):
            if norm.category not in categories:
                categories[norm.category] = []
            categories[norm.category].append(failure.failure_id)

        return categories

    def clear_cache(self) -> None:
        """Clear normalization and embedding caches."""
        self._normalization_cache.clear()
        self._embedding_cache.clear()


class FallbackSemanticSimilarity:
    """Fallback semantic similarity when LLM is unavailable.

    Uses simple text-based similarity on exception type + message.
    """

    def compute_similarity_matrix(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute similarity using simple text matching.

        Args:
            failures: List of failure instances.

        Returns:
            NxN similarity matrix.
        """
        n = len(failures)
        if n == 0:
            return np.array([])

        # Create simple signatures
        signatures = [
            f"{f.exception_type}:{f.exception_message[:100]}"
            for f in failures
        ]

        # Compute Jaccard similarity on character trigrams
        trigram_sets = [self._get_trigrams(s) for s in signatures]

        sim_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                if not trigram_sets[i] or not trigram_sets[j]:
                    sim = 0.0
                else:
                    intersection = len(trigram_sets[i] & trigram_sets[j])
                    union = len(trigram_sets[i] | trigram_sets[j])
                    sim = intersection / union if union > 0 else 0.0

                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        np.fill_diagonal(sim_matrix, 1.0)

        return sim_matrix

    def _get_trigrams(self, text: str) -> set[str]:
        """Extract character trigrams from text."""
        text = text.lower()
        if len(text) < 3:
            return {text}
        return {text[i : i + 3] for i in range(len(text) - 2)}


