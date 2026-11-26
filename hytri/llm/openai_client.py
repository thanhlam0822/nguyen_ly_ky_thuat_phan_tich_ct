"""OpenAI API Client Wrapper.

Provides structured output support for failure normalization and explanation generation.
"""

import json
import os
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel

from hytri.models import ClusterExplanation, NormalizedFailure

T = TypeVar("T", bound=BaseModel)


class OpenAIClient:
    """Wrapper for OpenAI API with structured output support.

    Handles:
    - Failure normalization (structured JSON output)
    - Cluster explanation generation
    - Embedding generation for semantic similarity
    """

    # Default models
    DEFAULT_CHAT_MODEL = "gpt-4o"
    DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

    # Prompts
    NORMALIZATION_SYSTEM_PROMPT = """You are an expert software engineer analyzing test failures.
Your task is to normalize a test failure into a structured format for similarity matching.

Given a test failure with its exception type, message, and stack trace, produce a JSON object with:
- category: A high-level category (one of: "Dependency/API mismatch", "Assertion failure", "Timeout", "Type error", "Import error", "Connection error", "Resource exhaustion", "Null/None reference", "Index/Key error", "Permission error", "Configuration error", "Other")
- suspected_component: The module or component most likely responsible (e.g., "pandas.io.parsers", "requests.sessions", "project.auth")
- likely_root_cause_hypothesis: A concise hypothesis about what went wrong
- key_evidence: List of 2-4 key pieces of evidence (exception type, specific frame, message pattern)
- normalized_signature: A canonical signature for matching similar failures (format: "ExceptionType@module.function -> description")

Be concise and focus on the essential diagnostic information."""

    EXPLANATION_SYSTEM_PROMPT = """You are an expert software engineer explaining test failure clusters to help developers prioritize fixes.

Given a set of related test failures from the same cluster, generate:
- title: A short, actionable title (max 60 chars) that describes the root cause (e.g., "Database Connection Pool Exhausted", "Missing API Authentication Header")
- summary: A list of 3-5 bullet points explaining:
  1. What is failing and why
  2. Key evidence from the failures
  3. Which components/modules are affected
  4. Suggested investigation steps

Ground your explanation ONLY in the evidence provided. Do not speculate about causes not supported by the stack traces and error messages."""

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str = DEFAULT_CHAT_MODEL,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        max_retries: int = 3,
    ):
        """Initialize the OpenAI client.

        Args:
            api_key: OpenAI API key. Falls back to OPENAI_API_KEY env var.
            chat_model: Model to use for chat completions.
            embedding_model: Model to use for embeddings.
            max_retries: Number of retries for failed requests.
        """
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.chat_model = chat_model
        self.embedding_model = embedding_model
        self.max_retries = max_retries

        if self.api_key:
            self.client = OpenAI(api_key=self.api_key, max_retries=max_retries)
        else:
            self.client = None

    def is_available(self) -> bool:
        """Check if the client is properly configured."""
        return self.client is not None and bool(self.api_key)

    def normalize_failure(
        self,
        exception_type: str,
        exception_message: str,
        stack_trace: str,
        test_name: str = "",
    ) -> NormalizedFailure:
        """Normalize a failure into a structured format.

        Args:
            exception_type: Type of exception.
            exception_message: Exception message.
            stack_trace: Full stack trace.
            test_name: Name of the failing test.

        Returns:
            NormalizedFailure with structured diagnostic information.
        """
        if not self.is_available():
            return self._fallback_normalization(exception_type, exception_message)

        user_prompt = f"""Analyze this test failure:

Test: {test_name}
Exception Type: {exception_type}
Message: {exception_message}

Stack Trace:
{stack_trace}

Produce the normalized JSON output."""

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.NORMALIZATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )

        content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(content)
            return NormalizedFailure(**data)
        except (json.JSONDecodeError, ValueError):
            return self._fallback_normalization(exception_type, exception_message)

    def _fallback_normalization(
        self, exception_type: str, exception_message: str
    ) -> NormalizedFailure:
        """Fallback normalization when LLM is unavailable."""
        return NormalizedFailure(
            category="Other",
            suspected_component="unknown",
            likely_root_cause_hypothesis=f"{exception_type}: {exception_message[:100]}",
            key_evidence=[f"exception: {exception_type}"],
            normalized_signature=f"{exception_type}@unknown -> {exception_message[:50]}",
        )

    def generate_cluster_explanation(
        self,
        failures_summary: str,
    ) -> ClusterExplanation:
        """Generate an explanation for a cluster of failures.

        Args:
            failures_summary: Text summary of representative failures in the cluster.

        Returns:
            ClusterExplanation with title and summary.
        """
        if not self.is_available():
            return self._fallback_explanation()

        user_prompt = f"""Analyze these related test failures and generate an explanation:

{failures_summary}

Produce a JSON object with:
- "title": A short, actionable title (max 60 chars)
- "summary": A list of 3-5 bullet point strings explaining the failures"""

        response = self.client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {"role": "system", "content": self.EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
        )

        content = response.choices[0].message.content or "{}"

        try:
            data = json.loads(content)
            return ClusterExplanation(
                title=data.get("title", "Unknown Failure Cluster"),
                summary=data.get("summary", ["No summary available"]),
            )
        except (json.JSONDecodeError, ValueError):
            return self._fallback_explanation()

    def _fallback_explanation(self) -> ClusterExplanation:
        """Fallback explanation when LLM is unavailable."""
        return ClusterExplanation(
            title="Failure Cluster",
            summary=["LLM unavailable - manual analysis required"],
        )

    def get_embeddings(self, texts: list[str], batch_size: int = 2000) -> list[list[float]]:
        """Get embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.
            batch_size: Maximum texts per API call (OpenAI limit is ~2048).

        Returns:
            List of embedding vectors.
        """
        if not self.is_available() or not texts:
            # Return zero vectors as fallback
            return [[0.0] * 1536 for _ in texts]

        # Sanitize texts: replace empty/whitespace-only with placeholder
        sanitized_texts = []
        for t in texts:
            if not t or not t.strip():
                sanitized_texts.append("empty_failure_signature")
            else:
                # Truncate very long texts (embedding model has token limits)
                sanitized_texts.append(t[:8000])

        # Process in batches with enhanced retry logic
        import time
        all_embeddings = []
        total_batches = (len(sanitized_texts) + batch_size - 1) // batch_size
        
        for i in range(0, len(sanitized_texts), batch_size):
            batch = sanitized_texts[i:i + batch_size]
            batch_num = i // batch_size + 1
            
            # Enhanced retry with exponential backoff
            max_retries = 5
            base_wait = 2  # Start with 2 seconds
            
            for attempt in range(max_retries):
                try:
                    # Small delay between batches to avoid rate limiting
                    if i > 0 and attempt == 0:
                        time.sleep(0.5)
                    
                    response = self.client.embeddings.create(
                        model=self.embedding_model,
                        input=batch,
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    print(f"  Embedded batch {batch_num}/{total_batches} ({len(batch)} texts)")
                    break
                except Exception as e:
                    error_str = str(e).lower()
                    is_rate_limit = "rate" in error_str or "429" in error_str
                    is_connection = "connection" in error_str or "timeout" in error_str
                    
                    if attempt < max_retries - 1:
                        # Longer wait for rate limits, shorter for connection errors
                        if is_rate_limit:
                            wait_time = base_wait * (2 ** attempt) + 5  # 7, 9, 13, 21 seconds
                        else:
                            wait_time = base_wait * (2 ** attempt)  # 2, 4, 8, 16 seconds
                        
                        error_type = "Rate limit" if is_rate_limit else "Connection" if is_connection else "Error"
                        print(f"  Batch {batch_num} attempt {attempt + 1}/{max_retries} - {error_type}: {e}")
                        print(f"    Retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        # Final attempt failed, use zero vectors
                        print(f"Warning: Embedding batch {batch_num} failed after {max_retries} attempts: {e}")
                        print(f"  Using zero vectors for {len(batch)} texts (will degrade clustering quality)")
                        all_embeddings.extend([[0.0] * 1536 for _ in batch])

        return all_embeddings

    def get_embedding(self, text: str) -> list[float]:
        """Get embedding for a single text.

        Args:
            text: Text string to embed.

        Returns:
            Embedding vector.
        """
        embeddings = self.get_embeddings([text])
        return embeddings[0] if embeddings else [0.0] * 1536

    def batch_normalize_failures(
        self,
        failures: list[tuple[str, str, str, str]],
    ) -> list[NormalizedFailure]:
        """Normalize multiple failures.

        Args:
            failures: List of (exception_type, exception_message, stack_trace, test_name) tuples.

        Returns:
            List of NormalizedFailure objects.
        """
        return [
            self.normalize_failure(exc_type, exc_msg, trace, test_name)
            for exc_type, exc_msg, trace, test_name in failures
        ]


