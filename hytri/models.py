"""Core data models for HyTri.

This module defines all the Pydantic models used throughout the system,
matching the JSONL schema for failure instances, builds, and clustering results.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class StackFrame(BaseModel):
    """A single frame in a stack trace."""

    file: str = Field(description="File path where the frame originated")
    func: str = Field(description="Function name")
    line: int = Field(description="Line number in the file")

    def __hash__(self) -> int:
        return hash((self.file, self.func, self.line))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, StackFrame):
            return False
        return self.file == other.file and self.func == other.func and self.line == other.line


class GroundTruth(BaseModel):
    """Ground truth information for a failure instance."""

    root_cause_id: str = Field(description="Identifier of the root cause (e.g., bug ID)")
    changed_files: list[str] = Field(
        default_factory=list, description="Files changed to fix the bug"
    )
    fault_location_hint: list[str] = Field(
        default_factory=list, description="Hints about fault location (e.g., 'module.py:44')"
    )


class FailureInstance(BaseModel):
    """A single test failure instance.

    This is the primary input to the HyTri system, representing one failing test execution.
    """

    build_id: str = Field(description="Identifier of the build/run")
    failure_id: str = Field(description="Unique identifier for this failure")
    test_name: str = Field(description="Full test name (pytest node id)")
    exception_type: str = Field(description="Type of exception raised")
    exception_message: str = Field(description="Exception message text")
    stack_trace: str = Field(description="Full stack trace as raw text")
    frames: list[StackFrame] = Field(
        default_factory=list, description="Parsed stack frames"
    )
    stderr: str = Field(default="", description="Captured stderr output")
    stdout: str = Field(default="", description="Captured stdout output")
    timestamp: datetime | None = Field(default=None, description="When the failure occurred")
    project: str = Field(default="", description="Project name")
    bug_ids: list[str] = Field(
        default_factory=list, description="Bug IDs associated with this build"
    )
    ground_truth: GroundTruth | None = Field(
        default=None, description="Ground truth for evaluation"
    )

    # Computed/cached fields (set during parsing)
    normalized_message: str = Field(
        default="", description="Message with numbers/IDs replaced by placeholders"
    )
    signature_tokens: list[str] = Field(
        default_factory=list, description="Tokens for signature-based matching"
    )
    log_templates: list[str] = Field(
        default_factory=list, description="Drain-style log templates from stderr/stdout"
    )


class NormalizedFailure(BaseModel):
    """LLM-generated normalized representation of a failure.

    This is the output of the semantic normalization step (Component B2).
    """

    category: str = Field(
        description="Failure category (e.g., 'Dependency/API mismatch', 'Assertion failure')"
    )
    suspected_component: str = Field(
        description="Module or component suspected to be the source"
    )
    likely_root_cause_hypothesis: str = Field(
        description="Hypothesis about the root cause"
    )
    key_evidence: list[str] = Field(
        description="Key evidence supporting the hypothesis"
    )
    normalized_signature: str = Field(
        description="Canonical signature for similarity matching"
    )


class ClusterExplanation(BaseModel):
    """LLM-generated explanation for a cluster."""

    title: str = Field(description="Short, actionable cluster title")
    summary: list[str] = Field(description="3-5 bullet points with evidence-backed explanation")
    groundedness_score: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Score indicating how well grounded the explanation is"
    )
    verified: bool = Field(default=False, description="Whether the explanation passed verification")


class Cluster(BaseModel):
    """A cluster of related failures.

    This is the primary output of the clustering step (Component C).
    """

    cluster_id: str = Field(description="Unique identifier for this cluster")
    failure_ids: list[str] = Field(description="IDs of failures in this cluster")
    representative_failure_id: str = Field(
        default="", description="ID of the representative failure"
    )
    representative_ids: list[str] = Field(
        default_factory=list, description="IDs of representative failures (for explanation)"
    )
    explanation: ClusterExplanation | None = Field(
        default=None, description="LLM-generated explanation"
    )
    severity_score: float = Field(default=0.0, description="Priority/severity score")

    # Cluster statistics
    exception_types: list[str] = Field(
        default_factory=list, description="Unique exception types in this cluster"
    )
    top_frames: list[str] = Field(
        default_factory=list, description="Most common frames (file:func format)"
    )
    num_unique_tests: int = Field(default=0, description="Number of unique tests in cluster")
    num_unique_modules: int = Field(default=0, description="Number of unique modules touched")
    
    # Extensible metadata for clustering methods
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata from clustering (e.g., categories)"
    )

    @property
    def title(self) -> str | None:
        """Convenience accessor for explanation title."""
        return self.explanation.title if self.explanation else None

    @property
    def summary(self) -> list[str] | None:
        """Convenience accessor for explanation summary."""
        return self.explanation.summary if self.explanation else None


class Build(BaseModel):
    """Metadata for a build/test run.

    This provides context about the environment and scope of a test run.
    """

    build_id: str = Field(description="Unique identifier for this build")
    project: str = Field(description="Project name")
    bug_ids: list[str] = Field(default_factory=list, description="Bug IDs in this build")
    num_failures: int = Field(default=0, description="Number of failures in this build")
    num_tests_total: int = Field(default=0, description="Total number of tests run")
    env: dict[str, str] = Field(
        default_factory=dict, description="Environment info (python version, os, etc.)"
    )
    timestamp: datetime | None = Field(default=None, description="When the build ran")
    notes: str = Field(default="", description="Additional notes about the build")


class TriageResult(BaseModel):
    """Complete result of a triage operation.

    This is the main output of the HyTriPipeline.
    """

    build_id: str = Field(description="Build that was triaged")
    clusters: list[Cluster] = Field(description="Identified clusters")
    priority_queue: list[str] = Field(
        description="Cluster IDs in priority order (highest first)"
    )
    num_failures_total: int = Field(description="Total number of failures processed")
    num_noise_failures: int = Field(
        default=0, description="Failures not assigned to any cluster"
    )
    noise_failure_ids: list[str] = Field(
        default_factory=list, description="IDs of noise failures"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata about the triage"
    )


class EvaluationResult(BaseModel):
    """Results from evaluating triage quality."""

    # Clustering metrics
    ari: float = Field(description="Adjusted Rand Index")
    nmi: float = Field(description="Normalized Mutual Information")
    v_measure: float = Field(description="V-measure score")
    cluster_count_error: int = Field(description="Difference between predicted and true cluster count")
    purity: float = Field(description="Purity score")

    # Prioritization metrics
    ndcg_at_1: float = Field(default=0.0, description="nDCG@1")
    ndcg_at_3: float = Field(default=0.0, description="nDCG@3")
    ndcg_at_5: float = Field(default=0.0, description="nDCG@5")
    top_1_coverage: float = Field(default=0.0, description="Failure coverage with top 1 cluster")
    top_3_coverage: float = Field(default=0.0, description="Failure coverage with top 3 clusters")
    top_5_coverage: float = Field(default=0.0, description="Failure coverage with top 5 clusters")

    # Explanation metrics
    avg_groundedness: float = Field(default=0.0, description="Average groundedness score")
    explanation_coverage: float = Field(
        default=0.0, description="Fraction of clusters with verified explanations"
    )


