"""HyTri - Hybrid Failure Triage System.

A system for clustering test failures by root cause, prioritizing fixes,
and generating actionable explanations using traditional methods combined with LLMs.
"""

from hytri.models import (
    Build,
    Cluster,
    FailureInstance,
    GroundTruth,
    NormalizedFailure,
    StackFrame,
    TriageResult,
)
from hytri.pipeline import HyTriPipeline, HyTriPipelineBuilder

__version__ = "0.1.0"

__all__ = [
    "HyTriPipeline",
    "HyTriPipelineBuilder",
    "FailureInstance",
    "StackFrame",
    "GroundTruth",
    "NormalizedFailure",
    "Cluster",
    "Build",
    "TriageResult",
]


