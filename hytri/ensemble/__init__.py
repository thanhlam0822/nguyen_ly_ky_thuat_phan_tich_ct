"""HyTri-E: Ensemble Failure Triage Architecture.

This module implements an ensemble approach that combines:
- BuildSheriff's domain-specific decision rules (complexity gate, failure stratification)
- LLM-powered semantic understanding (hypothesis generation, embeddings)
- Aggregation-based similarity (group-level comparison, confidence-weighted voting)

Key components:
- ComplexityGate: Short-circuits clustering when few changes exist
- FailureStratifier: Splits exception vs assertion failures
- MultiSignalComputer: Computes all similarity signals
- HypothesisGenerator: LLM-based root cause hypothesis
- AggregationEnsemble: Group-level similarity aggregation
- AdaptiveThreshold: Signal-aware threshold selection
- HyTriEnsemble: Main orchestrating pipeline
"""

from hytri.ensemble.pipeline import HyTriEnsemble
from hytri.ensemble.complexity_gate import ComplexityGate
from hytri.ensemble.stratifier import FailureStratifier, StratifiedFailures
from hytri.ensemble.signals import SimilaritySignals, MultiSignalComputer
from hytri.ensemble.hypothesis import HypothesisGenerator, Hypothesis
from hytri.ensemble.aggregator import AggregationEnsemble
from hytri.ensemble.threshold import AdaptiveThreshold
from hytri.ensemble.pdm import PDMSimilarity

__all__ = [
    "HyTriEnsemble",
    "ComplexityGate",
    "FailureStratifier",
    "StratifiedFailures",
    "SimilaritySignals",
    "MultiSignalComputer",
    "HypothesisGenerator",
    "Hypothesis",
    "AggregationEnsemble",
    "AdaptiveThreshold",
    "PDMSimilarity",
]
