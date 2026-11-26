"""Adaptive Threshold Selection.

Implements intelligent threshold selection based on:
1. Failure type (exception vs assertion) - from BuildSheriff
2. Signal agreement (high agreement = tighter threshold)
3. Build characteristics (many failures = tighter threshold)

BuildSheriff paper thresholds (ICSE 2022):
- dt_st = 0.57: Stack trace distance threshold
- dt_msg = 0.73: Message distance threshold
- dt_test = 0.50: Test code distance threshold

These are DISTANCE thresholds (lower = more similar = same cluster).
"""

from dataclasses import dataclass

import numpy as np

from hytri.ensemble.signals import SimilaritySignals


@dataclass
class ThresholdConfig:
    """Configuration for adaptive threshold selection."""
    
    # BuildSheriff paper thresholds (distance, not similarity)
    dt_stack: float = 0.57  # For exception failures with stack traces
    dt_message: float = 0.73  # For exception failures without good stacks
    dt_test: float = 0.50  # For assertion failures
    
    # Adjustment factors
    agreement_factor: float = 0.1  # How much signal agreement affects threshold
    size_factor: float = 0.05  # How much failure count affects threshold
    
    # Bounds
    min_threshold: float = 0.2
    max_threshold: float = 0.8


class AdaptiveThreshold:
    """Select clustering threshold based on multiple factors.
    
    The threshold determines when two failures are "close enough"
    to be in the same cluster. Lower threshold = stricter = more clusters.
    
    Adapts based on:
    1. Failure type: Assertions need different thresholds than exceptions
    2. Signal agreement: When signals agree, we can be more aggressive
    3. Failure count: More failures = tighter threshold to avoid mega-clusters
    """
    
    def __init__(self, config: ThresholdConfig | None = None):
        """Initialize adaptive threshold selector.
        
        Args:
            config: Threshold configuration.
        """
        self.config = config or ThresholdConfig()
    
    def select_threshold(
        self,
        signals: SimilaritySignals,
        failure_type: str = "exception",
        n_failures: int | None = None,
    ) -> float:
        """Select optimal threshold for clustering.
        
        Args:
            signals: Computed similarity signals.
            failure_type: "exception" or "assertion"
            n_failures: Number of failures (for size adjustment)
            
        Returns:
            Distance threshold for clustering.
        """
        # Start with base threshold for failure type
        base_threshold = self._get_base_threshold(failure_type, signals)
        
        # Adjust for signal agreement
        agreement_adjustment = self._compute_agreement_adjustment(signals)
        
        # Adjust for number of failures
        size_adjustment = self._compute_size_adjustment(n_failures or signals.n_failures)
        
        # Combine adjustments
        threshold = base_threshold + agreement_adjustment + size_adjustment
        
        # Clamp to valid range
        threshold = max(self.config.min_threshold, 
                       min(self.config.max_threshold, threshold))
        
        return threshold
    
    def _get_base_threshold(
        self,
        failure_type: str,
        signals: SimilaritySignals,
    ) -> float:
        """Get base threshold for failure type.
        
        Args:
            failure_type: "exception", "assertion", or "mixed"
            signals: Signals to check which are available
            
        Returns:
            Base distance threshold.
        """
        if failure_type == "assertion":
            return self.config.dt_test
        
        # For "mixed" or "exception", choose based on available signals.
        # Per-pair type-aware weighting is handled at the aggregator level,
        # so the threshold here is driven by signal confidence.
        pdm_conf = signals.get_confidence('pdm')
        overlap_conf = signals.get_confidence('overlap')
        
        if pdm_conf >= 0.3 or overlap_conf >= 0.3:
            return self.config.dt_stack
        else:
            return self.config.dt_message
    
    def _compute_agreement_adjustment(
        self,
        signals: SimilaritySignals,
    ) -> float:
        """Compute threshold adjustment based on signal agreement.
        
        When signals agree strongly, we can use a tighter threshold.
        When they disagree, we should be more conservative (looser threshold).
        
        Args:
            signals: Computed similarity signals.
            
        Returns:
            Adjustment to add to base threshold.
        """
        all_signals = signals.get_all_signals()
        if len(all_signals) < 2:
            return 0.0
        
        # Stack all signal matrices
        matrices = list(all_signals.values())
        signal_stack = np.stack(matrices, axis=0)
        
        # Compute mean correlation between signals
        correlations = []
        for i in range(len(matrices)):
            for j in range(i + 1, len(matrices)):
                flat_i = matrices[i].flatten()
                flat_j = matrices[j].flatten()
                
                if np.std(flat_i) > 0 and np.std(flat_j) > 0:
                    corr = np.corrcoef(flat_i, flat_j)[0, 1]
                    correlations.append(corr)
        
        if not correlations:
            return 0.0
        
        mean_agreement = np.mean(correlations)
        
        # High agreement (>0.5) = can use tighter threshold (negative adjustment)
        # Low agreement (<0.3) = use looser threshold (positive adjustment)
        if mean_agreement > 0.5:
            return -self.config.agreement_factor * (mean_agreement - 0.5)
        elif mean_agreement < 0.3:
            return self.config.agreement_factor * (0.3 - mean_agreement)
        else:
            return 0.0
    
    def _compute_size_adjustment(self, n_failures: int) -> float:
        """Compute threshold adjustment based on number of failures.
        
        More failures = tighter threshold to avoid mega-clusters.
        
        Args:
            n_failures: Number of failures.
            
        Returns:
            Adjustment to add to base threshold.
        """
        if n_failures <= 5:
            return 0.0  # Few failures, no adjustment
        elif n_failures <= 15:
            # Slight tightening
            return -self.config.size_factor * (n_failures - 5) / 10
        else:
            # More significant tightening for many failures
            return -self.config.size_factor * (1.0 + np.log10(n_failures / 15))
    
    def get_thresholds_for_stratified(
        self,
        exception_signals: SimilaritySignals | None,
        assertion_signals: SimilaritySignals | None,
    ) -> dict[str, float]:
        """Get thresholds for stratified clustering.
        
        Args:
            exception_signals: Signals for exception failures.
            assertion_signals: Signals for assertion failures.
            
        Returns:
            Dict with 'exception' and 'assertion' thresholds.
        """
        thresholds = {}
        
        if exception_signals and exception_signals.n_failures > 0:
            thresholds['exception'] = self.select_threshold(
                exception_signals, 
                failure_type="exception"
            )
        
        if assertion_signals and assertion_signals.n_failures > 0:
            thresholds['assertion'] = self.select_threshold(
                assertion_signals,
                failure_type="assertion"
            )
        
        return thresholds


class SignalAwareThreshold:
    """More sophisticated threshold that considers individual signals.
    
    Uses different thresholds when specific high-confidence signals
    provide strong evidence for clustering.
    """
    
    # Per-signal thresholds when that signal is dominant
    SIGNAL_THRESHOLDS = {
        'overlap': 0.40,  # Direct file overlap is very reliable
        'hypothesis': 0.45,  # Same hypothesis is strong evidence
        'pdm': 0.57,  # BuildSheriff stack threshold
        'semantic': 0.55,  # LLM embeddings
        'message': 0.73,  # BuildSheriff message threshold
        'test_name': 0.50,  # BuildSheriff test threshold
    }
    
    def __init__(self, config: ThresholdConfig | None = None):
        self.config = config or ThresholdConfig()
        self.base_selector = AdaptiveThreshold(config)
    
    def select_threshold(
        self,
        signals: SimilaritySignals,
        failure_type: str = "exception",
    ) -> float:
        """Select threshold based on dominant signal.
        
        If one signal has notably higher confidence, use its threshold.
        Otherwise, use the base adaptive threshold.
        """
        # Find dominant signal (if any)
        max_conf = 0.0
        dominant_signal = None
        
        for name in self.SIGNAL_THRESHOLDS.keys():
            conf = signals.get_confidence(name)
            if conf > max_conf and conf > 0.5:  # Must be >0.5 to be dominant
                max_conf = conf
                dominant_signal = name
        
        # If we have a clear dominant signal, use its threshold
        if dominant_signal and max_conf > 0.7:
            return self.SIGNAL_THRESHOLDS[dominant_signal]
        
        # Otherwise, use base adaptive threshold
        return self.base_selector.select_threshold(signals, failure_type)
    
    def get_per_pair_thresholds(
        self,
        signals: SimilaritySignals,
    ) -> np.ndarray:
        """Compute per-pair thresholds based on signal confidence.
        
        This allows different pairs to have different thresholds
        based on which signals provide evidence for that pair.
        
        Returns:
            NxN matrix of thresholds for each pair.
        """
        n = signals.n_failures
        thresholds = np.full((n, n), self.config.dt_stack)
        
        all_signals = signals.get_all_signals()
        
        for name, matrix in all_signals.items():
            signal_threshold = self.SIGNAL_THRESHOLDS.get(name, self.config.dt_stack)
            confidence = signals.get_confidence(name)
            
            # For pairs where this signal shows high similarity,
            # use this signal's threshold (weighted by confidence)
            high_sim_mask = matrix > 0.7
            
            # Blend with current threshold
            new_thresholds = np.where(
                high_sim_mask,
                thresholds * (1 - confidence) + signal_threshold * confidence,
                thresholds
            )
            thresholds = new_thresholds
        
        return thresholds
