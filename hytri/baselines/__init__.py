"""Baseline methods for comparison.

Implements methods from BuildSheriff paper and prior work:
- Rebucket (Lerch & Mezini 2013)
- PWOverlap: Position-weighted stack frame overlap
- TraceSim (Jiang et al., ASE 2020) – Weighted edit-distance trace similarity
- Naive (Exception-type clustering)
- BuildSheriff-lite (without change awareness)
- BuildSheriff (full, with change-aware clustering from ICSE 2022 paper)
- GPTrace (Herter et al. 2025) – LLM embedding + HDBSCAN clustering
"""

from hytri.baselines.methods import (
    NaiveClusterer,
    RebucketClusterer,
    PWOverlapClusterer,
    TraceSimClusterer,  # backward-compat alias for PWOverlapClusterer
    TraceSimOfficialClusterer,
    BuildSheriffLiteClusterer,
    BuildSheriffClusterer,
    GPTraceClusterer,
)

__all__ = [
    "NaiveClusterer",
    "RebucketClusterer",
    "PWOverlapClusterer",
    "TraceSimClusterer",
    "TraceSimOfficialClusterer",
    "BuildSheriffLiteClusterer",
    "BuildSheriffClusterer",
    "GPTraceClusterer",
]

