"""Component C: Graph Fusion + Clustering."""

from hytri.clustering.category_guided import CategoryGuidedClusterer, CategoryGuidedConfig
from hytri.clustering.change_aware import (
    ChangeAwareClusterer,
    ChangeAwareConfig,
    ChangeAnalyzer,
    CodeChange,
)
from hytri.clustering.clusterer import FailureClusterer
from hytri.clustering.fusion import SimilarityFusion
from hytri.clustering.two_stage import TwoStageClusterer

__all__ = [
    "SimilarityFusion",
    "FailureClusterer",
    "TwoStageClusterer",
    "CategoryGuidedClusterer",
    "CategoryGuidedConfig",
    "ChangeAwareClusterer",
    "ChangeAwareConfig",
    "ChangeAnalyzer",
    "CodeChange",
]


