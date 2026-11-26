"""Supervised Similarity Baseline.

Uses the same 7 signals as HyTri-E but combines them via a trained
supervised model instead of hand-tuned weights. The model predicts
P(same_cluster | 7 signals) for each failure pair, producing a
similarity matrix that feeds into hierarchical clustering.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage

from hytri.baselines.methods import BaselineClusterer
from hytri.ensemble.hypothesis import HypothesisGenerator
from hytri.ensemble.signals import MultiSignalComputer
from hytri.llm.openai_client import OpenAIClient
from hytri.models import Cluster, FailureInstance

SIGNAL_NAMES = ["pdm", "overlap", "semantic", "message", "hypothesis", "test_name", "diff_line"]


class SupervisedClusterer(BaselineClusterer):
    """Cluster failures using a trained model over HyTri's 7 signals.

    At inference time:
      1. Compute the 7 signal matrices (same as HyTri-E).
      2. For each pair (i,j), feed the 7 values to the trained model.
      3. The model outputs P(same_cluster), used as the similarity score.
      4. Hierarchical clustering (average linkage) with a tuned threshold.
    """

    def __init__(
        self,
        model_path: str | Path,
        openai_client: OpenAIClient | None = None,
        distance_threshold: float | None = None,
    ):
        """
        Args:
            model_path: Path to the saved model artifact (.pkl).
            openai_client: OpenAI client for semantic + hypothesis signals.
            distance_threshold: Override for the clustering threshold.
                If None, uses the threshold saved in the model artifact.
        """
        with open(model_path, "rb") as f:
            artifact = pickle.load(f)

        self._model = artifact["model"]
        self._model_name = artifact["model_name"]
        self._distance_threshold = distance_threshold or artifact["distance_threshold"]

        self._openai_client = openai_client
        self._signal_computer: MultiSignalComputer | None = None
        self._hypothesis_generator: HypothesisGenerator | None = None

        # Change info — set before calling cluster()
        self._changed_files: list[str] = []
        self._file_diffs: dict[str, str] = {}

    def _ensure_components(self) -> None:
        if self._signal_computer is None:
            client = self._openai_client or OpenAIClient()
            self._signal_computer = MultiSignalComputer(openai_client=client, use_semantic=True)
            self._hypothesis_generator = HypothesisGenerator(openai_client=client)

    def set_changes(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> None:
        """Set build change info (must be called before cluster())."""
        self._changed_files = changed_files
        self._file_diffs = file_diffs or {}

    def cluster(
        self, failures: list[FailureInstance]
    ) -> tuple[list[Cluster], list[str]]:
        if not failures:
            return [], []
        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        self._ensure_components()

        # Generate hypotheses
        hypotheses = None
        if self._changed_files and self._hypothesis_generator:
            batch = self._hypothesis_generator.generate(
                failures, self._changed_files, self._file_diffs
            )
            hypotheses = batch.hypotheses

        # Compute all 7 signal matrices
        signals = self._signal_computer.compute(
            failures,
            changed_files=self._changed_files or None,
            file_diffs=self._file_diffs or None,
            hypotheses=hypotheses,
        )

        all_signals = signals.get_all_signals()
        n = len(failures)

        # Build feature matrix for all upper-triangle pairs
        matrices = {}
        for name in SIGNAL_NAMES:
            matrices[name] = all_signals.get(name, np.zeros((n, n)))

        pair_features = []
        pair_indices = []
        for i in range(n):
            for j in range(i + 1, n):
                features = [matrices[name][i, j] for name in SIGNAL_NAMES]
                pair_features.append(features)
                pair_indices.append((i, j))

        X_pairs = np.array(pair_features, dtype=np.float32)

        # Predict P(same_cluster)
        probas = self._model.predict_proba(X_pairs)[:, 1]

        # Build similarity matrix
        sim_matrix = np.eye(n)
        for k, (i, j) in enumerate(pair_indices):
            sim_matrix[i, j] = probas[k]
            sim_matrix[j, i] = probas[k]

        # Hierarchical clustering
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_matrix[i, j])

        Z = linkage(condensed, method="average")
        labels = fcluster(Z, t=self._distance_threshold, criterion="distance") - 1

        # Convert labels to clusters
        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, []
