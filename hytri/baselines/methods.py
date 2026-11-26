"""Baseline Clustering Methods.

Implements baselines from BuildSheriff paper (ICSE 2022) for comparison:
1. Naive: Cluster by exception type only
2. Rebucket: Stack frame TF-IDF similarity (Lerch & Mezini, FSE 2013)
3. PWOverlap: Position-weighted stack frame overlap similarity
4. TraceSim: Weighted edit-distance trace similarity (Jiang et al., ASE 2020)
4. BuildSheriff-lite: BuildSheriff without change-awareness

These baselines allow fair comparison showing HyTri's improvements from:
- LLM semantic understanding
- Multi-signal fusion
- Evidence-verified explanations
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import math
import uuid
from typing import Literal

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.cluster import AgglomerativeClustering

from hytri.models import Cluster, FailureInstance

logger = logging.getLogger(__name__)


class BaselineClusterer(ABC):
    """Abstract base class for baseline clustering methods."""

    @abstractmethod
    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        """Cluster failures.

        Args:
            failures: List of failure instances.

        Returns:
            Tuple of (clusters, noise_failure_ids).
        """
        pass

    def _create_cluster(self, failures: list[FailureInstance]) -> Cluster:
        """Create a Cluster object from failures."""
        failure_ids = [f.failure_id for f in failures]
        exception_types = list(set(f.exception_type for f in failures if f.exception_type))

        frame_counts: Counter = Counter()
        for f in failures:
            for frame in f.frames:
                frame_counts[f"{frame.file}:{frame.func}"] += 1
        top_frames = [frame for frame, _ in frame_counts.most_common(5)]

        unique_tests = set(f.test_name for f in failures if f.test_name)
        
        unique_modules = set()
        for f in failures:
            for frame in f.frames:
                unique_modules.add(frame.file)

        return Cluster(
            cluster_id=str(uuid.uuid4())[:8],
            failure_ids=failure_ids,
            representative_ids=failure_ids[:5],
            exception_types=exception_types,
            top_frames=top_frames,
            num_unique_tests=len(unique_tests),
            num_unique_modules=len(unique_modules),
        )


class NaiveClusterer(BaselineClusterer):
    """Naive baseline: Cluster by exception type only.

    This is the simplest possible triage - group all failures with the
    same exception type (e.g., AssertionError, NullPointerException).

    From BuildSheriff paper: "All failures with the same exception type 
    are put into the same cluster."
    """

    def __init__(self, granularity: Literal["type", "message"] = "type"):
        """Initialize naive clusterer.

        Args:
            granularity: "type" groups by exception class, "message" by full message.
        """
        self.granularity = granularity

    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        """Cluster by exception type or message."""
        if not failures:
            return [], []

        groups: dict[str, list[FailureInstance]] = defaultdict(list)

        for f in failures:
            if self.granularity == "type":
                key = f.exception_type or "Unknown"
            else:
                key = f.error_message[:100] if f.error_message else "Unknown"
            groups[key].append(f)

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, []


class RebucketClusterer(BaselineClusterer):
    """ReBucket: Position Dependent Model for crash report clustering.

    From Dang et al., ICSE 2012: "ReBucket: A Method for Clustering 
    Duplicate Crash Reports Based on Call Stack Similarity"
    
    Key innovations:
    1. Position Dependent Model (PDM): Frames near crash point matter more
    2. Alignment offset weighting: Matched frames should appear at similar depths
    3. DP-based subsequence matching: Finds optimal common frame alignment
    4. Complete linkage clustering: Conservative merging strategy
    
    The similarity function uses:
        Q(Li) = Σk exp(-c·min(pos1,pos2)) · exp(-o·|pos1-pos2|)
        sim(C1, C2) = max{Li} Q(Li)
    
    Where:
        - c = weight for distance from crash (position decay)
        - o = weight for alignment offset (penalizes misaligned matches)
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        c: float = 0.1,  # Position decay weight
        o: float = 0.1,  # Offset penalty weight
        remove_recursive: bool = True,
        immune_functions: set[str] | None = None,
    ):
        """Initialize ReBucket clusterer.

        Args:
            distance_threshold: Threshold for hierarchical clustering (d in paper).
            c: Weight for distance from crash point. Higher = faster decay.
            o: Weight for alignment offset penalty. Higher = stricter alignment.
            remove_recursive: Whether to remove recursive function calls.
            immune_functions: Set of function names to filter out (stable library routines).
        """
        self.distance_threshold = distance_threshold
        self.c = c  # Position weight parameter
        self.o = o  # Offset weight parameter
        self.remove_recursive = remove_recursive
        self.immune_functions = immune_functions or set()

    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        """Cluster using ReBucket PDM similarity with complete linkage."""
        if not failures:
            return [], []

        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        # Compute PDM similarity matrix
        sim_matrix = self._compute_similarity_matrix(failures)

        # Convert to distance and cluster
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        # Complete linkage hierarchical clustering (as per paper)
        labels = self._hierarchical_cluster(dist_matrix)

        return self._labels_to_clusters(labels, failures)

    def _preprocess_frames(self, frames: list[str]) -> list[str]:
        """Preprocess stack frames: remove immune functions and recursive calls.
        
        From paper Section 2:
        1. Immune function removal: Remove stable library routines
        2. Recursive function removal: Keep only one occurrence of repeated calls
        """
        # Remove immune functions
        if self.immune_functions:
            frames = [f for f in frames if not any(
                immune in f for immune in self.immune_functions
            )]
        
        # Remove recursive function calls (keep first occurrence)
        if self.remove_recursive:
            frames = self._remove_recursive_calls(frames)
        
        return frames

    def _remove_recursive_calls(self, frames: list[str]) -> list[str]:
        """Remove repeated recursive calls, keeping only one occurrence.
        
        Uses Brodie et al.'s algorithm: if we see the same frame consecutively
        or in a pattern, collapse it to one occurrence.
        """
        if not frames:
            return frames
        
        result = []
        seen_consecutive = set()
        
        for i, frame in enumerate(frames):
            # Check if this is a consecutive repeat
            if result and frame == result[-1]:
                continue  # Skip consecutive duplicate
            
            # Check for recursive pattern (A -> B -> A pattern)
            if len(result) >= 2 and frame == result[-2]:
                # This might be a recursive call, skip it
                continue
            
            result.append(frame)
        
        return result

    def _compute_similarity_matrix(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute ReBucket PDM similarity matrix using DP."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        # Extract and preprocess frame signatures
        frame_lists = []
        for f in failures:
            frames = [f"{frame.file}:{frame.func}" for frame in f.frames]
            frames = self._preprocess_frames(frames)
            frame_lists.append(frames)

        # Compute pairwise PDM similarity
        for i in range(n):
            for j in range(i, n):
                sim = self._pdm_similarity(frame_lists[i], frame_lists[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _pdm_similarity(self, frames1: list[str], frames2: list[str]) -> float:
        """Compute Position Dependent Model similarity using DP.
        
        The PDM finds the best-matching common subsequence and scores it:
            Q(Li) = Σk exp(-c·min(pos1,pos2)) · exp(-o·|pos1-pos2|)
        
        Uses dynamic programming inspired by LCS to find optimal alignment.
        Time complexity: O(n * m) where n, m are stack lengths.
        """
        if not frames1 and not frames2:
            return 1.0
        if not frames1 or not frames2:
            return 0.0

        n, m = len(frames1), len(frames2)
        
        # DP matrix: M[i][j] = best similarity score using frames1[:i] and frames2[:j]
        # Each cell stores the maximum Q score achievable
        M = np.zeros((n + 1, m + 1))
        
        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Option 1: Don't match frame i with frame j
                M[i][j] = max(M[i-1][j], M[i][j-1])
                
                # Option 2: Match frame i-1 with frame j-1 (if they're the same)
                if frames1[i-1] == frames2[j-1]:
                    # Position indices (0-indexed from top of stack)
                    pos1 = i - 1  # Position in stack 1
                    pos2 = j - 1  # Position in stack 2
                    
                    # PDM score for this match:
                    # exp(-c * min(pos1, pos2)) * exp(-o * |pos1 - pos2|)
                    position_weight = math.exp(-self.c * min(pos1, pos2))
                    offset_penalty = math.exp(-self.o * abs(pos1 - pos2))
                    match_score = position_weight * offset_penalty
                    
                    # Add this match to the best previous score
                    M[i][j] = max(M[i][j], M[i-1][j-1] + match_score)
        
        # The final similarity is the best score found
        best_score = M[n][m]
        
        # Normalize by the maximum possible score (all frames match perfectly at same positions)
        # Max score would be if one stack is a prefix of the other, all frames matching
        max_possible = sum(math.exp(-self.c * k) for k in range(min(n, m)))
        
        if max_possible > 0:
            return best_score / max_possible
        return 0.0

    def _hierarchical_cluster(self, dist_matrix: np.ndarray) -> np.ndarray:
        """Perform hierarchical clustering with COMPLETE linkage.
        
        From paper: "A crash can join a bucket only if it is similar to 
        ALL crashes already in that bucket" - this is complete linkage.
        """
        condensed = []
        n = dist_matrix.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_matrix[i, j])

        if not condensed:
            return np.array([0])

        # Complete linkage as specified in paper
        Z = linkage(condensed, method="complete")
        labels = fcluster(Z, t=self.distance_threshold, criterion="distance")
        return labels - 1  # Make 0-indexed

    def _labels_to_clusters(
        self,
        labels: np.ndarray,
        failures: list[FailureInstance],
    ) -> tuple[list[Cluster], list[str]]:
        """Convert labels to clusters."""
        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, []


class PWOverlapClusterer(BaselineClusterer):
    """PWOverlap: Position-weighted stack frame overlap similarity.

    A simple baseline that compares stack traces position-by-position,
    weighting top frames (closest to crash) more heavily.
    """

    def __init__(
        self,
        distance_threshold: float = 0.5,
        position_weight: Literal["linear", "exponential"] = "exponential",
    ):
        self.distance_threshold = distance_threshold
        self.position_weight = position_weight

    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        if not failures:
            return [], []

        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        sim_matrix = self._compute_similarity_matrix(failures)
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)

        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = clusterer.fit_predict(dist_matrix)
        except ValueError:
            labels = np.zeros(len(failures), dtype=int)

        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, []

    def _compute_similarity_matrix(self, failures: list[FailureInstance]) -> np.ndarray:
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        frame_lists = []
        for f in failures:
            frames = [f"{frame.file}:{frame.func}" for frame in f.frames]
            frame_lists.append(frames)

        for i in range(n):
            for j in range(i, n):
                sim = self._pw_overlap_similarity(frame_lists[i], frame_lists[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _pw_overlap_similarity(self, frames1: list[str], frames2: list[str]) -> float:
        if not frames1 and not frames2:
            return 1.0
        if not frames1 or not frames2:
            return 0.0

        max_len = max(len(frames1), len(frames2))
        total_weight = 0.0
        matched_weight = 0.0

        for i in range(max_len):
            if self.position_weight == "exponential":
                weight = 1.0 / (2 ** i)
            else:
                weight = 1.0 / (i + 1)

            total_weight += weight

            f1 = frames1[i] if i < len(frames1) else None
            f2 = frames2[i] if i < len(frames2) else None

            if f1 is not None and f2 is not None and f1 == f2:
                matched_weight += weight

        return matched_weight / total_weight if total_weight > 0 else 0.0


# Keep backward-compatible alias
TraceSimClusterer = PWOverlapClusterer


class TraceSimOfficialClusterer(BaselineClusterer):
    """TraceSim: Weighted Edit-Distance Trace Similarity.

    From Jiang et al., ASE 2020: "TraceSim: An Alignment Method for
    Computing Stack Trace Similarity"

    TraceSim computes similarity between two stack traces using a weighted
    edit distance (Levenshtein) where frame weights combine:
    1. Local weight (positional): frames near the top of the stack are
       more important — weight = 1/(1+i)^alpha
    2. Global weight (IDF): rare frames across the corpus are more
       informative — sigmoid of IDF score: 1/(1+exp(-beta*idf+gamma))

    The similarity is: 1 - dist(t1,t2) / max_dist
    where max_dist = sum(weights1) + sum(weights2).

    Reference implementation:
    https://github.com/traceSimSubmission/trace-sim/blob/master/trace_sim.py
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 1.0,
        gamma: float = 5.0,
        distance_threshold: float = 0.5,
    ):
        """
        Args:
            alpha: Controls positional decay. Higher = faster decay from top.
            beta: IDF sigmoid steepness. Higher = sharper separation.
            gamma: IDF sigmoid midpoint shift. Adjusts the IDF cutoff.
            distance_threshold: Threshold for hierarchical clustering.
        """
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.distance_threshold = distance_threshold

    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        if not failures:
            return [], []

        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        frame_lists = []
        for f in failures:
            frames = [f"{frame.file}:{frame.func}" for frame in f.frames]
            frame_lists.append(frames)

        idf = self._compute_idf(frame_lists)

        n = len(failures)
        sim_matrix = np.zeros((n, n))
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self._tracesim_compare(
                    frame_lists[i], frame_lists[j], idf
                )
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = clusterer.fit_predict(dist_matrix)
        except ValueError:
            labels = np.zeros(n, dtype=int)

        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, []

    # ------------------------------------------------------------------
    # IDF computation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_idf(traces: list[list[str]]) -> dict[str, float]:
        """Compute IDF for each unique frame across all traces.

        IDF(frame) = log(N / df(frame))
        where N = number of traces and df = number of traces containing frame.
        """
        n = len(traces)
        if n == 0:
            return {}
        df: Counter = Counter()
        for trace in traces:
            for frame in set(trace):
                df[frame] += 1
        return {frame: math.log(n / count) for frame, count in df.items()}

    # ------------------------------------------------------------------
    # Core TraceSim algorithm (faithful to reference implementation)
    # ------------------------------------------------------------------

    def _tracesim_weights(
        self, trace: list[str], idf: dict[str, float]
    ) -> list[float]:
        """Compute per-frame weights combining position and IDF."""
        local_weights = [
            1.0 / (1 + i) ** self.alpha for i in range(len(trace))
        ]
        global_weights = [
            1.0 / (1.0 + math.exp(-self.beta * idf.get(frame, 0.0) + self.gamma))
            for frame in trace
        ]
        return [lw * gw for lw, gw in zip(local_weights, global_weights)]

    def _tracesim_dist(
        self,
        trace1: list[str],
        weights1: list[float],
        trace2: list[str],
        weights2: list[float],
    ) -> float:
        """Weighted edit distance between two traces.

        Uses dynamic programming analogous to Levenshtein distance,
        but with per-element weights for insert/delete/replace operations.
        """
        n1, n2 = len(trace1), len(trace2)
        if n1 == 0 or n2 == 0:
            return 0.0

        matrix = [[0.0] * (n1 + 1) for _ in range(n2 + 1)]

        prev_col = matrix[0]
        for i in range(n1):
            prev_col[i + 1] = prev_col[i] + weights1[i]

        curr_col = matrix[1]
        for i2 in range(n2):
            frame2 = trace2[i2]
            w2 = weights2[i2]
            curr_col[0] = prev_col[0] + w2

            for i1 in range(n1):
                frame1 = trace1[i1]
                w1 = weights1[i1]

                if frame1 == frame2:
                    curr_col[i1 + 1] = prev_col[i1]
                else:
                    change = w1 + w2 + prev_col[i1]
                    remove = w2 + prev_col[i1 + 1]
                    insert = w1 + curr_col[i1]
                    curr_col[i1 + 1] = min(change, remove, insert)

            if i2 != n2 - 1:
                prev_col = curr_col
                curr_col = matrix[i2 + 2]

        return curr_col[-1]

    def _tracesim_compare(
        self,
        trace1: list[str],
        trace2: list[str],
        idf: dict[str, float],
    ) -> float:
        """Compute TraceSim similarity between two traces."""
        weights1 = self._tracesim_weights(trace1, idf)
        weights2 = self._tracesim_weights(trace2, idf)

        max_dist = sum(weights1) + sum(weights2)
        if max_dist == 0:
            return 0.0

        dist = self._tracesim_dist(trace1, weights1, trace2, weights2)
        return 1.0 - dist / max_dist


class BuildSheriffLiteClusterer(BaselineClusterer):
    """BuildSheriff-lite: BuildSheriff without change-awareness.

    From Zhang et al., ICSE 2022: "BuildSheriff: Change-Aware Test 
    Failure Triage for Continuous Integration Builds"

    Implements their similarity pipeline WITHOUT the change-aware components:
    - Stack trace similarity (exception failures)
    - Exception message similarity (exception failures)
    - Test code similarity (assertion failures) - simplified to test name

    Note: The original BuildSheriff uses git diff information which we 
    don't have access to. This is a fair baseline that represents their 
    method in a change-agnostic context.
    """

    def __init__(
        self,
        distance_threshold: float = 0.4,
        stack_weight: float = 0.5,
        message_weight: float = 0.3,
        test_weight: float = 0.2,
    ):
        """Initialize BuildSheriff-lite.

        Args:
            distance_threshold: Clustering distance threshold.
            stack_weight: Weight for stack trace similarity.
            message_weight: Weight for exception message similarity.
            test_weight: Weight for test name similarity.
        """
        self.distance_threshold = distance_threshold
        self.stack_weight = stack_weight
        self.message_weight = message_weight
        self.test_weight = test_weight

    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        """Cluster using BuildSheriff-lite multi-signal approach."""
        if not failures:
            return [], []

        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        # Compute combined similarity
        sim_matrix = self._compute_combined_similarity(failures)
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)

        # Hierarchical clustering
        try:
            clusterer = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=self.distance_threshold,
                metric="precomputed",
                linkage="average",
            )
            labels = clusterer.fit_predict(dist_matrix)
        except ValueError:
            labels = np.zeros(len(failures), dtype=int)

        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, []

    def _compute_combined_similarity(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute weighted combination of similarity signals."""
        n = len(failures)

        stack_sim = self._compute_stack_similarity(failures)
        message_sim = self._compute_message_similarity(failures)
        test_sim = self._compute_test_similarity(failures)

        # Weighted combination
        combined = (
            self.stack_weight * stack_sim +
            self.message_weight * message_sim +
            self.test_weight * test_sim
        )

        # Normalize to [0, 1]
        combined = np.clip(combined, 0.0, 1.0)
        np.fill_diagonal(combined, 1.0)

        return combined

    def _compute_stack_similarity(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute stack trace Jaccard similarity."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        frame_sets = []
        for f in failures:
            frames = set(f"{frame.file}:{frame.func}" for frame in f.frames)
            frame_sets.append(frames)

        for i in range(n):
            for j in range(i, n):
                if not frame_sets[i] and not frame_sets[j]:
                    sim = 1.0
                elif not frame_sets[i] or not frame_sets[j]:
                    sim = 0.0
                else:
                    inter = len(frame_sets[i] & frame_sets[j])
                    union = len(frame_sets[i] | frame_sets[j])
                    sim = inter / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _compute_message_similarity(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute exception message similarity using token overlap."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        # Tokenize messages
        token_sets = []
        for f in failures:
            msg = f.exception_message or ""
            # Simple tokenization
            tokens = set(msg.lower().split())
            token_sets.append(tokens)

        for i in range(n):
            for j in range(i, n):
                if not token_sets[i] and not token_sets[j]:
                    sim = 1.0
                elif not token_sets[i] or not token_sets[j]:
                    sim = 0.0
                else:
                    inter = len(token_sets[i] & token_sets[j])
                    union = len(token_sets[i] | token_sets[j])
                    sim = inter / union if union > 0 else 0.0
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _compute_test_similarity(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute test name similarity."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                test_i = failures[i].test_name or ""
                test_j = failures[j].test_name or ""

                # Same test = 1.0, otherwise check token overlap
                if test_i and test_j and test_i == test_j:
                    sim = 1.0
                else:
                    # Token overlap on test names
                    tokens_i = set(test_i.lower().replace("_", " ").replace("::", " ").split())
                    tokens_j = set(test_j.lower().replace("_", " ").replace("::", " ").split())
                    if tokens_i and tokens_j:
                        inter = len(tokens_i & tokens_j)
                        union = len(tokens_i | tokens_j)
                        sim = 0.5 * (inter / union) if union > 0 else 0.0
                    else:
                        sim = 0.0

                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix


class BuildSheriffClusterer(BaselineClusterer):
    """Full BuildSheriff with change-awareness - Decision Pipeline Implementation.

    From Zhang et al., ICSE 2022: "BuildSheriff: Change-Aware Test 
    Failure Triage for Continuous Integration Builds"

    IMPORTANT: BuildSheriff is NOT a weighted-average approach. It uses a 
    DECISION PIPELINE that chooses which triage strategy to apply based on 
    failure type and build characteristics.

    The algorithm first splits failures into two DISJOINT categories:
    1. Exception failures (crashes, NPEs, etc.) - have stack traces
    2. Assertion failures (test assertions failing) - typically no meaningful stack

    PIPELINE FOR EXCEPTION FAILURES:
    --------------------------------
    Step 1 (S1_e): Complexity of code changes
        - If ≤ Δe (7) changed methods → assume single root cause → ONE cluster → STOP
        - Else → Go to Step 2
    
    Step 2 (S2_e): Change-aware stack trace similarity
        - Uses weighted stack-trace similarity with single-linkage clustering
        - Threshold: dt_st = 0.57 (distance threshold)
        - If stack traces are informative (≥ min_frames) → cluster using S2 → STOP
        - Else → Go to Step 3
    
    Step 3 (S3_e): Exception message similarity
        - Token-level Levenshtein similarity with single-linkage clustering
        - Threshold: dt_msg = 0.73
        - Form final clusters

    PIPELINE FOR ASSERTION FAILURES:
    --------------------------------
    Step 1 (S1_a): Complexity of code changes
        - If ≤ Δa (7) changed methods → assume single root cause → ONE cluster → STOP
        - Else → Go to Step 2
    
    Step 2 (S2_a): Test code similarity
        - Compare test method bodies/names
        - Single-linkage clustering with threshold dt_test = 0.5
        - → Form final clusters
    """

    # Paper-specified thresholds
    DELTA_EXCEPTION = 7      # Δe: changed methods threshold for exceptions
    DELTA_ASSERTION = 7      # Δa: changed methods threshold for assertions
    DT_STACK_TRACE = 0.57    # dt_st: stack trace distance threshold
    DT_MESSAGE = 0.73        # dt_msg: message distance threshold
    DT_TEST_CODE = 0.5       # dt_test: test code distance threshold
    MIN_INFORMATIVE_FRAMES = 3  # Minimum frames for stack trace to be "informative"

    def __init__(
        self,
        # Thresholds can be overridden if needed
        delta_exception: int = 7,
        delta_assertion: int = 7,
        dt_stack_trace: float = 0.57,
        dt_message: float = 0.73,
        dt_test_code: float = 0.5,
        min_informative_frames: int = 3,
        # Change-aware parameters
        change_boost_gamma: float = 0.5,  # γ: boost factor for changed frames
    ):
        """Initialize BuildSheriff with decision pipeline.

        Args:
            delta_exception: Δe - max changed methods before clustering exceptions.
            delta_assertion: Δa - max changed methods before clustering assertions.
            dt_stack_trace: Distance threshold for stack trace clustering.
            dt_message: Distance threshold for message clustering.
            dt_test_code: Distance threshold for test code clustering.
            min_informative_frames: Min frames for stack trace to be informative.
            change_boost_gamma: Boost factor for frames touching changed files.
        """
        self.delta_exception = delta_exception
        self.delta_assertion = delta_assertion
        self.dt_stack_trace = dt_stack_trace
        self.dt_message = dt_message
        self.dt_test_code = dt_test_code
        self.min_informative_frames = min_informative_frames
        self.change_boost_gamma = change_boost_gamma
        
        # Change information (set via set_changes)
        self.changed_files: list[str] = []
        self.changed_files_base: list[str] = []
        self.changed_methods_count: int = 0

    def set_changes(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
        changed_methods_count: int | None = None,
    ) -> None:
        """Set the code changes for this build.
        
        Args:
            changed_files: List of changed file paths.
            file_diffs: Optional dict mapping file path to diff content.
            changed_methods_count: Number of changed methods (if known).
                                   If not provided, estimated from file count.
        """
        self.changed_files = changed_files
        self.changed_files_base = [f.split('/')[-1] for f in changed_files]
        
        # Estimate changed methods count if not provided
        if changed_methods_count is not None:
            self.changed_methods_count = changed_methods_count
        elif file_diffs:
            # Rough estimate: count function-like patterns in diffs
            total_methods = 0
            for diff in file_diffs.values():
                # Count method signatures (rough heuristic)
                # Look for patterns like "def ", "public void", "private int", etc.
                total_methods += diff.count('\n+    def ')
                total_methods += diff.count('\n+  def ')
                total_methods += diff.count('\n+def ')
                total_methods += diff.count('\n+    public ')
                total_methods += diff.count('\n+    private ')
                total_methods += diff.count('\n+    protected ')
                # Also count modified methods (lines starting with @@ that have method context)
                total_methods += diff.count('@@ ')
            self.changed_methods_count = max(total_methods, len(changed_files))
        else:
            # Fallback: assume ~2 methods per file on average
            self.changed_methods_count = len(changed_files) * 2

    def cluster(self, failures: list[FailureInstance]) -> tuple[list[Cluster], list[str]]:
        """Cluster using BuildSheriff decision pipeline.
        
        This is the main entry point that implements the full decision tree:
        1. Split failures by type (exception vs assertion)
        2. Apply appropriate pipeline to each group
        3. Combine results
        """
        if not failures:
            return [], []

        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        # Step 0: Split failures by type
        exception_failures, assertion_failures = self._split_by_failure_type(failures)
        
        # Process each group with its respective pipeline
        all_clusters = []
        
        if exception_failures:
            exc_clusters = self._cluster_exception_failures(exception_failures)
            all_clusters.extend(exc_clusters)
        
        if assertion_failures:
            asrt_clusters = self._cluster_assertion_failures(assertion_failures)
            all_clusters.extend(asrt_clusters)

        return all_clusters, []

    def _split_by_failure_type(
        self, failures: list[FailureInstance]
    ) -> tuple[list[FailureInstance], list[FailureInstance]]:
        """Split failures into exception failures and assertion failures.
        
        Exception failures: Have meaningful exception types (NPE, RuntimeError, etc.)
        Assertion failures: AssertionError, assert failures, or no exception type
        """
        exception_failures = []
        assertion_failures = []
        
        assertion_types = {
            'assertionerror', 'assertion', 'assert', 'junit.framework.assertionfailederror',
            'org.junit.comparisonFailure', 'java.lang.assertionerror',
            'pytest.fail', 'unittest.case._assertionerror',
        }
        
        for f in failures:
            exc_type = (f.exception_type or "").lower()
            
            # Check if it's an assertion failure
            is_assertion = (
                any(at in exc_type for at in assertion_types) or
                'assert' in exc_type or
                (not exc_type and not f.frames)  # No exception and no stack = likely assertion
            )
            
            if is_assertion:
                assertion_failures.append(f)
            else:
                exception_failures.append(f)
        
        return exception_failures, assertion_failures

    def _cluster_exception_failures(
        self, failures: list[FailureInstance]
    ) -> list[Cluster]:
        """Apply exception failure pipeline: S1_e → S2_e → S3_e.
        
        Decision tree:
        1. S1_e: If few changed methods → single cluster
        2. S2_e: If stack traces informative → cluster by stack similarity
        3. S3_e: Fallback to message similarity
        """
        if not failures:
            return []
        
        if len(failures) == 1:
            return [self._create_cluster(failures)]

        # S1_e: Check complexity of code changes
        if self.changed_methods_count <= self.delta_exception:
            # Few changes → assume single root cause → one cluster
            return [self._create_cluster(failures)]

        # Check if stack traces are informative
        informative_count = sum(
            1 for f in failures 
            if len(f.frames) >= self.min_informative_frames
        )
        stack_traces_informative = informative_count >= len(failures) * 0.5

        if stack_traces_informative:
            # S2_e: Use change-aware stack trace similarity
            return self._cluster_by_stack_similarity(failures)
        else:
            # S3_e: Fallback to exception message similarity
            return self._cluster_by_message_similarity(failures)

    def _cluster_assertion_failures(
        self, failures: list[FailureInstance]
    ) -> list[Cluster]:
        """Apply assertion failure pipeline: S1_a → S2_a.
        
        Decision tree:
        1. S1_a: If few changed methods → single cluster
        2. S2_a: Cluster by test code/name similarity
        """
        if not failures:
            return []
        
        if len(failures) == 1:
            return [self._create_cluster(failures)]

        # S1_a: Check complexity of code changes
        if self.changed_methods_count <= self.delta_assertion:
            # Few changes → assume single root cause → one cluster
            return [self._create_cluster(failures)]

        # S2_a: Cluster by test code similarity
        return self._cluster_by_test_similarity(failures)

    def _cluster_by_stack_similarity(
        self, failures: list[FailureInstance]
    ) -> list[Cluster]:
        """S2_e: Cluster by change-aware stack trace similarity.
        
        Uses single-linkage clustering with threshold dt_st = 0.57.
        Frames touching changed files get boosted weight.
        """
        n = len(failures)
        sim_matrix = self._compute_change_aware_stack_similarity(failures)
        
        # Convert to distance
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        # Single-linkage clustering (as per paper)
        return self._single_linkage_cluster(
            failures, dist_matrix, self.dt_stack_trace
        )

    def _cluster_by_message_similarity(
        self, failures: list[FailureInstance]
    ) -> list[Cluster]:
        """S3_e: Cluster by exception message similarity.
        
        Uses token-level Levenshtein similarity with single-linkage clustering.
        Threshold: dt_msg = 0.73.
        """
        n = len(failures)
        sim_matrix = self._compute_message_levenshtein_similarity(failures)
        
        # Convert to distance
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        # Single-linkage clustering
        return self._single_linkage_cluster(
            failures, dist_matrix, self.dt_message
        )

    def _cluster_by_test_similarity(
        self, failures: list[FailureInstance]
    ) -> list[Cluster]:
        """S2_a: Cluster by test code/name similarity.
        
        Single-linkage clustering with threshold dt_test = 0.5.
        """
        n = len(failures)
        sim_matrix = self._compute_test_similarity(failures)
        
        # Convert to distance
        dist_matrix = 1.0 - sim_matrix
        np.fill_diagonal(dist_matrix, 0.0)
        dist_matrix = np.clip(dist_matrix, 0.0, 1.0)

        # Single-linkage clustering
        return self._single_linkage_cluster(
            failures, dist_matrix, self.dt_test_code
        )

    def _single_linkage_cluster(
        self,
        failures: list[FailureInstance],
        dist_matrix: np.ndarray,
        threshold: float,
    ) -> list[Cluster]:
        """Perform single-linkage hierarchical clustering.
        
        BuildSheriff uses single-linkage (minimum distance between clusters).
        """
        n = len(failures)
        
        if n <= 1:
            return [self._create_cluster(failures)] if failures else []

        # Convert to condensed form for scipy
        condensed = []
        for i in range(n):
            for j in range(i + 1, n):
                condensed.append(dist_matrix[i, j])

        if not condensed:
            return [self._create_cluster(failures)]

        # Single-linkage clustering
        Z = linkage(condensed, method="single")
        labels = fcluster(Z, t=threshold, criterion="distance")
        labels = labels - 1  # Make 0-indexed

        # Group by labels
        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        for f, label in zip(failures, labels):
            groups[label].append(f)

        return [self._create_cluster(group) for group in groups.values()]

    def _compute_change_aware_stack_similarity(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute Change-Aware Stack trace Similarity (CAS) from paper.
        
        Uses Position-Dependent Model (PDM) with change-boosting:
        - Each frame has a position weight (top = higher)
        - Frames touching changed files get additional boost (γ factor)
        """
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        # Extract frame data with change information
        frame_data = []
        for f in failures:
            frames_info = []
            for i, frame in enumerate(f.frames):
                frame_sig = f"{frame.file}:{frame.func}"
                frame_base = frame.file.split('/')[-1] if frame.file else ""
                
                # Check if this frame touches a changed file
                touches_change = self._frame_touches_change(frame_base)
                
                frames_info.append({
                    "sig": frame_sig,
                    "touches_change": touches_change,
                })
            frame_data.append(frames_info)

        # Compute pairwise CAS similarity
        for i in range(n):
            for j in range(i, n):
                sim = self._cas_similarity(frame_data[i], frame_data[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _frame_touches_change(self, frame_base: str) -> bool:
        """Check if a frame's file matches any changed file."""
        if not frame_base:
            return False
        for cf_base in self.changed_files_base:
            if cf_base and (
                cf_base in frame_base or 
                frame_base in cf_base or
                cf_base == frame_base
            ):
                return True
        return False

    def _cas_similarity(
        self,
        frames1: list[dict],
        frames2: list[dict],
    ) -> float:
        """Compute CAS similarity between two failure's frames.
        
        Formula from paper:
        CAS(f1, f2) = Σ w_k * (1 + γ * c_k) * match_k / Σ w_k * (1 + γ * c_k)
        
        where:
        - w_k = 1/(k+1) = position weight for frame k (PDM)
        - γ = change boost factor
        - c_k = 1 if frame touches changed file, 0 otherwise
        - match_k = 1 if frames match at position k, 0 otherwise
        """
        if not frames1 and not frames2:
            return 1.0
        if not frames1 or not frames2:
            return 0.0

        max_len = max(len(frames1), len(frames2))
        total_weight = 0.0
        matched_weight = 0.0

        for k in range(max_len):
            f1 = frames1[k] if k < len(frames1) else None
            f2 = frames2[k] if k < len(frames2) else None

            # Position weight (PDM: w_k = 1/(k+1))
            pos_weight = 1.0 / (k + 1)

            # Change boost (1 if either frame touches change)
            change_indicator = 0.0
            if (f1 and f1["touches_change"]) or (f2 and f2["touches_change"]):
                change_indicator = 1.0

            # Effective weight with change boost
            effective_weight = pos_weight * (1.0 + self.change_boost_gamma * change_indicator)
            total_weight += effective_weight

            # Check match
            if f1 and f2 and f1["sig"] == f2["sig"]:
                matched_weight += effective_weight

        return matched_weight / total_weight if total_weight > 0 else 0.0

    def _compute_message_levenshtein_similarity(
        self, failures: list[FailureInstance]
    ) -> np.ndarray:
        """Compute token-level Levenshtein similarity on exception messages.
        
        Paper specifies token-level Levenshtein for S3_e.
        """
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        # Tokenize messages
        token_lists = []
        for f in failures:
            msg = f.exception_message or ""
            # Simple tokenization: split on whitespace and punctuation
            tokens = msg.lower().replace(':', ' ').replace('.', ' ').replace(',', ' ').split()
            token_lists.append(tokens)

        for i in range(n):
            for j in range(i, n):
                sim = self._token_levenshtein_similarity(token_lists[i], token_lists[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix

    def _token_levenshtein_similarity(
        self, tokens1: list[str], tokens2: list[str]
    ) -> float:
        """Compute Levenshtein similarity at token level.
        
        Similarity = 1 - (edit_distance / max_length)
        """
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0

        # Compute Levenshtein distance on token sequences
        m, n = len(tokens1), len(tokens2)
        
        # Use DP for edit distance
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if tokens1[i-1] == tokens2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_distance = dp[m][n]
        max_len = max(m, n)
        
        return 1.0 - (edit_distance / max_len) if max_len > 0 else 1.0

    def _compute_test_similarity(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute test name/code similarity for assertion failures."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                test_i = failures[i].test_name or ""
                test_j = failures[j].test_name or ""

                if test_i and test_j and test_i == test_j:
                    sim = 1.0
                else:
                    # Token overlap on test names (normalized)
                    tokens_i = set(test_i.lower().replace("_", " ").replace("::", " ").replace(".", " ").split())
                    tokens_j = set(test_j.lower().replace("_", " ").replace("::", " ").replace(".", " ").split())
                    if tokens_i and tokens_j:
                        inter = len(tokens_i & tokens_j)
                        union = len(tokens_i | tokens_j)
                        sim = inter / union if union > 0 else 0.0
                    else:
                        sim = 0.0

                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim

        return sim_matrix


class GPTraceClusterer(BaselineClusterer):
    """GPTrace: LLM-Embedding-Based Crash Deduplication via Clustering.

    Adapted from Herter et al., 2025: "GPTrace: Effective Crash
    Deduplication Using LLM Embeddings" (arXiv:2512.01609).

    GPTrace deduplicates crashes by embedding crash information into a
    semantic vector space using LLM embeddings, then clustering those
    vectors to group crashes that likely stem from the same underlying bug.

    Pipeline (faithfully adapted for CI test failures):
    ---------------------------------------------------
    1. **Preprocessing**: For each failure, construct three complementary
       text views (analogous to GPTrace's full trace / coarse trace / ASan):
       (a) Full stack trace  – complete trace with function names, args, lines
       (b) Coarse stack trace – file:function signatures only (no line numbers)
       (c) Error summary      – exception type + message + test name

    2. **Embedding**: Each view is independently embedded using a text
       embedding model (text-embedding-3-small by default).

    3. **Fusion**: Embeddings are L2-normalized and summed (not concatenated),
       then the fused vector is L2-normalized again. This reinforces signals
       shared across views without inflating dimensionality.

    4. **Clustering**: Hybrid HDBSCAN with automatic epsilon search.
       - min_cluster_size=2, min_samples=1 (even singletons can be real bugs)
       - Searches over candidate epsilon values derived from the actual
         pairwise distance distribution
       - Selects the clustering that maximizes DBCV while penalizing
         excessive cluster counts (matching GPTrace's Algorithm 1 ranking)

    Key Difference from HyTri-E:
        GPTrace does NO structured extraction (no categories, components,
        hypotheses). It relies purely on the LLM embedding model to capture
        semantics from the raw crash text. This makes it the ideal baseline
        for demonstrating the value of HyTri-E's structured multi-signal
        architecture.
    """

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        min_cluster_size: int = 2,
        min_samples: int = 1,
        n_epsilon_candidates: int = 20,
        openai_client: object | None = None,
    ):
        """Initialize GPTrace clusterer.

        Args:
            embedding_model: OpenAI embedding model name.
            min_cluster_size: HDBSCAN min_cluster_size parameter.
            min_samples: HDBSCAN min_samples parameter.
            n_epsilon_candidates: Number of epsilon values to search over.
            openai_client: Optional pre-configured OpenAIClient instance.
                           If None, creates one from OPENAI_API_KEY env var.
        """
        self.embedding_model = embedding_model
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.n_epsilon_candidates = n_epsilon_candidates
        self._openai_client = openai_client

    @property
    def openai_client(self):
        """Lazy-initialize the OpenAI client."""
        if self._openai_client is None:
            from hytri.llm.openai_client import OpenAIClient

            self._openai_client = OpenAIClient(
                embedding_model=self.embedding_model
            )
        return self._openai_client

    # ------------------------------------------------------------------
    # Stage 1: Preprocessing – construct three text views per failure
    # ------------------------------------------------------------------

    def _build_full_trace_view(self, f: FailureInstance) -> str:
        """View (a): Full stack trace text.

        Analogous to GPTrace's full stack trace including function names,
        arguments and source locations.  We collapse consecutive duplicate
        frames (GPTrace collapses duplicate frames that repeat thousands
        of times in fuzzing traces; same spirit here).
        """
        if f.stack_trace:
            return self._collapse_consecutive_duplicate_lines(f.stack_trace)
        # Fallback: reconstruct from parsed frames
        lines = []
        for frame in f.frames:
            lines.append(f"at {frame.func}({frame.file}:{frame.line})")
        return "\n".join(lines) if lines else ""

    def _build_coarse_trace_view(self, f: FailureInstance) -> str:
        """View (b): Coarse stack trace – file:function only, no line numbers.

        Analogous to GPTrace's coarse trace with arguments removed.
        Preserves call structure while suppressing noisy positional details.
        """
        seen = []
        for frame in f.frames:
            sig = f"{frame.file}:{frame.func}"
            # Collapse consecutive identical signatures
            if not seen or seen[-1] != sig:
                seen.append(sig)
        return "\n".join(seen) if seen else ""

    def _build_error_summary_view(self, f: FailureInstance) -> str:
        """View (c): Error summary – exception semantics without stack.

        Analogous to GPTrace's sanitized ASan report: captures the semantic
        description of the error without the embedded stack trace.
        """
        parts = []
        if f.exception_type:
            parts.append(f"Exception: {f.exception_type}")
        if f.exception_message:
            parts.append(f"Message: {f.exception_message}")
        if f.test_name:
            parts.append(f"Test: {f.test_name}")
        return "\n".join(parts) if parts else "Unknown failure"

    @staticmethod
    def _collapse_consecutive_duplicate_lines(text: str) -> str:
        """Collapse consecutive duplicate lines (GPTrace collapses
        repeated frames to avoid redundant embedding weight)."""
        lines = text.split("\n")
        result: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not result or stripped != result[-1]:
                result.append(stripped)
        return "\n".join(result)

    # ------------------------------------------------------------------
    # Stage 2: LLM Embedding
    # ------------------------------------------------------------------

    def _embed_views(
        self,
        failures: list[FailureInstance],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed each of the three views for all failures.

        Returns:
            Tuple of (full_embeds, coarse_embeds, summary_embeds), each
            shape (n_failures, embed_dim).
        """
        full_texts = [self._build_full_trace_view(f) for f in failures]
        coarse_texts = [self._build_coarse_trace_view(f) for f in failures]
        summary_texts = [self._build_error_summary_view(f) for f in failures]

        logger.info("GPTrace: Embedding %d failures (3 views each)...", len(failures))
        full_embeds = np.array(self.openai_client.get_embeddings(full_texts))
        coarse_embeds = np.array(self.openai_client.get_embeddings(coarse_texts))
        summary_embeds = np.array(self.openai_client.get_embeddings(summary_texts))

        return full_embeds, coarse_embeds, summary_embeds

    # ------------------------------------------------------------------
    # Stage 3: Embedding Fusion & Normalization
    # ------------------------------------------------------------------

    @staticmethod
    def _fuse_embeddings(
        full: np.ndarray,
        coarse: np.ndarray,
        summary: np.ndarray,
    ) -> np.ndarray:
        """Fuse three embedding views via L2-normalize-then-sum.

        From the paper:
            v_s = Σ_{d ∈ D}  v_{d,s} / ||v_{d,s}||

        After summation, the fused vector is L2-normalized again so that
        Euclidean distance ≈ cosine distance and the triangle inequality
        holds (important for HDBSCAN's mutual reachability distance).
        """
        def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)  # avoid div-by-zero
            return matrix / norms

        fused = (
            _l2_normalize(full)
            + _l2_normalize(coarse)
            + _l2_normalize(summary)
        )
        return _l2_normalize(fused)

    # ------------------------------------------------------------------
    # Stage 4: Hybrid HDBSCAN Clustering with Epsilon Search
    # ------------------------------------------------------------------

    def _cluster_embeddings(
        self,
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """Cluster fused embeddings using hybrid HDBSCAN + epsilon search.

        GPTrace (Algorithm 1):
        1. Run base HDBSCAN to build the condensed tree / MST.
        2. Generate candidate epsilon values from the pairwise distance
           distribution (percentiles of actual distances).
        3. For each epsilon, extract a flat clustering using HDBSCAN in
           "leaf" or "eom" mode with that epsilon as cluster_selection_epsilon.
        4. Rank candidates by (DBCV, persistence, fewer clusters) and pick
           the best.

        Returns:
            Integer cluster labels (0-indexed). -1 = noise.
        """
        try:
            import hdbscan as _hdbscan
        except ImportError:
            logger.warning(
                "hdbscan package not installed. Falling back to agglomerative "
                "clustering. Install with: pip install hdbscan"
            )
            return self._fallback_agglomerative(embeddings)

        n = embeddings.shape[0]
        if n <= 2:
            return np.zeros(n, dtype=int)

        # --- Compute pairwise Euclidean distance matrix ---
        # (On L2-normalized vectors, Euclidean² = 2 - 2·cos, so
        #  Euclidean distance is a monotone transform of cosine distance.)
        from scipy.spatial.distance import pdist, squareform as _squareform

        dists_condensed = pdist(embeddings, metric="euclidean")
        dist_matrix = _squareform(dists_condensed)

        # --- Generate candidate epsilon values ---
        # Sample from actual distance percentiles (as GPTrace does)
        percentiles = np.linspace(5, 95, self.n_epsilon_candidates)
        eps_candidates = np.unique(np.percentile(dists_condensed, percentiles))

        # Always include 0.0 (pure HDBSCAN, no epsilon flattening)
        eps_candidates = np.concatenate([[0.0], eps_candidates])

        # --- Search for best epsilon ---
        best_labels = None
        best_score = -np.inf

        for eps in eps_candidates:
            clusterer = _hdbscan.HDBSCAN(
                min_cluster_size=self.min_cluster_size,
                min_samples=self.min_samples,
                metric="precomputed",
                cluster_selection_epsilon=float(eps),
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(dist_matrix)

            n_clusters = len(set(labels) - {-1})
            n_noise = int(np.sum(labels == -1))

            if n_clusters < 1:
                continue

            # Compute DBCV (GPTrace's primary ranking criterion).
            # hdbscan stores relative_validity_ after fit.
            dbcv = getattr(clusterer, "relative_validity_", 0.0)

            # Penalise excessive noise (>50% noise is bad)
            noise_penalty = max(0.0, (n_noise / n) - 0.5)

            # Mildly prefer fewer clusters (GPTrace: fewer = less triage)
            cluster_bonus = 1.0 / (1.0 + n_clusters)

            score = dbcv - noise_penalty + 0.05 * cluster_bonus

            if score > best_score:
                best_score = score
                best_labels = labels

        if best_labels is None:
            # No valid clustering found – put everything in one cluster
            return np.zeros(n, dtype=int)

        return best_labels

    def _fallback_agglomerative(self, embeddings: np.ndarray) -> np.ndarray:
        """Fallback when hdbscan is not installed.

        Uses agglomerative clustering with average linkage on cosine
        distance, cutting at median pairwise distance.
        """
        from scipy.spatial.distance import pdist

        n = embeddings.shape[0]
        if n <= 2:
            return np.zeros(n, dtype=int)

        dists_condensed = pdist(embeddings, metric="cosine")
        threshold = float(np.median(dists_condensed))
        threshold = max(threshold, 0.01)

        Z = linkage(dists_condensed, method="average")
        labels = fcluster(Z, t=threshold, criterion="distance")
        return labels - 1  # 0-indexed

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def cluster(
        self, failures: list[FailureInstance]
    ) -> tuple[list[Cluster], list[str]]:
        """Cluster failures using the GPTrace pipeline.

        1. Preprocess → 3 text views per failure
        2. Embed each view via LLM
        3. Fuse embeddings (L2-norm, sum, L2-norm)
        4. Cluster with hybrid HDBSCAN + epsilon search

        Args:
            failures: List of failure instances.

        Returns:
            Tuple of (clusters, noise_failure_ids).
        """
        if not failures:
            return [], []

        if len(failures) == 1:
            return [self._create_cluster(failures)], []

        # --- GPTrace Stage 0: Remove exact duplicates (dedup) ---
        # GPTrace removes identical crash inputs before embedding.
        # We deduplicate on (exception_type, exception_message, coarse trace)
        # but keep a mapping so we can expand labels back to all failures.
        dedup_map, unique_failures = self._deduplicate(failures)

        if len(unique_failures) == 1:
            return [self._create_cluster(failures)], []

        # --- Stages 1-3: Embed & fuse ---
        full_emb, coarse_emb, summary_emb = self._embed_views(unique_failures)
        fused = self._fuse_embeddings(full_emb, coarse_emb, summary_emb)

        # --- Stage 4: Cluster ---
        labels = self._cluster_embeddings(fused)

        # --- Expand labels back to all failures (undo dedup) ---
        expanded_labels = self._expand_labels(labels, dedup_map, len(failures))

        # --- Build output clusters ---
        return self._labels_to_clusters(expanded_labels, failures)

    # ------------------------------------------------------------------
    # Deduplication helpers
    # ------------------------------------------------------------------

    def _deduplicate(
        self,
        failures: list[FailureInstance],
    ) -> tuple[dict[int, list[int]], list[FailureInstance]]:
        """Deduplicate failures by (exception_type, message, coarse trace).

        Returns:
            dedup_map: Maps unique_index → list of original indices.
            unique_failures: Deduplicated failure list.
        """
        seen: dict[str, int] = {}  # fingerprint → unique_index
        dedup_map: dict[int, list[int]] = {}  # unique_index → [orig_indices]
        unique_failures: list[FailureInstance] = []

        for orig_idx, f in enumerate(failures):
            fingerprint = (
                f"{f.exception_type}|||"
                f"{f.exception_message}|||"
                f"{self._build_coarse_trace_view(f)}"
            )
            if fingerprint in seen:
                uid = seen[fingerprint]
                dedup_map[uid].append(orig_idx)
            else:
                uid = len(unique_failures)
                seen[fingerprint] = uid
                dedup_map[uid] = [orig_idx]
                unique_failures.append(f)

        return dedup_map, unique_failures

    @staticmethod
    def _expand_labels(
        unique_labels: np.ndarray,
        dedup_map: dict[int, list[int]],
        total_count: int,
    ) -> np.ndarray:
        """Expand labels from unique failures back to all failures."""
        expanded = np.full(total_count, -1, dtype=int)
        for uid, orig_indices in dedup_map.items():
            for idx in orig_indices:
                expanded[idx] = int(unique_labels[uid])
        return expanded

    def _labels_to_clusters(
        self,
        labels: np.ndarray,
        failures: list[FailureInstance],
    ) -> tuple[list[Cluster], list[str]]:
        """Convert integer labels to Cluster objects.

        Failures with label -1 are treated as noise.
        """
        groups: dict[int, list[FailureInstance]] = defaultdict(list)
        noise_ids: list[str] = []

        for f, label in zip(failures, labels):
            if label == -1:
                noise_ids.append(f.failure_id)
            else:
                groups[label].append(f)

        # If HDBSCAN put everything in noise, fall back to single cluster
        if not groups and noise_ids:
            return [self._create_cluster(failures)], []

        clusters = [self._create_cluster(group) for group in groups.values()]
        return clusters, noise_ids


def run_baseline_comparison(
    failures: list[FailureInstance],
    ground_truth_labels: list[int] | None = None,
    changed_files: list[str] | None = None,
    file_diffs: dict[str, str] | None = None,
    changed_methods_count: int | None = None,
    include_gptrace: bool = False,
    openai_client: object | None = None,
) -> dict:
    """Run all baselines and return clustering results.

    Args:
        failures: List of failure instances.
        ground_truth_labels: Optional labels for evaluation.
        changed_files: Optional list of changed files for BuildSheriff.
        file_diffs: Optional dict of file diffs for BuildSheriff.
        changed_methods_count: Optional count of changed methods for BuildSheriff.
        include_gptrace: Whether to include GPTrace baseline (requires OpenAI API).
        openai_client: Optional pre-configured OpenAIClient for GPTrace.

    Returns:
        Dict with results from each baseline.
    """
    baselines: dict[str, BaselineClusterer] = {
        "naive_type": NaiveClusterer(granularity="type"),
        "naive_message": NaiveClusterer(granularity="message"),
        "rebucket": RebucketClusterer(),
        "pw_overlap": PWOverlapClusterer(),
        "tracesim": TraceSimOfficialClusterer(),
        "buildsheriff_lite": BuildSheriffLiteClusterer(),
    }
    
    # Add full BuildSheriff if changes are provided
    if changed_files:
        buildsheriff = BuildSheriffClusterer()
        buildsheriff.set_changes(changed_files, file_diffs, changed_methods_count)
        baselines["buildsheriff"] = buildsheriff

    # Add GPTrace if requested (requires OpenAI API for embeddings)
    if include_gptrace:
        baselines["gptrace"] = GPTraceClusterer(openai_client=openai_client)

    results = {}
    for name, clusterer in baselines.items():
        clusters, noise = clusterer.cluster(failures)
        results[name] = {
            "num_clusters": len(clusters),
            "num_noise": len(noise),
            "clusters": clusters,
        }

    return results

