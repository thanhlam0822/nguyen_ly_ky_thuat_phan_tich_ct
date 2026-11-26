"""Multi-Signal Similarity Computation.

Computes ALL similarity signals for the ensemble approach:
1. S_pdm: Position-Dependent Model (ReBucket-style)
2. S_overlap: Direct file-frame overlap (BuildSheriff-style)
3. S_semantic: LLM embedding cosine similarity
4. S_message: Exception message token Levenshtein
5. S_hypothesis: LLM root cause hypothesis similarity
6. S_test_name: Test name token overlap (for assertions)

Each signal also provides a confidence score indicating how reliable
it is for the given set of failures.
"""

import re
from dataclasses import dataclass, field

import numpy as np

from hytri.models import FailureInstance
from hytri.ensemble.pdm import PDMSimilarity, PDMConfig
from hytri.similarity.semantic import SemanticSimilarity, FallbackSemanticSimilarity
from hytri.llm.openai_client import OpenAIClient


@dataclass
class SimilaritySignals:
    """Container for all computed similarity signals.
    
    Each signal is an NxN matrix where entry [i,j] is the similarity
    between failure i and failure j according to that signal.
    
    Confidences indicate how reliable each signal is (0-1).
    """
    
    # Core similarity matrices
    s_pdm: np.ndarray = field(default_factory=lambda: np.array([]))
    s_overlap: np.ndarray = field(default_factory=lambda: np.array([]))
    s_semantic: np.ndarray = field(default_factory=lambda: np.array([]))
    s_message: np.ndarray = field(default_factory=lambda: np.array([]))
    s_hypothesis: np.ndarray = field(default_factory=lambda: np.array([]))
    s_test_name: np.ndarray = field(default_factory=lambda: np.array([]))
    s_diff_line: np.ndarray = field(default_factory=lambda: np.array([]))  # NEW: Diff-line correlation
    
    # Confidence scores for each signal (0-1)
    confidences: dict[str, float] = field(default_factory=dict)
    
    # Number of failures
    n_failures: int = 0
    
    def get_signal(self, name: str) -> np.ndarray:
        """Get a signal matrix by name."""
        return getattr(self, f"s_{name}", np.array([]))
    
    def get_confidence(self, name: str) -> float:
        """Get confidence for a signal."""
        return self.confidences.get(name, 0.0)
    
    def get_all_signals(self) -> dict[str, np.ndarray]:
        """Get all non-empty signal matrices."""
        signals = {}
        for name in ['pdm', 'overlap', 'semantic', 'message', 'hypothesis', 'test_name', 'diff_line']:
            matrix = self.get_signal(name)
            if matrix.size > 0:
                signals[name] = matrix
        return signals
    
    def weighted_average(self, weights: dict[str, float] | None = None) -> np.ndarray:
        """Compute weighted average of all signals.
        
        Args:
            weights: Optional dict of signal_name -> weight.
                    If not provided, uses confidences as weights.
                    
        Returns:
            Weighted average similarity matrix.
        """
        if weights is None:
            weights = self.confidences
        
        signals = self.get_all_signals()
        if not signals:
            return np.array([])
        
        # Normalize weights
        total_weight = sum(weights.get(name, 0.0) for name in signals.keys())
        if total_weight == 0:
            total_weight = len(signals)
            weights = {name: 1.0 for name in signals.keys()}
        
        # Compute weighted average
        result = np.zeros_like(next(iter(signals.values())))
        for name, matrix in signals.items():
            w = weights.get(name, 0.0) / total_weight
            result += w * matrix
        
        return result


class MultiSignalComputer:
    """Computes all similarity signals for a set of failures.
    
    This is the core signal computation engine for the ensemble approach.
    It computes multiple complementary similarity measures and provides
    confidence scores for each.
    """
    
    def __init__(
        self,
        openai_client: OpenAIClient | None = None,
        use_semantic: bool = True,
        message_confidence_cap: float = 0.6,
        test_name_confidence_cap: float = 0.5,
        semantic_confidence: float = 0.7,
        hypothesis_confidence: float = 0.8,
        suspected_change_match_similarity: float = 0.9,
        diff_line_tolerance: int = 5,
        diff_line_confidence_cap: float = 0.9,
    ):
        """Initialize the signal computer.
        
        Args:
            openai_client: OpenAI client for LLM operations.
            use_semantic: Whether to compute semantic similarity.
            message_confidence_cap: Max confidence for message signal.
            test_name_confidence_cap: Max confidence for test-name signal.
            semantic_confidence: Fixed confidence for semantic (LLM) signal.
            hypothesis_confidence: Fixed confidence for hypothesis (LLM) signal.
            suspected_change_match_similarity: Similarity for exact suspected change match.
            diff_line_tolerance: Line-number tolerance for diff-line matching.
            diff_line_confidence_cap: Maximum confidence for diff-line signal.
        """
        self.openai_client = openai_client
        self.use_semantic = use_semantic
        self.message_confidence_cap = message_confidence_cap
        self.test_name_confidence_cap = test_name_confidence_cap
        self.semantic_confidence = semantic_confidence
        self.hypothesis_confidence = hypothesis_confidence
        self.suspected_change_match_similarity = suspected_change_match_similarity
        self.diff_line_tolerance = diff_line_tolerance
        self.diff_line_confidence_cap = diff_line_confidence_cap
        
        # Initialize component computers
        self.pdm_sim = PDMSimilarity(PDMConfig())
        
        if use_semantic and openai_client:
            self.semantic_sim = SemanticSimilarity(openai_client=openai_client)
        else:
            self.semantic_sim = FallbackSemanticSimilarity()
    
    def compute(
        self,
        failures: list[FailureInstance],
        changed_files: list[str] | None = None,
        file_diffs: dict[str, str] | None = None,
        hypotheses: list | None = None,
    ) -> SimilaritySignals:
        """Compute all similarity signals for failures.
        
        Args:
            failures: List of failure instances.
            changed_files: Optional list of changed files for overlap signal.
            file_diffs: Optional dict of file path -> diff content.
            hypotheses: Optional pre-computed hypotheses for hypothesis signal.
            
        Returns:
            SimilaritySignals with all computed matrices.
        """
        n = len(failures)
        if n == 0:
            return SimilaritySignals()
        
        signals = SimilaritySignals(n_failures=n)
        
        # 1. PDM similarity (stack trace position-dependent)
        signals.s_pdm = self.pdm_sim.compute_similarity_matrix(failures)
        signals.confidences['pdm'] = self._compute_pdm_confidence(failures)
        
        # 2. Direct file-frame overlap (BuildSheriff-style)
        if changed_files:
            signals.s_overlap = self._compute_overlap_similarity(failures, changed_files)
            signals.confidences['overlap'] = self._compute_overlap_confidence(
                failures, changed_files, signals.s_overlap
            )
        else:
            signals.s_overlap = np.zeros((n, n))
            signals.confidences['overlap'] = 0.0
        
        # 3. Semantic similarity (LLM embeddings)
        if self.use_semantic:
            signals.s_semantic = self.semantic_sim.compute_similarity_matrix(failures)
            signals.confidences['semantic'] = self.semantic_confidence
        else:
            signals.s_semantic = np.eye(n)
            signals.confidences['semantic'] = 0.0
        
        # 4. Exception message similarity
        signals.s_message = self._compute_message_similarity(failures)
        signals.confidences['message'] = self._compute_message_confidence(failures)
        
        # 5. Test name similarity
        signals.s_test_name = self._compute_test_name_similarity(failures)
        signals.confidences['test_name'] = self._compute_test_name_confidence(failures)
        
        # 6. Hypothesis similarity (computed externally, set here)
        if hypotheses:
            signals.s_hypothesis = self._compute_hypothesis_similarity(hypotheses)
            signals.confidences['hypothesis'] = self.hypothesis_confidence
        else:
            signals.s_hypothesis = np.zeros((n, n))
            signals.confidences['hypothesis'] = 0.0
        
        # 7. Diff-line correlation (NEW: direct stack-to-diff matching)
        if file_diffs and changed_files:
            signals.s_diff_line = self._compute_diff_line_similarity(failures, changed_files, file_diffs)
            signals.confidences['diff_line'] = self._compute_diff_line_confidence(
                failures, signals.s_diff_line
            )
        else:
            signals.s_diff_line = np.zeros((n, n))
            signals.confidences['diff_line'] = 0.0
        
        return signals
    
    def _compute_overlap_similarity(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
    ) -> np.ndarray:
        """Compute direct file-frame overlap similarity.
        
        This is the BuildSheriff-style signal: failures that touch
        the same changed files are likely related.
        """
        n = len(failures)
        sim_matrix = np.zeros((n, n))
        
        # Get base filenames/class names of changed files
        changed_base = set()
        for f in changed_files:
            basename = f.split('/')[-1].replace('.java', '').replace('.py', '')
            changed_base.add(basename.lower())
            if '.' in basename:
                changed_base.add(basename.split('.')[-1].lower())
        
        # For each failure, find which changed files it touches
        touched_files = []
        for failure in failures:
            touched = self._get_touched_files(failure, changed_base)
            touched_files.append(touched)
        
        # Compute Jaccard similarity of touched files
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                if touched_files[i] and touched_files[j]:
                    intersection = len(touched_files[i] & touched_files[j])
                    union = len(touched_files[i] | touched_files[j])
                    if union > 0 and intersection > 0:
                        sim = intersection / union
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _get_touched_files(
        self,
        failure: FailureInstance,
        changed_base: set[str],
    ) -> set[str]:
        """Get which changed files a failure touches."""
        touched = set()
        
        # Check parsed frames
        for frame in failure.frames:
            frame_file = frame.file.split('/')[-1].replace('.java', '').replace('.py', '') if frame.file else ""
            if frame_file and frame_file.lower() not in ('unknown', ''):
                for cf in changed_base:
                    if cf and (cf in frame_file.lower() or frame_file.lower() in cf):
                        touched.add(cf)
            
            # Check func name for class info
            if frame.func:
                parts = frame.func.split('.')
                for part in parts:
                    if part and len(part) > 2 and part[0].isupper():
                        for cf in changed_base:
                            if cf and (cf in part.lower() or part.lower() in cf):
                                touched.add(cf)
        
        # Parse raw stack_trace text
        stack_text = failure.stack_trace or ""
        if stack_text:
            at_patterns = re.findall(r'at\s+([\w.]+)', stack_text)
            for pattern in at_patterns:
                parts = pattern.split('.')
                for part in parts:
                    if part and len(part) > 2 and part[0].isupper():
                        for cf in changed_base:
                            if cf and (cf in part.lower() or part.lower() in cf):
                                touched.add(cf)
        
        return touched
    
    def _compute_message_similarity(
        self,
        failures: list[FailureInstance],
    ) -> np.ndarray:
        """Compute exception message token-level Levenshtein similarity."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))
        
        # Tokenize messages
        token_lists = []
        for f in failures:
            msg = f.exception_message or ""
            tokens = msg.lower().replace(':', ' ').replace('.', ' ').replace(',', ' ').split()
            token_lists.append(tokens)
        
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self._token_levenshtein_similarity(token_lists[i], token_lists[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _token_levenshtein_similarity(
        self,
        tokens1: list[str],
        tokens2: list[str],
    ) -> float:
        """Compute Levenshtein similarity at token level."""
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        m, n = len(tokens1), len(tokens2)
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
    
    def _compute_test_name_similarity(
        self,
        failures: list[FailureInstance],
    ) -> np.ndarray:
        """Compute test name token overlap similarity."""
        n = len(failures)
        sim_matrix = np.zeros((n, n))
        
        # Tokenize test names
        token_sets = []
        for f in failures:
            test_name = f.test_name or ""
            tokens = set(test_name.lower().replace('_', ' ').replace('::', ' ').replace('.', ' ').split())
            token_sets.append(tokens)
        
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                test_i = failures[i].test_name or ""
                test_j = failures[j].test_name or ""
                
                if test_i and test_j and test_i == test_j:
                    sim = 1.0
                elif token_sets[i] and token_sets[j]:
                    intersection = len(token_sets[i] & token_sets[j])
                    union = len(token_sets[i] | token_sets[j])
                    sim = intersection / union if union > 0 else 0.0
                else:
                    sim = 0.0
                
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _compute_hypothesis_similarity(self, hypotheses: list) -> np.ndarray:
        """Compute similarity based on LLM-generated hypotheses.
        
        Failures with similar suspected_change or mechanism are grouped.
        """
        n = len(hypotheses)
        sim_matrix = np.zeros((n, n))
        
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                h1, h2 = hypotheses[i], hypotheses[j]
                
                sim = 0.0
                
                # Same suspected change = high similarity
                if (hasattr(h1, 'suspected_change') and hasattr(h2, 'suspected_change') and
                    h1.suspected_change and h2.suspected_change and
                    h1.suspected_change == h2.suspected_change):
                    sim = self.suspected_change_match_similarity
                elif (hasattr(h1, 'mechanism') and hasattr(h2, 'mechanism') and
                      h1.mechanism and h2.mechanism):
                    # Compare mechanisms via token overlap
                    tokens1 = set(h1.mechanism.lower().split())
                    tokens2 = set(h2.mechanism.lower().split())
                    if tokens1 and tokens2:
                        intersection = len(tokens1 & tokens2)
                        union = len(tokens1 | tokens2)
                        sim = intersection / union if union > 0 else 0.0
                
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _compute_pdm_confidence(self, failures: list[FailureInstance]) -> float:
        """Compute confidence for PDM signal."""
        if not failures:
            return 0.0
        
        # Average confidence across failures
        confidences = [self.pdm_sim.get_confidence(f) for f in failures]
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _compute_overlap_confidence(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
        overlap_matrix: np.ndarray,
    ) -> float:
        """Compute confidence for overlap signal."""
        if not failures or not changed_files:
            return 0.0
        
        # High confidence if many failures have overlaps
        n = len(failures)
        non_zero_count = np.sum(overlap_matrix > 0) - n  # Exclude diagonal
        total_pairs = n * (n - 1)
        
        if total_pairs == 0:
            return 0.0
        
        overlap_ratio = non_zero_count / total_pairs
        
        # Also consider how many changed files there are
        # Few files = higher confidence in overlaps
        file_factor = min(1.0, 7 / max(len(changed_files), 1))
        
        return overlap_ratio * 0.5 + file_factor * 0.5
    
    def _compute_message_confidence(self, failures: list[FailureInstance]) -> float:
        """Compute confidence for message similarity signal."""
        if not failures:
            return 0.0
        
        # High confidence if messages are non-empty and varied
        non_empty = sum(1 for f in failures if f.exception_message)
        ratio = non_empty / len(failures)
        
        return ratio * self.message_confidence_cap
    
    def _compute_test_name_confidence(self, failures: list[FailureInstance]) -> float:
        """Compute confidence for test name similarity signal."""
        if not failures:
            return 0.0
        
        # High confidence if test names are non-empty
        non_empty = sum(1 for f in failures if f.test_name)
        ratio = non_empty / len(failures)
        
        return ratio * self.test_name_confidence_cap
    
    def _compute_diff_line_similarity(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
        file_diffs: dict[str, str],
    ) -> np.ndarray:
        """Compute similarity based on stack trace lines matching diff hunks.
        
        This is a direct correlation: if two failures have stack frames
        pointing to lines that were modified in the same diff hunk,
        they're likely caused by the same change.
        """
        n = len(failures)
        sim_matrix = np.zeros((n, n))
        
        # Parse all diff hunks to extract (file, line_start, line_end) ranges
        diff_hunks = self._parse_diff_hunks(changed_files, file_diffs)
        
        if not diff_hunks:
            return sim_matrix
        
        # For each failure, find which hunks it touches
        failure_hunks = []
        for failure in failures:
            hunks = self._get_matching_hunks(failure, diff_hunks)
            failure_hunks.append(hunks)
        
        # Compute similarity based on shared hunks
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                if failure_hunks[i] and failure_hunks[j]:
                    intersection = len(failure_hunks[i] & failure_hunks[j])
                    union = len(failure_hunks[i] | failure_hunks[j])
                    if intersection > 0:
                        # Strong signal: same diff hunk
                        sim = 0.5 + 0.5 * (intersection / union)
                    else:
                        # Check if they touch the same file at least
                        files_i = {h[0] for h in failure_hunks[i]}
                        files_j = {h[0] for h in failure_hunks[j]}
                        if files_i & files_j:
                            sim = 0.3
                        else:
                            sim = 0.0
                    sim_matrix[i, j] = sim
                    sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _parse_diff_hunks(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str],
    ) -> list[tuple[str, int, int]]:
        """Parse diffs to extract (file_base, line_start, line_end) tuples."""
        hunks = []
        
        for file_path in changed_files:
            if file_path not in file_diffs:
                continue
            
            diff_content = file_diffs[file_path]
            file_base = file_path.split('/')[-1].replace('.java', '').replace('.py', '').lower()
            
            # Parse @@ -old_start,old_count +new_start,new_count @@ patterns
            hunk_pattern = r'@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@'
            for match in re.finditer(hunk_pattern, diff_content):
                line_start = int(match.group(1))
                line_count = int(match.group(2)) if match.group(2) else 1
                line_end = line_start + line_count
                hunks.append((file_base, line_start, line_end))
        
        return hunks
    
    def _get_matching_hunks(
        self,
        failure: FailureInstance,
        diff_hunks: list[tuple[str, int, int]],
    ) -> set[tuple[str, int, int]]:
        """Find which diff hunks a failure's stack trace touches."""
        matching = set()
        
        for frame in failure.frames:
            if not frame.file or not frame.line:
                continue
            
            frame_base = frame.file.split('/')[-1].replace('.java', '').replace('.py', '').lower()
            if frame_base.lower() in ('unknown', ''):
                continue
            
            for hunk_file, hunk_start, hunk_end in diff_hunks:
                # Check if frame file matches hunk file
                if hunk_file in frame_base or frame_base in hunk_file:
                    # Check if frame line is in hunk range (with some tolerance)
                    if hunk_start - self.diff_line_tolerance <= frame.line <= hunk_end + self.diff_line_tolerance:
                        matching.add((hunk_file, hunk_start, hunk_end))
        
        # Also check raw stack_trace text
        if failure.stack_trace:
            # Look for patterns like "at com.foo.Bar.method(Bar.java:42)"
            line_pattern = r'(\w+)\.java:(\d+)'
            for match in re.finditer(line_pattern, failure.stack_trace):
                class_name = match.group(1).lower()
                line_num = int(match.group(2))
                
                for hunk_file, hunk_start, hunk_end in diff_hunks:
                    if hunk_file in class_name or class_name in hunk_file:
                        if hunk_start - self.diff_line_tolerance <= line_num <= hunk_end + self.diff_line_tolerance:
                            matching.add((hunk_file, hunk_start, hunk_end))
        
        return matching
    
    def _compute_diff_line_confidence(
        self,
        failures: list[FailureInstance],
        diff_line_matrix: np.ndarray,
    ) -> float:
        """Compute confidence for diff-line similarity signal."""
        if not failures:
            return 0.0
        
        n = len(failures)
        
        # Count how many failures have non-zero diff correlations
        has_correlation = 0
        for i in range(n):
            if np.any(diff_line_matrix[i, :i] > 0) or np.any(diff_line_matrix[i, i+1:] > 0):
                has_correlation += 1
        
        ratio = has_correlation / n
        
        # High confidence when many failures correlate with diffs
        return min(ratio * 1.5, self.diff_line_confidence_cap)