"""Position-Dependent Model (PDM) Similarity.

Implements the PDM similarity from ReBucket (Dang et al., ICSE 2012):
"ReBucket: A Method for Clustering Duplicate Crash Reports Based on Call Stack Similarity"

Key innovations:
1. Position Dependent Model: Frames near crash point matter more
2. Alignment offset weighting: Matched frames should appear at similar depths
3. DP-based subsequence matching: Finds optimal common frame alignment

The similarity function uses:
    Q(Li) = Σk exp(-c·min(pos1,pos2)) · exp(-o·|pos1-pos2|)
    sim(C1, C2) = max{Li} Q(Li)

Where:
    - c = weight for distance from crash (position decay)
    - o = weight for alignment offset (penalizes misaligned matches)
"""

import math
import re
from dataclasses import dataclass

import numpy as np

from hytri.models import FailureInstance


@dataclass
class PDMConfig:
    """Configuration for PDM similarity computation."""
    
    c: float = 0.1  # Position decay weight (higher = faster decay from top)
    o: float = 0.1  # Offset penalty weight (higher = stricter alignment)
    remove_recursive: bool = True  # Remove recursive call patterns
    max_frames: int = 20  # Maximum frames to consider
    
    # Frame filtering
    filter_test_frameworks: bool = True
    filter_common_libraries: bool = True


class PDMSimilarity:
    """Position-Dependent Model similarity for stack traces.
    
    From ReBucket paper, this captures:
    - Frames near the crash point are more diagnostic
    - Similar crashes should have similar frame positions
    - Recursive calls should be collapsed
    """
    
    # Common test/library frames to filter
    FILTER_PATTERNS = {
        # Java test frameworks
        'junit', 'org.junit', 'testng', 'org.testng',
        'java.lang.reflect', 'sun.reflect',
        # Python test frameworks
        'pytest', '_pytest', 'unittest', 'pluggy',
        # Common libraries
        'java.util.concurrent', 'java.lang.Thread',
    }
    
    def __init__(self, config: PDMConfig | None = None):
        """Initialize PDM similarity.
        
        Args:
            config: PDM configuration parameters.
        """
        self.config = config or PDMConfig()
    
    def compute_similarity_matrix(
        self,
        failures: list[FailureInstance],
    ) -> np.ndarray:
        """Compute PDM similarity matrix for failures.
        
        Args:
            failures: List of failure instances.
            
        Returns:
            NxN similarity matrix.
        """
        n = len(failures)
        sim_matrix = np.zeros((n, n))
        
        # Extract and preprocess frame signatures
        frame_lists = [self._extract_frames(f) for f in failures]
        
        # Compute pairwise PDM similarity
        for i in range(n):
            sim_matrix[i, i] = 1.0
            for j in range(i + 1, n):
                sim = self._pdm_similarity(frame_lists[i], frame_lists[j])
                sim_matrix[i, j] = sim
                sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def compute_pairwise(
        self,
        failure1: FailureInstance,
        failure2: FailureInstance,
    ) -> float:
        """Compute PDM similarity between two failures.
        
        Args:
            failure1: First failure.
            failure2: Second failure.
            
        Returns:
            Similarity score in [0, 1].
        """
        frames1 = self._extract_frames(failure1)
        frames2 = self._extract_frames(failure2)
        return self._pdm_similarity(frames1, frames2)
    
    def _extract_frames(self, failure: FailureInstance) -> list[str]:
        """Extract and preprocess frame signatures from a failure.
        
        Args:
            failure: The failure instance.
            
        Returns:
            List of frame signature strings.
        """
        frames = []
        
        for frame in failure.frames[:self.config.max_frames]:
            # Build frame signature
            sig = self._frame_to_signature(frame.file, frame.func)
            if sig:
                frames.append(sig)
        
        # Apply preprocessing
        if self.config.filter_test_frameworks or self.config.filter_common_libraries:
            frames = self._filter_frames(frames)
        
        if self.config.remove_recursive:
            frames = self._remove_recursive_calls(frames)
        
        return frames
    
    def _frame_to_signature(self, file: str, func: str) -> str | None:
        """Convert frame file and function to signature.
        
        Args:
            file: File path or name.
            func: Function/method name.
            
        Returns:
            Frame signature string or None if invalid.
        """
        # Handle missing info
        if not file and not func:
            return None
        
        # Normalize file path
        if file:
            # Take just the filename, strip extension
            file = file.split('/')[-1].split('\\')[-1]
            file = re.sub(r'\.(java|py|js|ts|cpp|c|h)$', '', file)
        else:
            file = "Unknown"
        
        # Normalize function name
        if func:
            # Strip line numbers and common prefixes
            func = re.sub(r':\d+$', '', func)
            # For Java, might have full package path
            if '.' in func:
                parts = func.split('.')
                # Keep class.method if present
                if len(parts) >= 2:
                    func = '.'.join(parts[-2:])
        else:
            func = "unknown"
        
        return f"{file}:{func}"
    
    def _filter_frames(self, frames: list[str]) -> list[str]:
        """Filter out test framework and library frames.
        
        Args:
            frames: List of frame signatures.
            
        Returns:
            Filtered list.
        """
        result = []
        for frame in frames:
            frame_lower = frame.lower()
            is_filtered = any(
                pattern in frame_lower
                for pattern in self.FILTER_PATTERNS
            )
            if not is_filtered:
                result.append(frame)
        return result
    
    def _remove_recursive_calls(self, frames: list[str]) -> list[str]:
        """Remove repeated recursive calls, keeping only one occurrence.
        
        Uses Brodie et al.'s algorithm: collapse consecutive and A->B->A patterns.
        
        Args:
            frames: List of frame signatures.
            
        Returns:
            Deduplicated list.
        """
        if not frames:
            return frames
        
        result = []
        
        for frame in frames:
            # Skip consecutive duplicates
            if result and frame == result[-1]:
                continue
            
            # Skip A->B->A patterns (recursive through another function)
            if len(result) >= 2 and frame == result[-2]:
                continue
            
            result.append(frame)
        
        return result
    
    def _pdm_similarity(self, frames1: list[str], frames2: list[str]) -> float:
        """Compute Position Dependent Model similarity using DP.
        
        Finds the best-matching common subsequence and scores it:
            Q(Li) = Σk exp(-c·min(pos1,pos2)) · exp(-o·|pos1-pos2|)
        
        Uses dynamic programming inspired by LCS to find optimal alignment.
        
        Args:
            frames1: First stack's frames.
            frames2: Second stack's frames.
            
        Returns:
            Similarity score in [0, 1].
        """
        if not frames1 and not frames2:
            return 1.0
        if not frames1 or not frames2:
            return 0.0
        
        n, m = len(frames1), len(frames2)
        c = self.config.c
        o = self.config.o
        
        # DP matrix: M[i][j] = best similarity score using frames1[:i] and frames2[:j]
        M = np.zeros((n + 1, m + 1))
        
        # Fill DP matrix
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                # Option 1: Don't match frame i with frame j
                M[i][j] = max(M[i-1][j], M[i][j-1])
                
                # Option 2: Match frame i-1 with frame j-1 (if they're the same)
                if frames1[i-1] == frames2[j-1]:
                    # Position indices (0-indexed from top of stack)
                    pos1 = i - 1
                    pos2 = j - 1
                    
                    # PDM score for this match:
                    # exp(-c * min(pos1, pos2)) * exp(-o * |pos1 - pos2|)
                    position_weight = math.exp(-c * min(pos1, pos2))
                    offset_penalty = math.exp(-o * abs(pos1 - pos2))
                    match_score = position_weight * offset_penalty
                    
                    # Add this match to the best previous score
                    M[i][j] = max(M[i][j], M[i-1][j-1] + match_score)
        
        # The final similarity is the best score found
        best_score = M[n][m]
        
        # Normalize by the maximum possible score
        # Max score would be if one stack is a prefix of the other, all frames matching
        max_possible = sum(math.exp(-c * k) for k in range(min(n, m)))
        
        if max_possible > 0:
            return best_score / max_possible
        return 0.0
    
    def get_confidence(self, failure: FailureInstance) -> float:
        """Get confidence score for using PDM on this failure.
        
        Higher confidence when:
        - More informative frames available
        - Frames have good file/function info
        
        Args:
            failure: The failure instance.
            
        Returns:
            Confidence score in [0, 1].
        """
        frames = self._extract_frames(failure)
        
        if not frames:
            return 0.0
        
        # Base confidence on number of useful frames
        # Saturates around 5-10 frames
        frame_confidence = min(len(frames) / 5.0, 1.0)
        
        # Check quality of frames (how many have good info)
        good_frames = sum(
            1 for f in frames
            if 'Unknown' not in f and 'unknown' not in f
        )
        quality_ratio = good_frames / len(frames) if frames else 0.0
        
        return frame_confidence * quality_ratio
