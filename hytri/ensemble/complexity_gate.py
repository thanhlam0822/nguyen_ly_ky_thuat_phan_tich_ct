"""Complexity Gate - BuildSheriff S1 Rule.

Implements the complexity check from BuildSheriff (ICSE 2022):
If the number of changed methods is below a threshold, assume all failures
share a single root cause and skip clustering.

This simple heuristic handles ~40% of CI builds perfectly because most builds
have few changes, and few changes typically mean a single root cause.
"""

import re
from dataclasses import dataclass


@dataclass
class ComplexityAnalysis:
    """Result of complexity analysis."""
    
    changed_files_count: int
    estimated_changed_methods: int
    should_single_cluster: bool
    reason: str


class ComplexityGate:
    """BuildSheriff S1 rule: Few changes = single root cause.
    
    From the BuildSheriff paper (ICSE 2022):
    - Δe = 7: Threshold for exception failures
    - Δa = 7: Threshold for assertion failures
    
    If the estimated number of changed methods is <= threshold,
    all failures are placed in a single cluster.
    
    NOTE: This rule works well when failures share a root cause,
    but can over-merge when multiple independent bugs happen to
    have few total changes. Use with caution on datasets with
    high-quality stack traces (like BugSwarm) where lexical methods
    already achieve near-perfect clustering.
    """
    
    # Paper-specified thresholds
    DELTA_EXCEPTION = 7
    DELTA_ASSERTION = 7
    
    # Heuristics for estimating method count from diffs
    METHODS_PER_FILE_ESTIMATE = 2  # Conservative estimate when no diff available
    
    # Minimum files to trigger single-cluster (safety check)
    MIN_FILES_FOR_SINGLE_CLUSTER = 1
    MAX_FILES_FOR_SINGLE_CLUSTER = 3  # If >3 files changed, don't single-cluster
    
    def __init__(
        self,
        delta_exception: int = 7,
        delta_assertion: int = 7,
        max_files_for_single_cluster: int = 3,
        require_file_overlap: bool = True,
    ):
        """Initialize the complexity gate.
        
        Args:
            delta_exception: Threshold for exception failures.
            delta_assertion: Threshold for assertion failures.
            max_files_for_single_cluster: Maximum changed files to allow single-cluster.
                                         Prevents over-merging when many files changed.
            require_file_overlap: If True, only single-cluster if diffs touch
                                 overlapping packages/directories.
        """
        self.delta_exception = delta_exception
        self.delta_assertion = delta_assertion
        self.max_files_for_single_cluster = max_files_for_single_cluster
        self.require_file_overlap = require_file_overlap
    
    def analyze(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> ComplexityAnalysis:
        """Analyze build complexity to determine if single-cluster is appropriate.
        
        Args:
            changed_files: List of changed file paths.
            file_diffs: Optional dict mapping file path to diff content.
            
        Returns:
            ComplexityAnalysis with the decision and reasoning.
        """
        if not changed_files:
            return ComplexityAnalysis(
                changed_files_count=0,
                estimated_changed_methods=0,
                should_single_cluster=True,
                reason="No changed files - likely infrastructure or external issue",
            )
        
        # Safety check: Don't single-cluster if many files changed
        # This prevents over-merging when multiple independent bugs are present
        if len(changed_files) > self.max_files_for_single_cluster:
            return ComplexityAnalysis(
                changed_files_count=len(changed_files),
                estimated_changed_methods=len(changed_files) * self.METHODS_PER_FILE_ESTIMATE,
                should_single_cluster=False,
                reason=f"Too many files changed ({len(changed_files)} > {self.max_files_for_single_cluster}) - proceeding with clustering",
            )
        
        # Check if files are in the same package/directory (require overlap)
        if self.require_file_overlap and len(changed_files) > 1:
            if not self._files_share_package(changed_files):
                return ComplexityAnalysis(
                    changed_files_count=len(changed_files),
                    estimated_changed_methods=len(changed_files) * self.METHODS_PER_FILE_ESTIMATE,
                    should_single_cluster=False,
                    reason=f"Changed files span multiple packages - likely multiple root causes",
                )
        
        # Estimate changed methods
        if file_diffs:
            estimated_methods = self._estimate_changed_methods_from_diffs(file_diffs)
        else:
            # Fallback: estimate based on file count
            estimated_methods = len(changed_files) * self.METHODS_PER_FILE_ESTIMATE
        
        # Use the more conservative threshold (exception)
        threshold = min(self.delta_exception, self.delta_assertion)
        should_single = estimated_methods <= threshold
        
        if should_single:
            reason = (
                f"Few changes detected ({estimated_methods} methods in "
                f"{len(changed_files)} files) - assuming single root cause"
            )
        else:
            reason = (
                f"Many changes detected ({estimated_methods} methods in "
                f"{len(changed_files)} files) - proceeding with clustering"
            )
        
        return ComplexityAnalysis(
            changed_files_count=len(changed_files),
            estimated_changed_methods=estimated_methods,
            should_single_cluster=should_single,
            reason=reason,
        )
    
    def _files_share_package(self, files: list[str]) -> bool:
        """Check if all files share at least one common parent directory.
        
        This helps detect when changes span multiple unrelated areas.
        """
        if len(files) <= 1:
            return True
        
        # Extract parent directories (2 levels deep)
        def get_parents(path: str) -> set[str]:
            parts = path.replace('\\', '/').split('/')
            parents = set()
            for i in range(1, min(len(parts), 4)):  # Up to 3 levels
                parents.add('/'.join(parts[:i]))
            return parents
        
        # Check if any common parent exists
        common = get_parents(files[0])
        for f in files[1:]:
            common &= get_parents(f)
        
        return len(common) > 0
    
    def should_single_cluster(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> bool:
        """Quick check if failures should be placed in a single cluster.
        
        Args:
            changed_files: List of changed file paths.
            file_diffs: Optional dict mapping file path to diff content.
            
        Returns:
            True if all failures should be in one cluster.
        """
        return self.analyze(changed_files, file_diffs).should_single_cluster
    
    def _estimate_changed_methods_from_diffs(
        self,
        file_diffs: dict[str, str],
    ) -> int:
        """Estimate number of changed methods from diff content.
        
        Parses diffs to find method/function signatures that were modified.
        
        Args:
            file_diffs: Dict mapping file path to diff content.
            
        Returns:
            Estimated number of changed methods.
        """
        total_methods = 0
        
        for file_path, diff_content in file_diffs.items():
            if not diff_content:
                total_methods += 1  # Assume at least one change
                continue
            
            # Detect language and count appropriately
            if file_path.endswith('.java'):
                total_methods += self._count_java_methods(diff_content)
            elif file_path.endswith('.py'):
                total_methods += self._count_python_methods(diff_content)
            else:
                # Generic: count @@ hunks as rough method estimate
                total_methods += self._count_diff_hunks(diff_content)
        
        return max(total_methods, len(file_diffs))  # At least one per file
    
    def _count_java_methods(self, diff_content: str) -> int:
        """Count Java method changes in a diff."""
        count = 0
        
        # Look for added/modified method signatures
        # Patterns: public/private/protected + return type + method name
        method_pattern = r'^\+\s*(public|private|protected)?\s*\w+\s+\w+\s*\('
        count += len(re.findall(method_pattern, diff_content, re.MULTILINE))
        
        # Also count @@ context lines that mention methods
        hunk_pattern = r'@@.*@@.*\b\w+\s*\([^)]*\)'
        count += len(re.findall(hunk_pattern, diff_content))
        
        return max(count, 1)
    
    def _count_python_methods(self, diff_content: str) -> int:
        """Count Python function/method changes in a diff."""
        count = 0
        
        # Look for added def statements
        def_pattern = r'^\+\s*def\s+\w+'
        count += len(re.findall(def_pattern, diff_content, re.MULTILINE))
        
        # Also count @@ context lines that mention functions
        hunk_pattern = r'@@.*@@.*def\s+\w+'
        count += len(re.findall(hunk_pattern, diff_content))
        
        return max(count, 1)
    
    def _count_diff_hunks(self, diff_content: str) -> int:
        """Count diff hunks as a rough method estimate."""
        # Each @@ hunk typically represents a change in one logical unit
        hunk_count = diff_content.count('@@ ')
        return max(hunk_count, 1)
