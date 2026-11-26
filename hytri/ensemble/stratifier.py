"""Failure Type Stratifier.

Implements failure type stratification from BuildSheriff (ICSE 2022):
Different failure types need different similarity signals.

- Exception failures: Have meaningful stack traces, use stack similarity + change overlap
- Assertion failures: Stack traces just show assert line, use test name + message similarity

This prevents assertion failures from polluting exception clustering and vice versa.
"""

from dataclasses import dataclass, field

from hytri.models import FailureInstance


@dataclass
class StratifiedFailures:
    """Container for stratified failure groups."""
    
    exception_failures: list[FailureInstance] = field(default_factory=list)
    assertion_failures: list[FailureInstance] = field(default_factory=list)
    
    @property
    def total_count(self) -> int:
        """Total number of failures across all strata."""
        return len(self.exception_failures) + len(self.assertion_failures)
    
    @property
    def has_exceptions(self) -> bool:
        """Whether there are any exception failures."""
        return len(self.exception_failures) > 0
    
    @property
    def has_assertions(self) -> bool:
        """Whether there are any assertion failures."""
        return len(self.assertion_failures) > 0


class FailureStratifier:
    """Split failures into exception vs assertion categories.
    
    From BuildSheriff paper:
    - Exception failures have meaningful stack traces (NPE, RuntimeError, etc.)
    - Assertion failures have trivial stacks (just the assert line)
    
    Each type needs different similarity signals:
    - Exceptions: Stack trace similarity, change overlap, semantic
    - Assertions: Test name similarity, message similarity, semantic
    """
    
    # Known assertion exception types (case-insensitive matching)
    ASSERTION_TYPES = {
        # Java
        'assertionerror',
        'java.lang.assertionerror',
        'junit.framework.assertionfailederror',
        'org.junit.comparionfailure',
        'org.junit.assertionerror',
        'org.opentest4j.assertionfailederror',
        # Python
        'assertionerror',
        'pytest.fail',
        'unittest.case._assertionerror',
        # Generic
        'assertion',
        'assert',
    }
    
    # Keywords that indicate assertion-like failures
    ASSERTION_KEYWORDS = {
        'expected',
        'but was',
        'assert',
        'should be',
        'should equal',
        'to equal',
        'to be',
        'not equal',
    }
    
    def __init__(
        self,
        custom_assertion_types: set[str] | None = None,
        min_frames_for_exception: int = 3,
    ):
        """Initialize the stratifier.
        
        Args:
            custom_assertion_types: Additional assertion type names to recognize.
            min_frames_for_exception: Minimum stack frames to consider informative.
        """
        self.assertion_types = self.ASSERTION_TYPES.copy()
        if custom_assertion_types:
            self.assertion_types.update(t.lower() for t in custom_assertion_types)
        self.min_frames_for_exception = min_frames_for_exception
    
    def stratify(self, failures: list[FailureInstance]) -> StratifiedFailures:
        """Split failures into exception and assertion categories.
        
        Args:
            failures: List of failure instances to stratify.
            
        Returns:
            StratifiedFailures with separated lists.
        """
        result = StratifiedFailures()
        
        for failure in failures:
            if self._is_assertion_failure(failure):
                result.assertion_failures.append(failure)
            else:
                result.exception_failures.append(failure)
        
        return result
    
    def _is_assertion_failure(self, failure: FailureInstance) -> bool:
        """Determine if a failure is an assertion failure.
        
        Uses multiple signals:
        1. Exception type name contains assertion keywords
        2. Exception message contains assertion patterns
        3. Stack trace is too short to be informative
        4. No real exception (test just failed without throwing)
        
        Args:
            failure: The failure instance to classify.
            
        Returns:
            True if this is an assertion failure.
        """
        exc_type = (failure.exception_type or "").lower()
        exc_message = (failure.exception_message or "").lower()
        
        # Check 1: Exception type is a known assertion type
        for assertion_type in self.assertion_types:
            if assertion_type in exc_type:
                return True
        
        # Check 2: Exception message has assertion patterns
        for keyword in self.ASSERTION_KEYWORDS:
            if keyword in exc_message:
                return True
        
        # Check 3: No exception type and no meaningful stack
        if not exc_type and len(failure.frames) < self.min_frames_for_exception:
            return True
        
        # Check 4: Exception type is explicitly "assertion" related
        if 'assert' in exc_type:
            return True
        
        # Default: treat as exception failure (has stack trace info)
        return False
    
    def get_failure_type(self, failure: FailureInstance) -> str:
        """Get the failure type category for a single failure.
        
        Args:
            failure: The failure instance to classify.
            
        Returns:
            "assertion" or "exception"
        """
        return "assertion" if self._is_assertion_failure(failure) else "exception"
    
    def get_informative_frame_count(self, failure: FailureInstance) -> int:
        """Count informative (non-trivial) stack frames.
        
        Filters out:
        - Frames with unknown/empty file
        - Test framework internal frames
        - JUnit/pytest infrastructure frames
        
        Args:
            failure: The failure instance to analyze.
            
        Returns:
            Number of informative frames.
        """
        informative = 0
        
        # Common test framework packages to filter
        framework_patterns = {
            'junit', 'org.junit', 'pytest', 'unittest',
            'java.lang.reflect', 'sun.reflect',
            '_pytest', 'pluggy',
        }
        
        for frame in failure.frames:
            # Skip frames without file info
            if not frame.file or frame.file.lower() in ('unknown', ''):
                continue
            
            # Skip test framework internals
            func_lower = (frame.func or "").lower()
            file_lower = frame.file.lower()
            
            is_framework = any(
                pattern in func_lower or pattern in file_lower
                for pattern in framework_patterns
            )
            
            if not is_framework:
                informative += 1
        
        return informative
