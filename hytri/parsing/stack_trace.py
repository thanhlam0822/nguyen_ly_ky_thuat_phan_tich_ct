"""Component A1: Stack Trace Parser.

Extracts structured frame information from raw stack traces.
Handles Python and Java trace formats and normalizes paths for consistent matching.
"""

import re
from dataclasses import dataclass

from hytri.models import StackFrame


@dataclass
class ParsedTrace:
    """Result of parsing a stack trace."""

    frames: list[StackFrame]
    failing_frame: StackFrame | None
    exception_type: str
    exception_message: str
    language: str = "python"


class StackTraceParser:
    """Parser for Python and Java stack traces.

    Extracts (file, function, line) tuples from traceback text,
    normalizes paths, and identifies the failing frame.
    Automatically detects language from trace format.
    """

    # Pattern for Python traceback frames
    # Matches: '  File "/path/to/file.py", line 42, in function_name'
    PYTHON_FRAME_PATTERN = re.compile(
        r'^\s*File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\S+)',
        re.MULTILINE,
    )

    # Pattern for Java stack trace frames
    # Matches: '    at org.package.Class.method(File.java:42)'
    # Also handles: 'at org.package.Class$Inner.method(File.java:42)'
    # And native methods: 'at java.lang.Thread.run(Unknown Source)'
    JAVA_FRAME_PATTERN = re.compile(
        r'^\s*at\s+([\w.$]+)\.([\w$<>]+)\(([^:)]+)(?::(\d+))?\)',
        re.MULTILINE,
    )
    
    # Pattern for Java exception line (first line of stack trace)
    # Matches: 'java.lang.NullPointerException: message'
    # Or: 'org.junit.ComparisonFailure: expected:<...> but was:<...>'
    JAVA_EXCEPTION_PATTERN = re.compile(
        r'^([\w.$]+(?:Exception|Error|Failure|Throwable))(?::\s*(.*))?$',
        re.MULTILINE,
    )

    # Pattern for exception line at the end
    # Matches: 'ExceptionType: message' or just 'ExceptionType'
    EXCEPTION_PATTERN = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)(?::\s*(.*))?$",
        re.MULTILINE,
    )

    # Common virtualenv/system path prefixes to strip
    PATH_PREFIXES_TO_STRIP = [
        r".*?/\.venv/lib/python[\d.]+/site-packages/",
        r".*?/venv/lib/python[\d.]+/site-packages/",
        r".*?/\.?virtualenvs/[^/]+/lib/python[\d.]+/site-packages/",
        r".*/lib/python[\d.]+/site-packages/",
        r".*/lib/python[\d.]+/",
        r".*/site-packages/",
    ]

    def __init__(self, top_k_frames: int = 5, normalize_paths: bool = True):
        """Initialize the parser.

        Args:
            top_k_frames: Maximum number of top frames to keep (from the exception).
            normalize_paths: Whether to strip virtualenv/system prefixes from paths.
        """
        self.top_k_frames = top_k_frames
        self.normalize_paths = normalize_paths
        self._path_prefix_patterns = [
            re.compile(p) for p in self.PATH_PREFIXES_TO_STRIP
        ]

    def parse(self, stack_trace: str, language: str | None = None) -> ParsedTrace:
        """Parse a raw stack trace string.

        Args:
            stack_trace: Raw stack trace text.
            language: Optional language hint ('python' or 'java'). Auto-detects if None.

        Returns:
            ParsedTrace with extracted frames and exception info.
        """
        # Auto-detect language if not provided
        if language is None:
            language = self._detect_language(stack_trace)

        frames = self._extract_frames(stack_trace, language)
        exception_type, exception_message = self._extract_exception(stack_trace, language)

        # In both Python and Java, the first/last frame is typically the failing frame
        # Python: last frame; Java: first frame (top of stack)
        if language == "java":
            failing_frame = frames[0] if frames else None
        else:
            failing_frame = frames[-1] if frames else None

        # Keep only top_k frames (closest to the exception)
        if len(frames) > self.top_k_frames:
            if language == "java":
                frames = frames[: self.top_k_frames]
            else:
                frames = frames[-self.top_k_frames :]

        return ParsedTrace(
            frames=frames,
            failing_frame=failing_frame,
            exception_type=exception_type,
            exception_message=exception_message,
            language=language,
        )

    def _detect_language(self, stack_trace: str) -> str:
        """Detect whether stack trace is Python or Java."""
        # Check for Java patterns
        if self.JAVA_FRAME_PATTERN.search(stack_trace):
            return "java"
        # Check for Python patterns
        if self.PYTHON_FRAME_PATTERN.search(stack_trace):
            return "python"
        # Default to Python
        return "python"

    def _extract_frames(self, stack_trace: str, language: str = "python") -> list[StackFrame]:
        """Extract frame tuples from stack trace."""
        frames = []

        if language == "java":
            for match in self.JAVA_FRAME_PATTERN.finditer(stack_trace):
                class_path = match.group(1)  # org.package.Class or org.package.Class$Inner
                method_name = match.group(2)  # method or <init>
                file_name = match.group(3)  # File.java or Unknown Source
                line_str = match.group(4)  # line number or None
                line_num = int(line_str) if line_str else 0

                # Skip native/unknown methods for cleaner analysis
                if file_name in ("Unknown Source", "Native Method"):
                    continue

                # Extract simple class name from full path for better matching
                # org.apache.commons.lang3.StringUtils -> StringUtils
                simple_class = class_path.split(".")[-1].split("$")[0]
                
                frames.append(
                    StackFrame(
                        file=file_name,
                        func=f"{simple_class}.{method_name}",
                        line=line_num,
                    )
                )
        else:
            for match in self.PYTHON_FRAME_PATTERN.finditer(stack_trace):
                file_path = match.group(1)
                line_num = int(match.group(2))
                func_name = match.group(3)

                if self.normalize_paths:
                    file_path = self._normalize_path(file_path)

                frames.append(
                    StackFrame(file=file_path, func=func_name, line=line_num)
                )

        return frames

    def _extract_exception(self, stack_trace: str, language: str = "python") -> tuple[str, str]:
        """Extract exception type and message from the trace."""
        lines = stack_trace.strip().split("\n")

        if language == "java":
            # Java: exception is typically at the START of the trace
            for line in lines:
                line = line.strip()
                if not line or line.startswith("at "):
                    continue
                
                match = self.JAVA_EXCEPTION_PATTERN.match(line)
                if match:
                    exc_type = match.group(1)
                    # Simplify: java.lang.NullPointerException -> NullPointerException
                    exc_type = exc_type.split(".")[-1]
                    exc_message = match.group(2) or ""
                    return exc_type, exc_message
            
            # Fallback: look for common JUnit assertion patterns
            full_trace = stack_trace.lower()
            if "assertionerror" in full_trace or "expected:" in full_trace:
                return "AssertionError", ""
            if "nullpointerexception" in full_trace:
                return "NullPointerException", ""
        else:
            # Python: exception is at the END of the trace
            for line in reversed(lines):
                line = line.strip()
                if not line:
                    continue

                # Skip lines that look like frame content
                if line.startswith("File ") or line.startswith("^"):
                    continue

                match = self.EXCEPTION_PATTERN.match(line)
                if match:
                    exc_type = match.group(1)
                    exc_message = match.group(2) or ""
                    return exc_type, exc_message

        return "UnknownException", ""

    def _normalize_path(self, path: str) -> str:
        """Strip common virtualenv/system prefixes from path."""
        for pattern in self._path_prefix_patterns:
            path = pattern.sub("", path)
        return path

    def parse_frames_only(self, stack_trace: str) -> list[StackFrame]:
        """Parse only the frames from a stack trace.

        Convenience method when you only need frames.
        """
        return self.parse(stack_trace).frames

    @staticmethod
    def frames_to_signature(frames: list[StackFrame]) -> str:
        """Convert frames to a signature string for matching.

        Returns a string like: "file1:func1 -> file2:func2 -> ..."
        """
        parts = [f"{f.file}:{f.func}" for f in frames]
        return " -> ".join(parts)


