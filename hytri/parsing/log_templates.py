"""Component A3: Log Template Mining.

Implements Drain-style log template mining for extracting patterns
from stderr/stdout logs. This helps group failures with similar log patterns.
"""

import re
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class LogTemplate:
    """A mined log template with its frequency."""

    template: str
    count: int = 1
    example_lines: list[str] = field(default_factory=list)


class LogTemplateMiner:
    """Drain-style log template miner.

    Uses a tree-based approach to efficiently group log lines into templates,
    replacing variable parts with wildcards.

    This is a simplified implementation of the Drain algorithm.
    Reference: He et al., "Drain: An Online Log Parsing Approach with Fixed Depth Tree"
    """

    # Default patterns to replace with wildcards
    DEFAULT_PATTERNS = [
        (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?"), "<TIMESTAMP>"),
        (re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"), "<IP>"),
        (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"), "<UUID>"),
        (re.compile(r"\b0x[0-9a-fA-F]+\b"), "<HEX>"),
        (re.compile(r"(?<=[^a-zA-Z])\d+(?=[^a-zA-Z])|\b\d+\b"), "<NUM>"),
        (re.compile(r"/[^\s]+"), "<PATH>"),
    ]

    def __init__(
        self,
        depth: int = 4,
        similarity_threshold: float = 0.4,
        max_children: int = 100,
        patterns: list[tuple[re.Pattern, str]] | None = None,
    ):
        """Initialize the miner.

        Args:
            depth: Maximum depth of the prefix tree.
            similarity_threshold: Minimum similarity to match a template.
            max_children: Maximum children per tree node.
            patterns: Custom regex patterns for pre-processing.
        """
        self.depth = depth
        self.similarity_threshold = similarity_threshold
        self.max_children = max_children
        self.patterns = patterns or self.DEFAULT_PATTERNS

        # Tree structure: {length: {first_token: {second_token: ...}}}
        self._tree: dict = defaultdict(lambda: defaultdict(dict))
        self._templates: dict[str, LogTemplate] = {}

    def preprocess(self, line: str) -> str:
        """Preprocess a log line by applying regex substitutions."""
        result = line.strip()
        for pattern, replacement in self.patterns:
            result = pattern.sub(replacement, result)
        return result

    def tokenize(self, line: str) -> list[str]:
        """Split a preprocessed line into tokens."""
        return line.split()

    def mine(self, log_text: str) -> list[LogTemplate]:
        """Mine templates from log text.

        Args:
            log_text: Raw log text (multiple lines).

        Returns:
            List of discovered log templates.
        """
        lines = log_text.strip().split("\n")

        for line in lines:
            if not line.strip():
                continue
            self._add_line(line)

        return list(self._templates.values())

    def _add_line(self, line: str) -> str:
        """Add a line to the template tree.

        Returns the matched/created template.
        """
        preprocessed = self.preprocess(line)
        tokens = self.tokenize(preprocessed)

        if not tokens:
            return ""

        # Find or create template
        template = self._match_or_create(tokens, line)
        return template

    def _match_or_create(self, tokens: list[str], original_line: str) -> str:
        """Match tokens to existing template or create new one."""
        length = len(tokens)

        # Navigate tree by length and prefix
        length_node = self._tree[length]

        # Use first few tokens as prefix (up to depth)
        prefix = tuple(tokens[: self.depth])

        # Look for matching template in this prefix group
        if prefix in length_node:
            candidates = length_node[prefix]
            for template_str, template in candidates.items():
                if self._similarity(tokens, template_str) >= self.similarity_threshold:
                    # Update template with new line
                    template.count += 1
                    if len(template.example_lines) < 3:
                        template.example_lines.append(original_line)
                    return template_str
        else:
            length_node[prefix] = {}

        # Create new template
        template_str = " ".join(tokens)
        new_template = LogTemplate(
            template=template_str,
            count=1,
            example_lines=[original_line],
        )
        length_node[prefix][template_str] = new_template
        self._templates[template_str] = new_template

        return template_str

    def _similarity(self, tokens: list[str], template_str: str) -> float:
        """Calculate similarity between tokens and a template."""
        template_tokens = template_str.split()

        if len(tokens) != len(template_tokens):
            return 0.0

        matches = sum(
            1 for t1, t2 in zip(tokens, template_tokens)
            if t1 == t2 or t1.startswith("<") or t2.startswith("<")
        )

        return matches / len(tokens)

    def get_templates(self) -> list[LogTemplate]:
        """Get all mined templates, sorted by frequency."""
        return sorted(
            self._templates.values(),
            key=lambda t: t.count,
            reverse=True,
        )

    def get_template_strings(self) -> list[str]:
        """Get just the template strings, sorted by frequency."""
        return [t.template for t in self.get_templates()]

    def reset(self) -> None:
        """Reset the miner state."""
        self._tree = defaultdict(lambda: defaultdict(dict))
        self._templates = {}


def extract_log_templates(
    stderr: str,
    stdout: str = "",
    max_templates: int = 10,
) -> list[str]:
    """Convenience function to extract log templates from failure output.

    Args:
        stderr: Captured stderr.
        stdout: Captured stdout.
        max_templates: Maximum number of templates to return.

    Returns:
        List of template strings.
    """
    miner = LogTemplateMiner()

    if stderr:
        miner.mine(stderr)
    if stdout:
        miner.mine(stdout)

    templates = miner.get_template_strings()
    return templates[:max_templates]


