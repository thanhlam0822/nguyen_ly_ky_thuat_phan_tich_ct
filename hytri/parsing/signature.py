"""Component A2: Failure Signature Tokenizer.

Creates stable signature tokens from failure instances for use in
lexical similarity matching.
"""

import re
from dataclasses import dataclass

from hytri.models import FailureInstance, StackFrame


@dataclass
class FailureSignature:
    """Structured signature extracted from a failure."""

    exception_type: str
    message_template: str
    top_frame_funcs: list[str]
    top_frame_modules: list[str]
    tokens: list[str]


class SignatureTokenizer:
    """Tokenizer for creating stable failure signatures.

    Produces tokens that can be used for BM25/TF-IDF similarity matching.
    Normalizes variable parts (numbers, UUIDs, etc.) to create stable templates.
    """

    # Patterns for normalization
    NUMBER_PATTERN = re.compile(r"\b\d+\b")
    UUID_PATTERN = re.compile(
        r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
    )
    HEX_PATTERN = re.compile(r"\b0x[0-9a-fA-F]+\b")
    MEMORY_ADDR_PATTERN = re.compile(r"at 0x[0-9a-fA-F]+")
    OBJECT_REPR_PATTERN = re.compile(r"<[^>]+object at 0x[0-9a-fA-F]+>")
    QUOTES_PATTERN = re.compile(r"['\"]([^'\"]+)['\"]")

    # Common canonical module names to preserve
    CANONICAL_MODULES = {
        "numpy",
        "pandas",
        "scipy",
        "sklearn",
        "torch",
        "tensorflow",
        "requests",
        "django",
        "flask",
        "pytest",
        "unittest",
    }

    def __init__(self, top_k_frames: int = 3, include_line_numbers: bool = False):
        """Initialize the tokenizer.

        Args:
            top_k_frames: Number of top frames to include in signature.
            include_line_numbers: Whether to include line numbers in tokens.
        """
        self.top_k_frames = top_k_frames
        self.include_line_numbers = include_line_numbers

    def tokenize(self, failure: FailureInstance) -> FailureSignature:
        """Create a signature from a failure instance.

        Args:
            failure: The failure instance to tokenize.

        Returns:
            FailureSignature with structured signature data.
        """
        # Normalize exception message to template
        message_template = self._normalize_message(failure.exception_message)

        # Get top frames
        top_frames = failure.frames[-self.top_k_frames :] if failure.frames else []
        top_frame_funcs = [f.func for f in top_frames]
        top_frame_modules = self._extract_modules(top_frames)

        # Build token list
        tokens = self._build_tokens(
            failure.exception_type,
            message_template,
            top_frame_funcs,
            top_frame_modules,
        )

        return FailureSignature(
            exception_type=failure.exception_type,
            message_template=message_template,
            top_frame_funcs=top_frame_funcs,
            top_frame_modules=top_frame_modules,
            tokens=tokens,
        )

    def _normalize_message(self, message: str) -> str:
        """Normalize exception message to a template.

        Replaces variable parts with placeholders for stable matching.
        """
        template = message

        # Replace object representations first (contains hex addresses)
        template = self.OBJECT_REPR_PATTERN.sub("<OBJECT>", template)

        # Replace memory addresses
        template = self.MEMORY_ADDR_PATTERN.sub("at <ADDR>", template)

        # Replace UUIDs
        template = self.UUID_PATTERN.sub("<UUID>", template)

        # Replace hex numbers
        template = self.HEX_PATTERN.sub("<HEX>", template)

        # Replace numbers (but preserve some context)
        template = self.NUMBER_PATTERN.sub("<NUM>", template)

        return template.strip()

    def _extract_modules(self, frames: list[StackFrame]) -> list[str]:
        """Extract module names from frame file paths."""
        modules = []
        for frame in frames:
            # Extract module from file path
            module = self._file_to_module(frame.file)
            if module:
                modules.append(module)
        return modules

    def _file_to_module(self, file_path: str) -> str:
        """Convert a file path to a module name."""
        # Remove .py extension
        path = file_path.replace(".py", "")

        # Get the last component (file/module name)
        parts = path.replace("\\", "/").split("/")
        if parts:
            return parts[-1]
        return ""

    def _build_tokens(
        self,
        exception_type: str,
        message_template: str,
        funcs: list[str],
        modules: list[str],
    ) -> list[str]:
        """Build the final token list for the signature."""
        tokens = []

        # Exception type is always first (high weight)
        tokens.append(f"EXC:{exception_type}")

        # Add message template tokens
        message_tokens = self._tokenize_text(message_template)
        for tok in message_tokens:
            tokens.append(f"MSG:{tok}")

        # Add function names
        for func in funcs:
            if func and func != "<module>":
                tokens.append(f"FUNC:{func}")

        # Add module names
        for module in modules:
            if module:
                tokens.append(f"MOD:{module}")

                # Add canonical module markers
                for canonical in self.CANONICAL_MODULES:
                    if canonical in module.lower():
                        tokens.append(f"LIB:{canonical}")

        return tokens

    def _tokenize_text(self, text: str) -> list[str]:
        """Split text into meaningful tokens."""
        # Extract quoted strings as separate tokens
        quoted = self.QUOTES_PATTERN.findall(text)

        # Remove quotes and split on whitespace/punctuation
        text_clean = self.QUOTES_PATTERN.sub(" ", text)
        words = re.split(r"[\s,;:!?\[\]{}()]+", text_clean)

        tokens = []
        for word in words:
            word = word.strip().lower()
            if word and len(word) > 1 and word not in {"<num>", "<uuid>", "<hex>", "<addr>"}:
                tokens.append(word)

        # Add quoted strings as tokens
        for q in quoted:
            q = q.strip().lower()
            if q and len(q) > 1:
                tokens.append(f"Q:{q}")

        return tokens

    def get_signature_string(self, failure: FailureInstance) -> str:
        """Get a single string signature for the failure.

        Useful for direct string matching or display.
        """
        sig = self.tokenize(failure)
        return " ".join(sig.tokens)

    def update_failure_signature(self, failure: FailureInstance) -> FailureInstance:
        """Update a failure instance with computed signature tokens.

        Modifies the failure in place and returns it.
        """
        sig = self.tokenize(failure)
        failure.normalized_message = sig.message_template
        failure.signature_tokens = sig.tokens
        return failure

    def batch_tokenize(
        self, failures: list[FailureInstance]
    ) -> list[FailureSignature]:
        """Tokenize a batch of failures."""
        return [self.tokenize(f) for f in failures]


