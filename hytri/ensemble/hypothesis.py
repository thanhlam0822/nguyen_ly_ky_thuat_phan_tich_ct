"""LLM Root Cause Hypothesis Generator.

Uses LLM to generate root cause hypotheses for each failure.
This is where LLM excels: understanding the semantic connection
between code changes and test failures.

Key insight: Two failures with the same hypothesis
"Change to Parser.java broke UTF-8 handling" will cluster together
even if their stack traces look completely different.
"""

import json
from dataclasses import dataclass, field

from hytri.models import FailureInstance
from hytri.llm.openai_client import OpenAIClient


@dataclass
class Hypothesis:
    """Root cause hypothesis for a failure."""
    
    failure_id: str
    suspected_change: str | None  # Which file/change caused this
    mechanism: str  # How the change causes the failure
    confidence: float  # LLM's confidence (0-1)
    reasoning: str  # Detailed reasoning
    
    # Optional: categorization
    category: str = "unknown"  # e.g., "null_reference", "api_change", "config_error"
    affected_component: str = ""  # e.g., "parser", "database", "auth"


@dataclass
class HypothesisBatch:
    """Batch of hypotheses for a failure set."""
    
    hypotheses: list[Hypothesis] = field(default_factory=list)
    
    def get_by_failure_id(self, failure_id: str) -> Hypothesis | None:
        """Get hypothesis for a specific failure."""
        for h in self.hypotheses:
            if h.failure_id == failure_id:
                return h
        return None
    
    def get_by_suspected_change(self, change: str) -> list[Hypothesis]:
        """Get all hypotheses pointing to a specific change."""
        return [h for h in self.hypotheses if h.suspected_change == change]
    
    def get_change_groups(self) -> dict[str, list[str]]:
        """Group failure IDs by their suspected change."""
        groups: dict[str, list[str]] = {}
        for h in self.hypotheses:
            change = h.suspected_change or "unknown"
            if change not in groups:
                groups[change] = []
            groups[change].append(h.failure_id)
        return groups


class HypothesisGenerator:
    """Generate root cause hypotheses for failures using LLM.
    
    This leverages LLM's semantic understanding to:
    1. Analyze what each code change does
    2. Understand how failures manifest
    3. Connect changes to failures even without direct file overlap
    """
    
    # Prompt template for hypothesis generation (with diff content)
    HYPOTHESIS_PROMPT = """You are an expert software engineer analyzing test failures.

Given a test failure and the code changes in this build, generate a root cause hypothesis.

FAILURE:
- Test: {test_name}
- Exception: {exception_type}
- Message: {exception_message}
- Stack trace (top frames):
{stack_frames}

CHANGED FILES AND THEIR DIFFS:
{changed_files_with_diffs}

Analyze which specific code change most likely caused this failure and explain the mechanism.
Pay attention to:
- Which changed lines are mentioned in the stack trace
- What behavior the diff modifies (e.g., null checks, API changes, config)
- How the change could lead to this exception type

Respond in JSON format:
{{
    "suspected_change": "path/to/file.java or null if unclear",
    "suspected_line_range": "e.g., 45-52 or null if unclear",
    "mechanism": "Brief explanation of how the change causes this failure (1-2 sentences)",
    "confidence": 0.0 to 1.0,
    "reasoning": "Detailed reasoning for your hypothesis",
    "category": "One of: null_reference, api_change, config_error, type_error, assertion_mismatch, resource_error, concurrency, unknown",
    "affected_component": "The component/module affected (e.g., parser, database, auth)"
}}

If you cannot determine the cause, set suspected_change to null and explain in reasoning."""

    # Batch prompt for efficiency (with diff content)
    BATCH_PROMPT = """You are an expert software engineer analyzing test failures.

Given multiple test failures from the same build, generate root cause hypotheses for each.

CHANGED FILES AND THEIR DIFFS:
{changed_files_with_diffs}

FAILURES:
{failures}

For EACH failure, analyze which specific code change most likely caused it.
Pay attention to:
- Which changed lines appear in each stack trace
- What behavior each diff modifies
- How different changes could cause different failures

Respond in JSON format (array of objects):
[
    {{
        "failure_id": "the failure ID",
        "suspected_change": "path/to/file.java or null if unclear",
        "suspected_line_range": "e.g., 45-52 or null",
        "mechanism": "Brief explanation (1-2 sentences)",
        "confidence": 0.0 to 1.0,
        "reasoning": "Detailed reasoning",
        "category": "null_reference|api_change|config_error|type_error|assertion_mismatch|resource_error|concurrency|unknown",
        "affected_component": "component name"
    }},
    ...
]

Group failures that share the same root cause with the same suspected_change."""

    def __init__(
        self,
        openai_client: OpenAIClient | None = None,
        batch_size: int = 10,
        use_batch_prompts: bool = True,
    ):
        """Initialize the hypothesis generator.
        
        Args:
            openai_client: OpenAI client for LLM calls.
            batch_size: Max failures per batch LLM call.
            use_batch_prompts: Whether to use batch prompts for efficiency.
        """
        self.openai_client = openai_client or OpenAIClient()
        self.batch_size = batch_size
        self.use_batch_prompts = use_batch_prompts
        
        # Cache for hypotheses
        self._cache: dict[str, Hypothesis] = {}
    
    def generate(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> HypothesisBatch:
        """Generate hypotheses for all failures.
        
        Args:
            failures: List of failure instances.
            changed_files: List of changed file paths.
            file_diffs: Optional dict of file diffs for context.
            
        Returns:
            HypothesisBatch with all hypotheses.
        """
        if not failures:
            return HypothesisBatch()
        
        # Check cache first
        uncached_failures = []
        cached_hypotheses = []
        
        for f in failures:
            cache_key = self._get_cache_key(f, changed_files)
            if cache_key in self._cache:
                cached_hypotheses.append(self._cache[cache_key])
            else:
                uncached_failures.append(f)
        
        # Generate for uncached failures
        new_hypotheses = []
        if uncached_failures:
            if self.use_batch_prompts and len(uncached_failures) > 1:
                new_hypotheses = self._generate_batch(uncached_failures, changed_files, file_diffs)
            else:
                for f in uncached_failures:
                    h = self._generate_single(f, changed_files, file_diffs)
                    new_hypotheses.append(h)
            
            # Cache new hypotheses
            for f, h in zip(uncached_failures, new_hypotheses):
                cache_key = self._get_cache_key(f, changed_files)
                self._cache[cache_key] = h
        
        # Combine cached and new
        all_hypotheses = cached_hypotheses + new_hypotheses
        
        return HypothesisBatch(hypotheses=all_hypotheses)
    
    def _generate_single(
        self,
        failure: FailureInstance,
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> Hypothesis:
        """Generate hypothesis for a single failure."""
        # Format stack frames
        stack_frames = ""
        stack_lines = set()  # Track line numbers for diff relevance
        for i, frame in enumerate(failure.frames[:5]):
            stack_frames += f"    {i+1}. {frame.file}:{frame.line} in {frame.func}\n"
            if frame.file and frame.line:
                stack_lines.add((frame.file, frame.line))
        
        if not stack_frames:
            stack_frames = "    (no stack trace available)"
        
        # Format changed files WITH diff content
        changed_str = self._format_files_with_diffs(
            changed_files, file_diffs, stack_lines, max_files=10, max_diff_lines=30
        )
        
        prompt = self.HYPOTHESIS_PROMPT.format(
            test_name=failure.test_name or "unknown",
            exception_type=failure.exception_type or "Unknown",
            exception_message=(failure.exception_message or "")[:500],
            stack_frames=stack_frames,
            changed_files_with_diffs=changed_str,
        )
        
        try:
            response = self.openai_client._call_openai(
                prompt=prompt,
                system_prompt="You are a software engineer expert at root cause analysis. Respond only with valid JSON.",
                temperature=0.1,
            )
            
            result = json.loads(response)
            
            return Hypothesis(
                failure_id=failure.failure_id,
                suspected_change=result.get("suspected_change"),
                mechanism=result.get("mechanism", "Unknown mechanism"),
                confidence=float(result.get("confidence", 0.5)),
                reasoning=result.get("reasoning", ""),
                category=result.get("category", "unknown"),
                affected_component=result.get("affected_component", ""),
            )
            
        except Exception as e:
            # Fallback hypothesis
            return self._fallback_hypothesis(failure, changed_files)
    
    def _generate_batch(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses for a batch of failures."""
        hypotheses = []
        
        # Process in chunks
        for chunk_start in range(0, len(failures), self.batch_size):
            chunk = failures[chunk_start:chunk_start + self.batch_size]
            chunk_hypotheses = self._generate_batch_chunk(chunk, changed_files, file_diffs)
            hypotheses.extend(chunk_hypotheses)
        
        return hypotheses
    
    def _generate_batch_chunk(
        self,
        failures: list[FailureInstance],
        changed_files: list[str],
        file_diffs: dict[str, str] | None = None,
    ) -> list[Hypothesis]:
        """Generate hypotheses for a small batch chunk."""
        # Collect all stack lines from failures for diff relevance
        all_stack_lines = set()
        
        # Format failures
        failures_str = ""
        for i, f in enumerate(failures, 1):
            stack_frames = ""
            for j, frame in enumerate(f.frames[:3]):
                stack_frames += f"      {frame.file}:{frame.line} in {frame.func}\n"
                if frame.file and frame.line:
                    all_stack_lines.add((frame.file, frame.line))
            if not stack_frames:
                stack_frames = "      (no stack trace)\n"
            
            failures_str += f"""
{i}. Failure ID: {f.failure_id}
   Test: {f.test_name or "unknown"}
   Exception: {f.exception_type or "Unknown"}: {(f.exception_message or "")[:200]}
   Stack:
{stack_frames}
"""
        
        # Format changed files WITH diff content
        changed_str = self._format_files_with_diffs(
            changed_files, file_diffs, all_stack_lines, max_files=8, max_diff_lines=20
        )
        
        prompt = self.BATCH_PROMPT.format(
            changed_files_with_diffs=changed_str,
            failures=failures_str,
        )
        
        try:
            response = self.openai_client._call_openai(
                prompt=prompt,
                system_prompt="You are a software engineer expert at root cause analysis. Respond only with valid JSON array.",
                temperature=0.1,
            )
            
            results = json.loads(response)
            
            # Map results back to failures
            result_map = {r.get("failure_id"): r for r in results}
            
            hypotheses = []
            for f in failures:
                if f.failure_id in result_map:
                    r = result_map[f.failure_id]
                    hypotheses.append(Hypothesis(
                        failure_id=f.failure_id,
                        suspected_change=r.get("suspected_change"),
                        mechanism=r.get("mechanism", "Unknown mechanism"),
                        confidence=float(r.get("confidence", 0.5)),
                        reasoning=r.get("reasoning", ""),
                        category=r.get("category", "unknown"),
                        affected_component=r.get("affected_component", ""),
                    ))
                else:
                    hypotheses.append(self._fallback_hypothesis(f, changed_files))
            
            return hypotheses
            
        except Exception as e:
            # Fallback for all failures in chunk
            return [self._fallback_hypothesis(f, changed_files) for f in failures]
    
    def _fallback_hypothesis(
        self,
        failure: FailureInstance,
        changed_files: list[str],
    ) -> Hypothesis:
        """Generate fallback hypothesis without LLM."""
        # Try to find direct file overlap
        suspected_change = None
        
        if failure.frames and changed_files:
            changed_base = {f.split('/')[-1].replace('.java', '').replace('.py', '').lower() 
                          for f in changed_files}
            
            for frame in failure.frames:
                if frame.file:
                    frame_base = frame.file.split('/')[-1].replace('.java', '').replace('.py', '').lower()
                    for cf in changed_base:
                        if cf in frame_base or frame_base in cf:
                            # Found a match
                            for orig_file in changed_files:
                                if cf in orig_file.lower():
                                    suspected_change = orig_file
                                    break
                    if suspected_change:
                        break
        
        return Hypothesis(
            failure_id=failure.failure_id,
            suspected_change=suspected_change,
            mechanism=f"Failure in {failure.exception_type or 'test'}" + 
                      (f" likely related to change in {suspected_change}" if suspected_change else ""),
            confidence=0.3 if suspected_change else 0.1,
            reasoning="Fallback hypothesis based on file overlap analysis",
            category="unknown",
            affected_component="",
        )
    
    def _format_files_with_diffs(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str] | None,
        stack_lines: set[tuple[str, int]],
        max_files: int = 10,
        max_diff_lines: int = 30,
    ) -> str:
        """Format changed files with relevant diff snippets.
        
        Prioritizes files that appear in the stack trace and shows
        the most relevant hunks from the diffs.
        
        Args:
            changed_files: List of changed file paths.
            file_diffs: Dict of file path -> diff content.
            stack_lines: Set of (file, line) tuples from stack traces.
            max_files: Maximum files to include.
            max_diff_lines: Maximum total diff lines to include.
        """
        if not changed_files:
            return "  (no changed files)"
        
        file_diffs = file_diffs or {}
        
        # Score files by relevance to stack trace
        scored_files = []
        stack_file_bases = set()
        for file, line in stack_lines:
            if file:
                base = file.split('/')[-1].replace('.java', '').replace('.py', '').lower()
                stack_file_bases.add(base)
        
        for f in changed_files:
            f_base = f.split('/')[-1].replace('.java', '').replace('.py', '').lower()
            
            # High score if file appears in stack
            score = 0
            if f_base in stack_file_bases:
                score = 100
            elif any(f_base in sf or sf in f_base for sf in stack_file_bases):
                score = 50
            
            # Also consider diff size (prefer non-trivial changes)
            if f in file_diffs:
                diff_lines = len(file_diffs[f].split('\n'))
                score += min(diff_lines / 10, 10)  # Cap contribution at 10
            
            scored_files.append((score, f))
        
        # Sort by score descending, take top files
        scored_files.sort(key=lambda x: -x[0])
        selected_files = [f for _, f in scored_files[:max_files]]
        
        # Format output
        output_parts = []
        total_diff_lines = 0
        
        for f in selected_files:
            part = f"  - {f}"
            
            if f in file_diffs and total_diff_lines < max_diff_lines:
                diff_content = file_diffs[f]
                
                # Extract relevant hunks (prioritize those matching stack lines)
                relevant_hunks = self._extract_relevant_hunks(
                    diff_content, stack_lines, f,
                    max_lines=max_diff_lines - total_diff_lines
                )
                
                if relevant_hunks:
                    part += f"\n    ```diff\n{relevant_hunks}\n    ```"
                    total_diff_lines += len(relevant_hunks.split('\n'))
            
            output_parts.append(part)
        
        # Note if we truncated
        if len(changed_files) > max_files:
            output_parts.append(f"  ... and {len(changed_files) - max_files} more files")
        
        return "\n".join(output_parts)
    
    def _extract_relevant_hunks(
        self,
        diff_content: str,
        stack_lines: set[tuple[str, int]],
        file_path: str,
        max_lines: int = 20,
    ) -> str:
        """Extract the most relevant hunks from a diff.
        
        Prioritizes hunks that contain line numbers appearing in stack traces.
        """
        import re
        
        if not diff_content:
            return ""
        
        lines = diff_content.split('\n')
        
        # Parse hunks and score them
        hunks = []
        current_hunk = []
        hunk_start_line = 0
        hunk_end_line = 0
        
        for line in lines:
            # Detect hunk header: @@ -old_start,old_count +new_start,new_count @@
            hunk_match = re.match(r'^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@', line)
            
            if hunk_match:
                # Save previous hunk
                if current_hunk:
                    hunks.append({
                        'lines': current_hunk,
                        'start': hunk_start_line,
                        'end': hunk_end_line,
                    })
                
                # Start new hunk
                hunk_start_line = int(hunk_match.group(1))
                hunk_count = int(hunk_match.group(2)) if hunk_match.group(2) else 1
                hunk_end_line = hunk_start_line + hunk_count
                current_hunk = [line]
            elif current_hunk is not None and (
                line.startswith('+') or line.startswith('-') or line.startswith(' ') or line == ''
            ):
                current_hunk.append(line)
        
        # Don't forget last hunk
        if current_hunk:
            hunks.append({
                'lines': current_hunk,
                'start': hunk_start_line,
                'end': hunk_end_line,
            })
        
        if not hunks:
            # No hunks found, return first N lines of diff
            return '\n'.join(lines[:max_lines])
        
        # Score hunks by relevance to stack lines
        file_base = file_path.split('/')[-1].replace('.java', '').replace('.py', '').lower()
        
        for hunk in hunks:
            score = 0
            
            for stack_file, stack_line in stack_lines:
                if not stack_file:
                    continue
                    
                stack_base = stack_file.split('/')[-1].replace('.java', '').replace('.py', '').lower()
                
                # Check if this file matches and line is in hunk range
                if file_base in stack_base or stack_base in file_base:
                    if hunk['start'] <= stack_line <= hunk['end']:
                        score += 100  # Direct hit!
                    elif abs(stack_line - hunk['start']) < 20 or abs(stack_line - hunk['end']) < 20:
                        score += 30  # Close
            
            # Also score by added/removed lines (actual changes)
            change_lines = sum(1 for l in hunk['lines'] if l.startswith('+') or l.startswith('-'))
            score += change_lines
            
            hunk['score'] = score
        
        # Sort hunks by score, select top ones
        hunks.sort(key=lambda h: -h['score'])
        
        selected_lines = []
        for hunk in hunks:
            hunk_content = hunk['lines']
            if len(selected_lines) + len(hunk_content) > max_lines:
                # Can we fit a truncated version?
                remaining = max_lines - len(selected_lines)
                if remaining > 5:
                    selected_lines.extend(hunk_content[:remaining-1])
                    selected_lines.append('    ... (truncated)')
                break
            selected_lines.extend(hunk_content)
        
        return '\n'.join(selected_lines)
    
    def _get_cache_key(
        self,
        failure: FailureInstance,
        changed_files: list[str],
    ) -> str:
        """Generate cache key for a failure + changes combination."""
        # Include failure ID and hash of changed files
        changes_hash = hash(tuple(sorted(changed_files)))
        return f"{failure.failure_id}:{changes_hash}"
    
    def clear_cache(self) -> None:
        """Clear the hypothesis cache."""
        self._cache.clear()
