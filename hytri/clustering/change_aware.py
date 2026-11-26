"""Change-Aware Semantic Clustering.

Extends the category-guided approach with semantic understanding of code changes.
Uses LLM to understand what code changes do and how they relate to failures,
enabling indirect effect detection that lexical methods miss.

Key Innovation: While BuildSheriff uses lexical file overlap (changed file name in stack trace),
we use semantic reasoning to connect changes to failures even when there's no direct file overlap.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from hytri.llm.openai_client import OpenAIClient
from hytri.models import Cluster, FailureInstance, NormalizedFailure
from hytri.clustering.category_guided import CategoryGuidedFusion, CategoryGuidedConfig


@dataclass
class CodeChange:
    """Represents a code change (from git diff)."""
    file_path: str
    diff_content: str
    additions: int = 0
    deletions: int = 0
    
    # LLM-generated semantic summary
    semantic_summary: str = ""
    affected_components: list[str] = field(default_factory=list)
    change_type: str = ""  # e.g., "bug_fix", "refactor", "feature", "config_change"


@dataclass
class ChangeAttribution:
    """Attribution of a failure to a specific code change."""
    change_file: str
    confidence: float  # 0.0 to 1.0
    reasoning: str
    is_direct: bool  # Direct (file in stack) vs indirect (semantic connection)


@dataclass
class ChangeAwareNormalization:
    """Extended normalization with change attribution."""
    base_normalization: NormalizedFailure
    change_attributions: list[ChangeAttribution] = field(default_factory=list)
    primary_change: Optional[str] = None  # Most likely causal change


@dataclass 
class ChangeAwareConfig(CategoryGuidedConfig):
    """Configuration for change-aware clustering."""
    
    # Change-aware weights
    change_attribution_weight: float = 0.20      # Weight for change attribution similarity
    direct_change_boost: float = 0.15            # Boost when both failures touch same changed file
    indirect_change_boost: float = 0.10          # Boost when LLM attributes both to same change
    
    # Attribution confidence threshold
    min_attribution_confidence: float = 0.5
    
    # Whether to use LLM for change summarization (vs. just file names)
    semantic_change_analysis: bool = True
    
    # --- Mitigation for multi-change commits ---
    # When many files are changed, reduce boost to prevent false positive merges
    max_changes_for_full_boost: int = 5          # Above this, scale down boost
    
    # Require specific/exclusive attribution (not just any shared file)
    require_exclusive_primary: bool = True       # Only boost if primary changes match
    max_attributions_for_boost: int = 3          # If failure attributed to more files, reduce boost
    
    # Minimum specificity for boost: 1.0 / num_attributions
    # e.g., if failure attributed to 5 files, specificity = 0.2
    min_attribution_specificity: float = 0.25   # Requires attribution to ≤4 files for full boost


class ChangeAnalyzer:
    """Analyzes code changes semantically using LLM."""
    
    # Prompt template for change summarization
    CHANGE_SUMMARY_PROMPT = """Analyze this code change and provide a brief summary.

FILE: {file_path}
DIFF:
{diff_content}

Respond in JSON format:
{{
  "summary": "One sentence describing what this change does",
  "affected_components": ["list", "of", "affected", "components"],
  "change_type": "bug_fix|refactor|feature|config_change|dependency|test",
  "potential_side_effects": ["list", "of", "things", "this", "might", "break"]
}}"""

    # Prompt template for failure-change attribution
    ATTRIBUTION_PROMPT = """Given a test failure and recent code changes, determine if any change likely caused this failure.

FAILURE:
  Test: {test_name}
  Exception: {exception_type}
  Message: {exception_message}
  Stack Trace (top frames):
{stack_frames}

RECENT CODE CHANGES:
{changes_summary}

For each change, assess if it could have caused this failure.
Respond in JSON format:
{{
  "attributions": [
    {{
      "change_file": "path/to/file.java",
      "confidence": 0.0-1.0,
      "reasoning": "Why this change might have caused the failure",
      "is_direct": true/false  // true if changed file appears in stack trace
    }}
  ],
  "primary_change": "path/to/most/likely/cause.java or null if none"
}}"""

    def __init__(self, openai_client: OpenAIClient):
        self.openai_client = openai_client
        self._change_cache: dict[str, CodeChange] = {}
        
    def summarize_change(self, file_path: str, diff_content: str) -> CodeChange:
        """Get semantic summary of a code change."""
        cache_key = f"{file_path}:{hash(diff_content)}"
        if cache_key in self._change_cache:
            return self._change_cache[cache_key]
        
        # Count additions/deletions
        additions = diff_content.count('\n+') - diff_content.count('\n+++')
        deletions = diff_content.count('\n-') - diff_content.count('\n---')
        
        # Get LLM summary
        prompt = self.CHANGE_SUMMARY_PROMPT.format(
            file_path=file_path,
            diff_content=diff_content[:2000]  # Truncate large diffs
        )
        
        try:
            response = self.openai_client._call_openai(
                prompt=prompt,
                system_prompt="You are a code change analyzer. Respond only with valid JSON.",
                temperature=0.1,
            )
            
            import json
            result = json.loads(response)
            
            change = CodeChange(
                file_path=file_path,
                diff_content=diff_content,
                additions=additions,
                deletions=deletions,
                semantic_summary=result.get("summary", ""),
                affected_components=result.get("affected_components", []),
                change_type=result.get("change_type", "unknown"),
            )
        except Exception as e:
            # Fallback to basic info
            change = CodeChange(
                file_path=file_path,
                diff_content=diff_content,
                additions=additions,
                deletions=deletions,
                semantic_summary=f"Change to {file_path}",
                affected_components=[file_path.split('/')[-1].replace('.java', '').replace('.py', '')],
                change_type="unknown",
            )
        
        self._change_cache[cache_key] = change
        return change
    
    def attribute_failure_to_changes(
        self,
        failure: FailureInstance,
        changes: list[CodeChange],
    ) -> list[ChangeAttribution]:
        """Determine which changes likely caused a failure."""
        if not changes:
            return []
        
        # Build changes summary for prompt
        changes_summary = ""
        for i, change in enumerate(changes, 1):
            changes_summary += f"\n{i}. {change.file_path}\n"
            changes_summary += f"   Summary: {change.semantic_summary}\n"
            changes_summary += f"   Type: {change.change_type}\n"
            changes_summary += f"   Affected: {', '.join(change.affected_components)}\n"
        
        # Build stack frames summary
        stack_frames = ""
        for frame in failure.frames[:10]:
            stack_frames += f"    {frame.file}:{frame.line} in {frame.func}\n"
        
        prompt = self.ATTRIBUTION_PROMPT.format(
            test_name=failure.test_name,
            exception_type=failure.exception_type,
            exception_message=failure.exception_message[:500],
            stack_frames=stack_frames,
            changes_summary=changes_summary,
        )
        
        try:
            response = self.openai_client._call_openai(
                prompt=prompt,
                system_prompt="You are a failure analysis expert. Respond only with valid JSON.",
                temperature=0.1,
            )
            
            import json
            result = json.loads(response)
            
            attributions = []
            for attr in result.get("attributions", []):
                attributions.append(ChangeAttribution(
                    change_file=attr.get("change_file", ""),
                    confidence=float(attr.get("confidence", 0.0)),
                    reasoning=attr.get("reasoning", ""),
                    is_direct=attr.get("is_direct", False),
                ))
            
            return attributions
            
        except Exception as e:
            # Fallback: check for direct file overlap
            attributions = []
            stack_files = {f.file for f in failure.frames}
            
            for change in changes:
                # Check if changed file appears in stack trace
                file_name = change.file_path.split('/')[-1]
                is_direct = any(file_name in sf for sf in stack_files)
                
                if is_direct:
                    attributions.append(ChangeAttribution(
                        change_file=change.file_path,
                        confidence=0.7,  # Lower confidence for fallback
                        reasoning=f"Changed file {file_name} appears in stack trace",
                        is_direct=True,
                    ))
            
            return attributions


class ChangeAwareFusion(CategoryGuidedFusion):
    """Extends category-guided fusion with change-awareness."""
    
    def __init__(
        self,
        config: ChangeAwareConfig | None = None,
        openai_client: OpenAIClient | None = None,
        changes: list[CodeChange] | None = None,
    ):
        super().__init__(config=config or ChangeAwareConfig(), openai_client=openai_client)
        self.change_config = config or ChangeAwareConfig()
        self.changes = changes or []
        self.change_analyzer = ChangeAnalyzer(self.openai_client)
        
        # Cache for change-aware normalizations
        self._change_norm_cache: dict[str, ChangeAwareNormalization] = {}
    
    def set_changes(self, changes: list[CodeChange]) -> None:
        """Set the code changes for this build."""
        self.changes = changes
        self._change_norm_cache.clear()
    
    def compute_change_aware_similarity(
        self,
        failures: list[FailureInstance],
        s_lex: np.ndarray,
    ) -> tuple[np.ndarray, list[ChangeAwareNormalization]]:
        """Compute change-aware fused similarity matrix.
        
        Extends the category-guided approach with change attribution.
        """
        n = len(failures)
        if n == 0:
            return np.array([]), []
        
        # Step 1: Get base category-guided similarity
        s_fused_base, base_normalizations = self.compute_fused_similarity(failures, s_lex)
        
        # Step 2: Get change attributions for each failure
        change_aware_norms = self._get_change_attributions(failures, base_normalizations)
        
        # Step 3: Compute change attribution similarity matrix
        s_change = self._compute_change_attribution_matrix(change_aware_norms)
        
        # Step 4: Add change attribution signal to fused similarity
        s_final = s_fused_base + self.change_config.change_attribution_weight * s_change
        
        # Clip to [0, 1]
        s_final = np.clip(s_final, 0.0, 1.0)
        np.fill_diagonal(s_final, 1.0)
        
        return s_final, change_aware_norms
    
    def _get_change_attributions(
        self,
        failures: list[FailureInstance],
        base_normalizations: list[NormalizedFailure],
    ) -> list[ChangeAwareNormalization]:
        """Get change attributions for all failures."""
        change_aware_norms = []
        
        for failure, base_norm in zip(failures, base_normalizations):
            if failure.failure_id in self._change_norm_cache:
                change_aware_norms.append(self._change_norm_cache[failure.failure_id])
            else:
                # Get attributions
                attributions = self.change_analyzer.attribute_failure_to_changes(
                    failure, self.changes
                )
                
                # Find primary change (highest confidence above threshold)
                primary_change = None
                max_conf = self.change_config.min_attribution_confidence
                for attr in attributions:
                    if attr.confidence > max_conf:
                        max_conf = attr.confidence
                        primary_change = attr.change_file
                
                change_norm = ChangeAwareNormalization(
                    base_normalization=base_norm,
                    change_attributions=attributions,
                    primary_change=primary_change,
                )
                
                self._change_norm_cache[failure.failure_id] = change_norm
                change_aware_norms.append(change_norm)
        
        return change_aware_norms
    
    def _compute_change_attribution_matrix(
        self,
        change_norms: list[ChangeAwareNormalization],
    ) -> np.ndarray:
        """Compute similarity based on change attributions.
        
        Failures attributed to the same change are boosted, but with mitigations
        to prevent false positives when many files are changed:
        
        1. Scale boost inversely with number of total changes
        2. Require specific attribution (not scattered across many files)
        3. Prioritize PRIMARY change match over any shared attribution
        4. Require DIRECT file overlap (in stack trace) for strongest boost
        """
        n = len(change_norms)
        matrix = np.zeros((n, n))
        
        # Compute scaling factor based on number of changes
        num_changes = len(self.changes)
        if num_changes > self.change_config.max_changes_for_full_boost:
            # Scale down boost when many files changed
            # e.g., 10 changes with max=5 → scale = 5/10 = 0.5
            change_scale = self.change_config.max_changes_for_full_boost / num_changes
        else:
            change_scale = 1.0
        
        for i in range(n):
            for j in range(i + 1, n):
                norm_i, norm_j = change_norms[i], change_norms[j]
                
                # Compute specificity for each failure
                # Specificity = 1 / num_attributions (higher = more specific)
                num_attr_i = len([a for a in norm_i.change_attributions 
                                  if a.confidence >= self.change_config.min_attribution_confidence])
                num_attr_j = len([a for a in norm_j.change_attributions
                                  if a.confidence >= self.change_config.min_attribution_confidence])
                
                specificity_i = 1.0 / max(num_attr_i, 1)
                specificity_j = 1.0 / max(num_attr_j, 1)
                avg_specificity = (specificity_i + specificity_j) / 2
                
                # Check if attributions are specific enough
                is_specific = avg_specificity >= self.change_config.min_attribution_specificity
                
                # Check for shared PRIMARY change (strongest signal)
                if (norm_i.primary_change and norm_j.primary_change and 
                    norm_i.primary_change == norm_j.primary_change):
                    
                    if self.change_config.require_exclusive_primary:
                        # Only full boost if attributions are specific
                        if is_specific:
                            boost = 1.0 * change_scale
                        else:
                            # Reduced boost for non-specific attributions
                            boost = 0.5 * change_scale * avg_specificity
                    else:
                        boost = 1.0 * change_scale
                    
                    matrix[i, j] = boost
                    matrix[j, i] = boost
                    continue
                
                # If require_exclusive_primary is set, skip weaker signals
                if self.change_config.require_exclusive_primary:
                    # Only check for DIRECT overlap as fallback
                    direct_i = {a.change_file for a in norm_i.change_attributions if a.is_direct}
                    direct_j = {a.change_file for a in norm_j.change_attributions if a.is_direct}
                    
                    if direct_i & direct_j and is_specific:
                        # Direct overlap with specific attribution
                        boost = 0.6 * change_scale * avg_specificity
                        matrix[i, j] = boost
                        matrix[j, i] = boost
                    continue
                
                # --- Legacy behavior (when require_exclusive_primary=False) ---
                # Check for any shared attributions
                files_i = {a.change_file for a in norm_i.change_attributions 
                          if a.confidence >= self.change_config.min_attribution_confidence}
                files_j = {a.change_file for a in norm_j.change_attributions
                          if a.confidence >= self.change_config.min_attribution_confidence}
                
                shared = files_i & files_j
                if shared:
                    # Compute average confidence for shared files
                    total_conf = 0.0
                    count = 0
                    for attr in norm_i.change_attributions + norm_j.change_attributions:
                        if attr.change_file in shared:
                            total_conf += attr.confidence
                            count += 1
                    
                    avg_conf = total_conf / count if count > 0 else 0.0
                    # Scale by change count AND specificity
                    matrix[i, j] = avg_conf * change_scale * avg_specificity
                    matrix[j, i] = avg_conf * change_scale * avg_specificity
                
                # Check for direct file overlap (like BuildSheriff)
                direct_i = {a.change_file for a in norm_i.change_attributions if a.is_direct}
                direct_j = {a.change_file for a in norm_j.change_attributions if a.is_direct}
                
                if direct_i & direct_j:
                    current = matrix[i, j]
                    boost = 0.5 * change_scale * avg_specificity
                    matrix[i, j] = max(current, boost)
                    matrix[j, i] = max(current, boost)
        
        return matrix


class ChangeAwareClusterer:
    """Clusterer using change-aware fusion.
    
    Extends CategoryGuidedClusterer with code change understanding.
    Now includes direct file-frame overlap detection (BuildSheriff-style)
    for robust performance even when stack frames lack file info.
    """
    
    def __init__(
        self,
        config: ChangeAwareConfig | None = None,
        openai_client: OpenAIClient | None = None,
        direct_overlap_weight: float = 0.5,  # Weight for direct file overlap boost
    ):
        self.config = config or ChangeAwareConfig()
        self.openai_client = openai_client or OpenAIClient()
        self.fusion = ChangeAwareFusion(
            config=self.config, 
            openai_client=self.openai_client
        )
        self.direct_overlap_weight = direct_overlap_weight
        self.changed_files: list[str] = []
    
    def compute_direct_file_overlap(self, failures: list[FailureInstance]) -> np.ndarray:
        """Compute similarity based on direct frame-to-changed-file overlap.
        
        This is a BuildSheriff-style signal that works even when frames have file=Unknown
        by parsing raw stack_trace text to extract class names.
        
        This is FAST (no LLM) and provides a strong signal when failures touch 
        the same changed files.
        """
        import re
        n = len(failures)
        sim_matrix = np.zeros((n, n))
        
        if not self.changed_files:
            return sim_matrix
        
        # Get base filenames/class names of changed files for matching
        changed_base = set()
        for f in self.changed_files:
            # Handle both path/file.java and package.Class formats
            basename = f.split('/')[-1].replace('.java', '').replace('.py', '')
            changed_base.add(basename.lower())
            # Also add last part of package name if it's a path
            if '.' in basename:
                changed_base.add(basename.split('.')[-1].lower())
        
        # For each failure, find which changed files its frames/stack_trace touch
        touched_files = []
        for f in failures:
            touched = set()
            
            # Method 1: Check parsed frames
            for frame in f.frames:
                frame_file = frame.file.split('/')[-1].replace('.java', '').replace('.py', '') if frame.file else ""
                if frame_file and frame_file.lower() != "unknown":
                    for cf in changed_base:
                        if cf and (cf in frame_file.lower() or frame_file.lower() in cf):
                            touched.add(cf)
                
                # Also check func name for class info
                if frame.func:
                    # Extract class name from func like "com.package.ClassName.methodName"
                    parts = frame.func.split('.')
                    for part in parts:
                        if part and len(part) > 2 and part[0].isupper():  # Likely a class name
                            for cf in changed_base:
                                if cf and (cf in part.lower() or part.lower() in cf):
                                    touched.add(cf)
            
            # Method 2: Parse raw stack_trace text (for Bears where frames have file=Unknown)
            stack_text = f.stack_trace if hasattr(f, 'stack_trace') and f.stack_trace else ""
            if stack_text:
                # Find "at package.Class.method" patterns
                at_patterns = re.findall(r'at\s+([\w.]+)', stack_text)
                for pattern in at_patterns:
                    parts = pattern.split('.')
                    for part in parts:
                        if part and len(part) > 2 and part[0].isupper():  # Likely a class name
                            for cf in changed_base:
                                if cf and (cf in part.lower() or part.lower() in cf):
                                    touched.add(cf)
            
            touched_files.append(touched)
        
        # Compute similarity based on shared touched files
        for i in range(n):
            for j in range(i, n):
                if touched_files[i] and touched_files[j]:
                    # Jaccard similarity of touched changed files
                    intersection = len(touched_files[i] & touched_files[j])
                    union = len(touched_files[i] | touched_files[j])
                    if union > 0 and intersection > 0:
                        sim = intersection / union
                        sim_matrix[i, j] = sim
                        sim_matrix[j, i] = sim
        
        return sim_matrix
    
    def _compute_adaptive_threshold(self, failures: list[FailureInstance]) -> float:
        """Compute adaptive threshold based on exception type diversity.
        
        Inherited logic from CategoryGuidedClusterer.
        """
        if not self.config.adaptive_threshold or len(failures) <= 1:
            return self.config.distance_threshold
        
        exception_types = set(f.exception_type for f in failures)
        diversity_ratio = len(exception_types) / len(failures)
        
        if diversity_ratio < 0.1:
            return self.config.threshold_low_diversity
        elif diversity_ratio < 0.3:
            return self.config.threshold_medium_diversity
        else:
            return self.config.threshold_high_diversity
    
    def set_changes_from_build(
        self,
        changed_files: list[str],
        file_diffs: dict[str, str],
    ) -> None:
        """Set changes from build metadata (BugSwarm format)."""
        # Store changed files for direct overlap computation
        self.changed_files = changed_files
        
        changes = []
        for file_path in changed_files:
            diff_content = file_diffs.get(file_path, "")
            if self.config.semantic_change_analysis:
                change = self.fusion.change_analyzer.summarize_change(file_path, diff_content)
            else:
                # Basic change without semantic analysis
                change = CodeChange(
                    file_path=file_path,
                    diff_content=diff_content,
                    semantic_summary=f"Change to {file_path}",
                )
            changes.append(change)
        
        self.fusion.set_changes(changes)
    
    def cluster(
        self,
        failures: list[FailureInstance],
        s_lex: np.ndarray,
    ) -> tuple[list[Cluster], list[str], list[ChangeAwareNormalization]]:
        """Cluster failures using change-aware fusion with direct file overlap.
        
        This method combines:
        1. Lexical similarity (stack trace overlap)
        2. Semantic similarity (LLM embeddings)
        3. Direct file-frame overlap (BuildSheriff-style, no LLM)
        
        The direct file overlap provides a strong signal even when stack frames
        lack proper file information (e.g., Bears dataset).
        
        Args:
            failures: List of failure instances.
            s_lex: Pre-computed lexical similarity matrix.
            
        Returns:
            Tuple of (clusters, noise failure IDs, change-aware normalizations).
        """
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        
        n = len(failures)
        if n == 0:
            return [], [], []
        
        if n == 1:
            cluster = Cluster(
                cluster_id="cluster_0",
                failure_ids=[failures[0].failure_id],
                representative_failure_id=failures[0].failure_id,
            )
            # Get normalization
            s_lex_single = np.array([[1.0]])
            _, norms = self.fusion.compute_change_aware_similarity(failures, s_lex_single)
            return [cluster], [], norms
        
        # NEW: Compute direct file overlap similarity (BuildSheriff-style, no LLM)
        # This works even when frames have file=Unknown by parsing stack_trace text
        s_direct_overlap = self.compute_direct_file_overlap(failures)
        
        # Boost lexical similarity with direct overlap before fusion
        s_lex_boosted = s_lex + self.direct_overlap_weight * s_direct_overlap
        s_lex_boosted = np.clip(s_lex_boosted, 0.0, 1.0)
        
        # Compute change-aware similarity (uses LLM semantic analysis)
        s_fused, normalizations = self.fusion.compute_change_aware_similarity(failures, s_lex_boosted)
        
        # Convert to distance matrix
        distance_matrix = 1.0 - s_fused
        np.fill_diagonal(distance_matrix, 0.0)
        
        # Ensure symmetry and valid values
        distance_matrix = (distance_matrix + distance_matrix.T) / 2
        distance_matrix = np.clip(distance_matrix, 0.0, 1.0)
        
        # Perform hierarchical clustering
        condensed = squareform(distance_matrix, checks=False)
        linkage_matrix = linkage(condensed, method=self.config.linkage_method)
        
        # Compute adaptive threshold based on exception diversity
        threshold = self._compute_adaptive_threshold(failures)
        
        # Cut tree at threshold
        labels = fcluster(linkage_matrix, t=threshold, criterion='distance')
        
        # Convert labels to clusters with change attribution info
        clusters = self._labels_to_clusters(failures, labels, normalizations)
        
        return clusters, [], normalizations
    
    def _labels_to_clusters(
        self,
        failures: list[FailureInstance],
        labels: np.ndarray,
        normalizations: list[ChangeAwareNormalization],
    ) -> list[Cluster]:
        """Convert cluster labels to Cluster objects with change info."""
        from collections import defaultdict
        
        # Group failures by label
        label_to_data: dict[int, list[tuple[FailureInstance, ChangeAwareNormalization]]] = defaultdict(list)
        for failure, norm, label in zip(failures, normalizations, labels):
            label_to_data[label].append((failure, norm))
        
        clusters = []
        for label, data in sorted(label_to_data.items()):
            failure_ids = [f.failure_id for f, _ in data]
            
            # Find representative
            representative_id = failure_ids[0]
            
            # Collect change attributions for cluster
            all_attributions = []
            primary_changes = []
            for _, norm in data:
                all_attributions.extend(norm.change_attributions)
                if norm.primary_change:
                    primary_changes.append(norm.primary_change)
            
            # Find most common primary change
            cluster_primary_change = None
            if primary_changes:
                from collections import Counter
                cluster_primary_change = Counter(primary_changes).most_common(1)[0][0]
            
            # Collect categories
            categories = [n.base_normalization.category for _, n in data]
            most_common_category = max(set(categories), key=categories.count) if categories else "Unknown"
            
            cluster = Cluster(
                cluster_id=f"cluster_{label}",
                failure_ids=failure_ids,
                representative_failure_id=representative_id,
            )
            
            # Add rich metadata
            cluster.metadata = {
                "categories": list(set(categories)),
                "primary_category": most_common_category,
                "attributed_change": cluster_primary_change,
                "change_attributions": [
                    {"file": a.change_file, "confidence": a.confidence, "is_direct": a.is_direct}
                    for a in all_attributions if a.confidence >= 0.5
                ],
                "explanation": self._generate_cluster_explanation(data, cluster_primary_change),
            }
            
            clusters.append(cluster)
        
        return clusters
    
    def _generate_cluster_explanation(
        self,
        data: list[tuple[FailureInstance, ChangeAwareNormalization]],
        primary_change: str | None,
    ) -> str:
        """Generate human-readable explanation for cluster."""
        num_failures = len(data)
        categories = list(set(n.base_normalization.category for _, n in data))
        
        if primary_change:
            return f"{num_failures} failures ({', '.join(categories)}) likely caused by change to {primary_change}"
        else:
            return f"{num_failures} failures with categories: {', '.join(categories)}"

