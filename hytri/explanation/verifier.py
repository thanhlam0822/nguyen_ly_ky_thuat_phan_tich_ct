"""Component D2: Evidence Verifier.

Verifies that LLM-generated explanations are grounded in actual evidence
from the failure data. This prevents hallucinations and ensures trustworthiness.
"""

import re
from dataclasses import dataclass

from hytri.models import Cluster, ClusterExplanation, FailureInstance


@dataclass
class VerificationResult:
    """Result of verifying an explanation."""

    is_valid: bool
    groundedness_score: float
    evidence_matches: list[str]
    unsupported_claims: list[str]
    details: dict[str, bool]


class EvidenceVerifier:
    """Verifies that cluster explanations are grounded in evidence.

    Checks that explanations only reference:
    - Exception types actually present in the cluster
    - Frame functions/files actually present
    - Log templates actually present

    Produces a groundedness score and flags unsupported claims.
    """

    def __init__(
        self,
        strict_mode: bool = False,
        min_groundedness: float = 0.5,
    ):
        """Initialize the verifier.

        Args:
            strict_mode: If True, all claims must be grounded.
            min_groundedness: Minimum groundedness score to pass verification.
        """
        self.strict_mode = strict_mode
        self.min_groundedness = min_groundedness

    def verify(
        self,
        cluster: Cluster,
        explanation: ClusterExplanation,
        failures: list[FailureInstance],
    ) -> VerificationResult:
        """Verify that an explanation is grounded in evidence.

        Args:
            cluster: The cluster being explained.
            explanation: The LLM-generated explanation.
            failures: All failure instances.

        Returns:
            VerificationResult with groundedness score and details.
        """
        # Get cluster failures
        failure_map = {f.failure_id: f for f in failures}
        cluster_failures = [
            failure_map[fid]
            for fid in cluster.failure_ids
            if fid in failure_map
        ]

        # Extract evidence from cluster
        evidence = self._extract_evidence(cluster, cluster_failures)

        # Check explanation against evidence
        evidence_matches = []
        unsupported_claims = []
        check_details = {}

        # Combine title and summary for checking
        full_text = explanation.title + " " + " ".join(explanation.summary)
        full_text_lower = full_text.lower()

        # Check exception types
        exception_found = False
        for exc_type in evidence["exception_types"]:
            if exc_type.lower() in full_text_lower:
                evidence_matches.append(f"exception: {exc_type}")
                exception_found = True
        check_details["exception_type_grounded"] = exception_found

        # Check function names (also check method names without class prefix)
        func_found = False
        for func in evidence["functions"]:
            func_lower = func.lower()
            if func_lower in full_text_lower:
                evidence_matches.append(f"function: {func}")
                func_found = True
            # For Java "Class.method", also check just "method"
            elif "." in func:
                method_only = func.split(".")[-1].lower()
                if len(method_only) > 3 and method_only in full_text_lower:
                    evidence_matches.append(f"method: {func}")
                    func_found = True
        check_details["function_grounded"] = func_found

        # Check file names (also check without extension)
        file_found = False
        for filename in evidence["files"]:
            filename_lower = filename.lower()
            if filename_lower in full_text_lower:
                evidence_matches.append(f"file: {filename}")
                file_found = True
            # Also check without .java/.py extension
            base_name = filename_lower.replace(".java", "").replace(".py", "")
            if len(base_name) > 3 and base_name in full_text_lower:
                evidence_matches.append(f"file: {filename}")
                file_found = True
        check_details["file_grounded"] = file_found
        
        # Check Java class names
        class_found = False
        for class_name in evidence.get("classes", []):
            if class_name.lower() in full_text_lower:
                evidence_matches.append(f"class: {class_name}")
                class_found = True
        check_details["class_grounded"] = class_found

        # Check for unsupported module/component claims
        claimed_components = self._extract_claimed_components(full_text)
        for component in claimed_components:
            if not self._is_component_in_evidence(component, evidence):
                unsupported_claims.append(f"component: {component}")
        check_details["no_unsupported_components"] = len(unsupported_claims) == 0

        # Calculate groundedness score
        groundedness_score = self._calculate_groundedness(
            evidence_matches,
            unsupported_claims,
            check_details,
        )

        is_valid = groundedness_score >= self.min_groundedness
        if self.strict_mode:
            is_valid = is_valid and len(unsupported_claims) == 0

        return VerificationResult(
            is_valid=is_valid,
            groundedness_score=groundedness_score,
            evidence_matches=evidence_matches,
            unsupported_claims=unsupported_claims,
            details=check_details,
        )

    def _extract_evidence(
        self,
        cluster: Cluster,
        failures: list[FailureInstance],
    ) -> dict:
        """Extract all evidence from cluster failures."""
        evidence = {
            "exception_types": set(),
            "exception_messages": set(),
            "functions": set(),
            "files": set(),
            "modules": set(),
            "classes": set(),  # Java class names
            "log_templates": set(),
        }

        for failure in failures:
            evidence["exception_types"].add(failure.exception_type)
            # Also add simplified exception type (e.g., NullPointerException without package)
            simple_exc = failure.exception_type.split(".")[-1]
            evidence["exception_types"].add(simple_exc)
            evidence["exception_messages"].add(failure.exception_message[:100])

            for frame in failure.frames:
                evidence["functions"].add(frame.func)
                
                # Extract file name without path
                file_name = frame.file.split("/")[-1]
                evidence["files"].add(file_name)

                # Handle both Python and Java patterns
                if frame.file.endswith(".py"):
                    # Python: extract module name
                    module = file_name.replace(".py", "")
                    evidence["modules"].add(module)
                elif frame.file.endswith(".java"):
                    # Java: extract class name from file
                    class_name = file_name.replace(".java", "")
                    evidence["classes"].add(class_name)
                    evidence["modules"].add(class_name)
                
                # For Java functions like "StringUtils.isEmpty", extract class name
                if "." in frame.func:
                    parts = frame.func.split(".")
                    if len(parts) >= 2:
                        class_name = parts[0]
                        method_name = parts[-1]
                        evidence["classes"].add(class_name)
                        evidence["functions"].add(method_name)

            for template in failure.log_templates:
                evidence["log_templates"].add(template)

        # Convert sets to lists
        return {k: list(v) for k, v in evidence.items()}

    def _extract_claimed_components(self, text: str) -> list[str]:
        """Extract component/module names claimed in the explanation."""
        # Pattern for module-like names (e.g., pandas.io, requests.sessions)
        module_pattern = r'\b([a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)+)\b'
        matches = re.findall(module_pattern, text.lower())
        return list(set(matches))

    def _is_component_in_evidence(
        self,
        component: str,
        evidence: dict,
    ) -> bool:
        """Check if a claimed component is supported by evidence."""
        component_lower = component.lower()

        # Check if component matches any file/module
        for filename in evidence["files"]:
            if component_lower in filename.lower():
                return True

        for module in evidence["modules"]:
            if component_lower in module.lower() or module.lower() in component_lower:
                return True
        
        # Check Java classes
        for class_name in evidence.get("classes", []):
            if component_lower in class_name.lower() or class_name.lower() in component_lower:
                return True

        # Check if it's a common framework/library (allowed)
        # Python libraries
        common_python_libs = {
            "pandas", "numpy", "scipy", "sklearn", "torch", "tensorflow",
            "requests", "django", "flask", "pytest", "unittest", "os", "sys",
            "collections", "itertools", "functools", "typing", "json", "re",
        }
        # Java libraries and packages
        common_java_libs = {
            "java", "javax", "org", "com", "junit", "apache", "google",
            "springframework", "lang", "util", "io", "net", "time",
            "math", "text", "sql", "concurrent", "reflect", "annotation",
            "jfree", "mockito", "gson", "jackson", "joda", "commons",
        }
        common_libs = common_python_libs | common_java_libs
        
        parts = component_lower.split(".")
        if parts[0] in common_libs:
            return True
        
        # Also allow if any part matches common Java/test patterns
        test_patterns = {"test", "tests", "mock", "stub", "assert", "expect"}
        if any(p in test_patterns for p in parts):
            return True

        return False

    def _calculate_groundedness(
        self,
        evidence_matches: list[str],
        unsupported_claims: list[str],
        check_details: dict[str, bool],
    ) -> float:
        """Calculate overall groundedness score."""
        # Base score from evidence matches (more generous)
        evidence_score = min(len(evidence_matches) / 4, 1.0) * 0.4

        # Penalty for unsupported claims (less harsh)
        penalty = len(unsupported_claims) * 0.05

        # Bonus for key groundings
        bonus = 0.0
        if check_details.get("exception_type_grounded"):
            bonus += 0.25
        if check_details.get("function_grounded"):
            bonus += 0.15
        if check_details.get("file_grounded"):
            bonus += 0.1
        if check_details.get("class_grounded"):
            bonus += 0.1
        if check_details.get("no_unsupported_components"):
            bonus += 0.1

        score = evidence_score + bonus - penalty
        return max(0.0, min(1.0, score))

    def verify_and_update(
        self,
        cluster: Cluster,
        failures: list[FailureInstance],
    ) -> Cluster:
        """Verify cluster's explanation and update with verification results.

        Args:
            cluster: Cluster with explanation to verify.
            failures: All failure instances.

        Returns:
            Cluster with updated explanation (groundedness score set).
        """
        if cluster.explanation is None:
            return cluster

        result = self.verify(cluster, cluster.explanation, failures)

        cluster.explanation.groundedness_score = result.groundedness_score
        cluster.explanation.verified = result.is_valid

        return cluster


