#!/usr/bin/env python3
"""Collect Defects4J failures for HyTri evaluation.

This script runs inside Docker to capture real Java test failures
from the Defects4J benchmark.
"""

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path("/output")
WORK_DIR = Path("/tmp/defects4j_work")

# Defects4J projects and their bug counts
DEFECTS4J_PROJECTS = {
    "Chart": 26,
    "Cli": 40,
    "Closure": 176,
    "Codec": 18,
    "Collections": 4,
    "Compress": 47,
    "Csv": 16,
    "Gson": 18,
    "JacksonCore": 26,
    "JacksonDatabind": 112,
    "JacksonXml": 6,
    "Jsoup": 93,
    "JxPath": 22,
    "Lang": 65,
    "Math": 106,
    "Mockito": 38,
    "Time": 27,
}


def run_command(cmd, cwd=None, timeout=600):
    """Run a command and return output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "stdout": "",
            "stderr": f"Timeout after {timeout}s",
            "returncode": -1,
        }
    except Exception as e:
        return {
            "success": False,
            "stdout": "",
            "stderr": str(e),
            "returncode": -1,
        }


def checkout_bug(project: str, bug_id: int, work_dir: Path, verbose: bool = False) -> bool:
    """Checkout a buggy version using defects4j checkout."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    if bug_dir.exists():
        shutil.rmtree(bug_dir, ignore_errors=True)
    bug_dir.mkdir(parents=True)
    
    result = run_command(
        f"defects4j checkout -p {project} -v {bug_id}b -w {bug_dir}",
        timeout=300,
    )
    
    if not result["success"] and verbose:
        print(f"\n    Checkout error: {result['stderr'][:200]}")
    
    return result["success"] and bug_dir.exists()


def compile_bug(project: str, bug_id: int, work_dir: Path) -> bool:
    """Compile the buggy version."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    
    result = run_command(
        "defects4j compile",
        cwd=str(bug_dir),
        timeout=600,
    )
    
    return result["success"]


def run_tests(project: str, bug_id: int, work_dir: Path) -> dict:
    """Run tests and capture failures."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    
    # Run failing tests (tests that expose the bug)
    result = run_command(
        "defects4j test",
        cwd=str(bug_dir),
        timeout=600,
    )
    
    output = result["stdout"] + "\n" + result["stderr"]
    
    # Check for failing tests file
    failing_tests_file = bug_dir / "failing_tests"
    failing_tests = []
    if failing_tests_file.exists():
        with open(failing_tests_file) as f:
            failing_tests = [line.strip() for line in f if line.strip()]
    
    return {
        "success": result["returncode"] == 0 and not failing_tests,
        "output": output,
        "failing_tests": failing_tests,
        "returncode": result["returncode"],
    }


def get_test_output(project: str, bug_id: int, work_dir: Path, test_name: str) -> str:
    """Get detailed output for a specific failing test."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    
    # Run the specific failing test to get detailed output
    result = run_command(
        f"defects4j test -t {test_name}",
        cwd=str(bug_dir),
        timeout=300,
    )
    
    return result["stdout"] + "\n" + result["stderr"]


def parse_java_failure(output: str, test_name: str, project: str, bug_id: int) -> dict:
    """Parse JUnit test output to extract failure information."""
    
    # Extract exception type and message
    exc_type = "TestFailure"
    exc_message = ""
    
    # Common Java exception patterns
    patterns = [
        r'(\w+(?:Error|Exception|Failure)):\s*(.+?)(?:\n|$)',
        r'java\.lang\.(\w+(?:Error|Exception)):\s*(.+?)(?:\n|$)',
        r'org\.junit\.(\w+(?:Error|Exception)):\s*(.+?)(?:\n|$)',
        r'(AssertionError)(?::\s*(.+?))?(?:\n|$)',
        r'expected:<(.+?)> but was:<(.+?)>',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output, re.IGNORECASE)
        if match:
            exc_type = match.group(1)
            if len(match.groups()) > 1 and match.group(2):
                exc_message = match.group(2).strip()[:500]
            break
    
    # Handle JUnit assertion format
    if "expected:" in output.lower() and "but was:" in output.lower():
        exc_type = "AssertionError"
        assert_match = re.search(r'expected:<(.+?)> but was:<(.+?)>', output)
        if assert_match:
            exc_message = f"expected:<{assert_match.group(1)}> but was:<{assert_match.group(2)}>"
    
    # Extract stack frames (Java format)
    frames = []
    # Java stack trace format: at package.Class.method(File.java:123)
    frame_pattern = re.compile(r'at\s+([\w.$]+)\.([\w$]+)\(([\w.]+):(\d+)\)')
    for match in frame_pattern.finditer(output):
        frames.append({
            "file": match.group(3),
            "line": int(match.group(4)),
            "func": f"{match.group(1)}.{match.group(2)}",
        })
    
    # Get stack trace section
    stack_trace = ""
    tb_match = re.search(
        r'((?:java\.lang\.\w+|org\.junit\.\w+|junit\.framework\.\w+).*?(?:\n\s+at .+)+)',
        output,
        re.DOTALL
    )
    if tb_match:
        stack_trace = tb_match.group(1)
    else:
        # Just get lines starting with "at "
        at_lines = re.findall(r'^\s*at .+$', output, re.MULTILINE)
        if at_lines:
            stack_trace = '\n'.join(at_lines[:20])
        else:
            stack_trace = output[-3000:]
    
    return {
        "test_name": test_name,
        "exception_type": exc_type,
        "exception_message": exc_message,
        "stack_trace": stack_trace[:5000],
        "frames": frames,
    }


def main():
    """Main function to collect Defects4J failures."""
    print("=" * 60)
    print("Defects4J Failure Collection")
    print("=" * 60)
    print(f"Java version: ", end="")
    os.system("java -version 2>&1 | head -1")
    
    # Check defects4j is available
    result = run_command("which defects4j && defects4j info -p Lang", timeout=30)
    if not result["success"]:
        print(f"\nWARNING: defects4j not properly configured!")
        print(f"  Error: {result['stderr'][:500]}")
        print(f"  Trying to run init.sh...")
        os.system("cd /opt/defects4j && ./init.sh 2>&1 | tail -10")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get configuration from environment
    projects_str = os.environ.get("PROJECTS", "")
    if projects_str:
        projects = {p: DEFECTS4J_PROJECTS.get(p, 10) for p in projects_str.split(",") if p in DEFECTS4J_PROJECTS}
    else:
        projects = DEFECTS4J_PROJECTS
    
    max_bugs = int(os.environ.get("MAX_BUGS", "100"))
    
    print(f"\nProjects: {', '.join(projects.keys())}")
    print(f"Max bugs per project: {max_bugs}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_failures = []
    all_builds = []
    stats = {
        "projects_processed": 0,
        "bugs_processed": 0,
        "failures_captured": 0,
        "compile_errors": 0,
        "tests_passed": 0,
    }
    
    for project, total_bugs in projects.items():
        print(f"\n{'='*60}")
        print(f"Processing: {project}")
        print(f"{'='*60}")
        
        num_bugs = min(total_bugs, max_bugs)
        print(f"  Processing {num_bugs} bugs")
        stats["projects_processed"] += 1
        
        project_failures = []
        
        for bug_id in range(1, num_bugs + 1):
            print(f"\n  Bug {bug_id}/{num_bugs}:", end=" ", flush=True)
            stats["bugs_processed"] += 1
            
            # Checkout (verbose for first bug per project to show errors)
            verbose = (bug_id == 1)
            if not checkout_bug(project, bug_id, WORK_DIR, verbose=verbose):
                print("checkout failed")
                continue
            
            # Compile
            if not compile_bug(project, bug_id, WORK_DIR):
                print("compile failed")
                stats["compile_errors"] += 1
                continue
            
            # Run tests
            result = run_tests(project, bug_id, WORK_DIR)
            
            if result["success"]:
                print("PASSED (unexpected)")
                stats["tests_passed"] += 1
                continue
            
            # Process each failing test
            for test_name in result["failing_tests"][:20]:  # Limit to 20 tests per bug
                test_output = get_test_output(project, bug_id, WORK_DIR, test_name)
                parsed = parse_java_failure(test_output, test_name, project, bug_id)
                
                if parsed:
                    stats["failures_captured"] += 1
                    
                    build_id = f"defects4j_{project}_multi"
                    
                    failure_data = {
                        "build_id": build_id,
                        "failure_id": f"{build_id}::{parsed['test_name']}",
                        "test_name": parsed["test_name"],
                        "exception_type": parsed["exception_type"],
                        "exception_message": parsed["exception_message"],
                        "stack_trace": parsed["stack_trace"],
                        "frames": parsed["frames"],
                        "stderr": result["output"][-2000:],
                        "stdout": "",
                        "project": project,
                        "bug_ids": [f"{project}-{bug_id}"],
                        "timestamp": datetime.now().isoformat(),
                        "ground_truth": {
                            "root_cause_id": f"{project}-{bug_id}",
                            "changed_files": [],
                            "fault_location_hint": [],
                        },
                        "language": "java",
                    }
                    
                    project_failures.append(failure_data)
            
            if result["failing_tests"]:
                print(f"✓ {len(result['failing_tests'])} failing tests")
            else:
                print("no failing tests found")
            
            # Cleanup to save disk space
            bug_dir = WORK_DIR / f"{project}_{bug_id}"
            if bug_dir.exists():
                shutil.rmtree(bug_dir, ignore_errors=True)
        
        if project_failures:
            all_failures.extend(project_failures)
            
            build_data = {
                "build_id": f"defects4j_{project}_multi",
                "project": project,
                "bug_ids": list(set(f["ground_truth"]["root_cause_id"] for f in project_failures)),
                "num_failures": len(project_failures),
                "timestamp": datetime.now().isoformat(),
                "language": "java",
            }
            all_builds.append(build_data)
            
            print(f"\n  ✓ Captured {len(project_failures)} failures from {project}")
    
    # Save output
    if all_failures:
        failures_path = OUTPUT_DIR / "failure_instances_defects4j.jsonl"
        builds_path = OUTPUT_DIR / "builds_defects4j.jsonl"
        
        with open(failures_path, "w") as f:
            for failure in all_failures:
                f.write(json.dumps(failure, default=str) + "\n")
        
        with open(builds_path, "w") as f:
            for build in all_builds:
                f.write(json.dumps(build, default=str) + "\n")
        
        print(f"\n{'='*60}")
        print("Dataset saved!")
        print(f"  - {failures_path}: {len(all_failures)} failures")
        print(f"  - {builds_path}: {len(all_builds)} builds")
    
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Done!")


if __name__ == "__main__":
    main()

