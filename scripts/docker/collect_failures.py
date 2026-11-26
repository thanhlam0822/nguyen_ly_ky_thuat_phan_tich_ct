#!/usr/bin/env python3
"""Collect real BugsInPy failures from inside Docker container.

This script runs inside the Docker container with Python 3.8 to capture
the actual bugs (not Python version compatibility issues).
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

BUGSINPY_PATH = Path("/bugsinpy")
OUTPUT_DIR = Path("/output")
WORK_DIR = Path("/tmp/bugsinpy_work")

# Projects to skip due to incompatible dependencies (TensorFlow 1.x, etc.)
SKIP_PROJECTS = {
    "keras",      # Requires TensorFlow 1.15 (Python 3.7 only)
    "spacy",      # Heavy C dependencies
    "scrapy",     # Complex Twisted dependencies
    "matplotlib", # Requires specific numpy/C builds
    "pandas",     # Heavy numpy dependency issues
}


def get_all_projects():
    """Get list of all BugsInPy projects."""
    projects_dir = BUGSINPY_PATH / "projects"
    return sorted([p.name for p in projects_dir.iterdir() if p.is_dir()])


def get_bug_ids(project: str, max_bugs: int = 100):
    """Get list of bug IDs for a project."""
    bugs_dir = BUGSINPY_PATH / "projects" / project / "bugs"
    if not bugs_dir.exists():
        return []
    
    bug_ids = sorted(
        [d.name for d in bugs_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x)
    )
    return bug_ids[:max_bugs]


def run_command(cmd, cwd=None, timeout=300):
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


def checkout_bug(project: str, bug_id: str, work_dir: Path) -> bool:
    """Checkout a buggy version."""
    bug_work_dir = work_dir / f"{project}_{bug_id}"
    if bug_work_dir.exists():
        shutil.rmtree(bug_work_dir)
    bug_work_dir.mkdir(parents=True)
    
    result = run_command(
        f"bugsinpy-checkout -p {project} -i {bug_id} -v 0 -w {bug_work_dir}",
        cwd=str(BUGSINPY_PATH),
        timeout=300,
    )
    
    return result["success"] or (bug_work_dir / project).exists()


def compile_bug(project: str, bug_id: str, work_dir: Path) -> bool:
    """Compile/setup a buggy version."""
    project_dir = work_dir / f"{project}_{bug_id}" / project
    if not project_dir.exists():
        return False
    
    result = run_command(
        "bugsinpy-compile",
        cwd=str(project_dir),
        timeout=600,
    )
    
    return (project_dir / "bugsinpy_compile_flag").exists()


def run_test(project: str, bug_id: str, work_dir: Path) -> dict:
    """Run test and capture output."""
    project_dir = work_dir / f"{project}_{bug_id}" / project
    if not project_dir.exists():
        return {"success": False, "output": "Project directory not found"}
    
    # Run bugsinpy-test
    result = run_command(
        "bugsinpy-test",
        cwd=str(project_dir),
        timeout=300,
    )
    
    output = result["stdout"] + "\n" + result["stderr"]
    
    return {
        "success": result["returncode"] == 0 and "PASSED" in output,
        "output": output,
        "returncode": result["returncode"],
    }


def parse_failure(output: str, project: str, bug_id: str) -> dict | None:
    """Parse test output to extract failure information."""
    
    # Check if there's actually a failure
    if not output:
        return None
    
    # Look for actual test failures, not just any output
    has_failure = any(marker in output for marker in [
        "FAILED", "ERROR", "Error", "Exception", "Traceback",
        "AssertionError", "assert", "FAILURES"
    ])
    
    # Skip if test passed
    if "PASSED" in output and "FAILED" not in output and not has_failure:
        return None
    
    if not has_failure:
        return None
    
    # Extract exception type and message
    exc_type = "TestFailure"
    exc_message = ""
    
    patterns = [
        r'E\s+(\w+(?:Error|Exception|Failure)):\s*(.+?)(?:\n|$)',
        r'(\w+(?:Error|Exception|Failure)):\s*(.+?)(?:\n|$)',
        r'raise\s+(\w+(?:Error|Exception))\((.*?)\)',
        r'(\w+Error):\s+(.+?)(?:\n|$)',
        r'(AssertionError)(?::\s*(.+?))?(?:\n|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, output)
        if match:
            exc_type = match.group(1)
            if len(match.groups()) > 1 and match.group(2):
                exc_message = match.group(2).strip()[:500]
            break
    
    # Special handling for assertion failures
    if "AssertionError" in output or "assert " in output.lower():
        exc_type = "AssertionError"
        assert_match = re.search(r'(?:E\s+)?assert\s+(.+?)(?:\n|$)', output)
        if assert_match:
            exc_message = assert_match.group(1).strip()[:200]
    
    # Extract stack frames
    frames = []
    frame_pattern = re.compile(r'File\s+"([^"]+)",\s+line\s+(\d+),\s+in\s+(\S+)')
    for match in frame_pattern.finditer(output):
        frames.append({
            "file": match.group(1),
            "line": int(match.group(2)),
            "func": match.group(3),
        })
    
    # Pytest format
    pytest_frame = re.compile(r'(\S+\.py):(\d+):\s+in\s+(\S+)')
    for match in pytest_frame.finditer(output):
        frame = {
            "file": match.group(1),
            "line": int(match.group(2)),
            "func": match.group(3),
        }
        if frame not in frames:
            frames.append(frame)
    
    # Get test name
    test_name = f"{project}_bug_{bug_id}"
    test_match = re.search(r'(?:FAILED|ERROR)\s+(\S+::\S+)', output)
    if test_match:
        test_name = test_match.group(1)
    
    # Get stack trace section
    stack_trace = ""
    tb_match = re.search(
        r'(Traceback \(most recent call last\):.*?)(?:\n\n|\n={3,})',
        output,
        re.DOTALL
    )
    if tb_match:
        stack_trace = tb_match.group(1)
    else:
        # Get error section
        error_section = re.search(
            r'(_{10,}\s*(?:ERROR|FAILED).*?)(?:={10,}|short test summary)',
            output,
            re.DOTALL
        )
        if error_section:
            stack_trace = error_section.group(1)
        else:
            stack_trace = output[-5000:]
    
    return {
        "test_name": test_name,
        "exception_type": exc_type,
        "exception_message": exc_message,
        "stack_trace": stack_trace[:5000],
        "frames": frames,
    }


def main():
    """Main function to collect all failures."""
    print("=" * 60)
    print("BugsInPy Real Failure Collection (Python 3.8)")
    print("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    projects = os.environ.get("PROJECTS", "").split(",")
    if not projects or projects == [""]:
        projects = get_all_projects()
    
    max_bugs = int(os.environ.get("MAX_BUGS", "100"))
    
    print(f"\nProjects: {', '.join(projects)}")
    print(f"Max bugs per project: {max_bugs}")
    print(f"Output directory: {OUTPUT_DIR}")
    
    all_failures = []
    all_builds = []
    stats = {
        "projects_processed": 0,
        "bugs_processed": 0,
        "failures_captured": 0,
        "tests_passed": 0,
        "errors": 0,
    }
    
    for project in projects:
        # Skip projects with incompatible dependencies
        if project in SKIP_PROJECTS:
            print(f"\n{'='*60}")
            print(f"Skipping: {project} (incompatible dependencies)")
            print(f"{'='*60}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {project}")
        print(f"{'='*60}")
        
        bug_ids = get_bug_ids(project, max_bugs)
        if not bug_ids:
            print(f"  No bugs found for {project}")
            continue
        
        print(f"  Found {len(bug_ids)} bugs")
        stats["projects_processed"] += 1
        
        project_failures = []
        
        for bug_id in bug_ids:
            print(f"\n  Bug {bug_id}:", end=" ", flush=True)
            stats["bugs_processed"] += 1
            
            # Checkout
            if not checkout_bug(project, bug_id, WORK_DIR):
                print("checkout failed")
                stats["errors"] += 1
                continue
            
            # Compile
            if not compile_bug(project, bug_id, WORK_DIR):
                print("compile failed")
                stats["errors"] += 1
                continue
            
            # Run test
            result = run_test(project, bug_id, WORK_DIR)
            
            if result["success"]:
                print("PASSED (unexpected)")
                stats["tests_passed"] += 1
                continue
            
            # Parse failure
            parsed = parse_failure(result["output"], project, bug_id)
            
            if parsed:
                stats["failures_captured"] += 1
                
                build_id = f"bugsinpy_{project}_multi"
                
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
                }
                
                project_failures.append(failure_data)
                print(f"✓ {parsed['exception_type']}: {parsed['exception_message'][:40]}")
            else:
                print("no failure parsed")
                stats["errors"] += 1
            
            # Cleanup to save disk space
            bug_work_dir = WORK_DIR / f"{project}_{bug_id}"
            if bug_work_dir.exists():
                shutil.rmtree(bug_work_dir, ignore_errors=True)
        
        if project_failures:
            all_failures.extend(project_failures)
            
            build_data = {
                "build_id": f"bugsinpy_{project}_multi",
                "project": project,
                "bug_ids": [f["ground_truth"]["root_cause_id"] for f in project_failures],
                "num_failures": len(project_failures),
                "timestamp": datetime.now().isoformat(),
            }
            all_builds.append(build_data)
            
            print(f"\n  ✓ Captured {len(project_failures)} failures")
    
    # Save output
    if all_failures:
        failures_path = OUTPUT_DIR / "failure_instances_py38.jsonl"
        builds_path = OUTPUT_DIR / "builds_py38.jsonl"
        
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

