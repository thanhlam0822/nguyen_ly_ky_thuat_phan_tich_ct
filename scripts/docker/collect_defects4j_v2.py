#!/usr/bin/env python3
"""Collect Defects4J failures for HyTri evaluation - v2.

Fixed to properly parse the failing_tests file format where
each test failure includes the full stack trace.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
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
            cmd, shell=True, capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "stdout": "", "stderr": "Timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def checkout_bug(project: str, bug_id: int, work_dir: Path) -> bool:
    """Checkout a buggy version."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    if bug_dir.exists():
        shutil.rmtree(bug_dir, ignore_errors=True)
    bug_dir.mkdir(parents=True)
    
    result = run_command(f"defects4j checkout -p {project} -v {bug_id}b -w {bug_dir}", timeout=300)
    return result["success"] and bug_dir.exists()


def compile_bug(project: str, bug_id: int, work_dir: Path) -> bool:
    """Compile the buggy version."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    result = run_command("defects4j compile", cwd=str(bug_dir), timeout=600)
    return result["success"]


def run_tests(project: str, bug_id: int, work_dir: Path) -> list[dict]:
    """Run tests and parse failures from failing_tests file."""
    bug_dir = work_dir / f"{project}_{bug_id}"
    
    # Run all tests
    run_command("defects4j test", cwd=str(bug_dir), timeout=600)
    
    # Parse failing_tests file
    failing_tests_file = bug_dir / "failing_tests"
    if not failing_tests_file.exists():
        return []
    
    with open(failing_tests_file) as f:
        content = f.read()
    
    # Parse format: each failure starts with "--- test.name::method"
    # followed by exception and stack trace until next "--- " or EOF
    failures = []
    blocks = re.split(r'^--- ', content, flags=re.MULTILINE)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        
        lines = block.split('\n')
        if not lines:
            continue
        
        # First line is test name (e.g., "org.apache.commons.lang3.math.NumberUtilsTest::TestLang747")
        test_name = lines[0].strip()
        if not test_name or '::' not in test_name:
            continue
        
        # Rest is the stack trace
        stack_trace = '\n'.join(lines[1:]).strip()
        
        # Parse exception from stack trace
        exc_type = "TestFailure"
        exc_message = ""
        
        # Look for exception pattern in first few lines
        for line in lines[1:5]:
            # Pattern: ExceptionType: message
            match = re.match(r'^([\w.]+(?:Error|Exception)):\s*(.*)$', line.strip())
            if match:
                exc_type = match.group(1).split('.')[-1]  # Just class name
                exc_message = match.group(2)[:500]
                break
            
            # Pattern for assertion: expected:<x> but was:<y>
            if 'expected:' in line.lower():
                exc_type = "AssertionError"
                assert_match = re.search(r'expected:<(.+?)> but was:<(.+?)>', line)
                if assert_match:
                    exc_message = f"expected:<{assert_match.group(1)[:100]}> but was:<{assert_match.group(2)[:100]}>"
                break
        
        # Extract stack frames
        frames = []
        # Pattern: at package.Class.method(File.java:123) or at package.Class.method(File.java:123) ~[...]
        frame_pattern = re.compile(r'at\s+([\w.$]+)\.([\w$<>]+)\(([\w.]+):(\d+)\)')
        for match in frame_pattern.finditer(stack_trace):
            class_name = match.group(1)
            method = match.group(2)
            file_name = match.group(3)
            line_num = int(match.group(4))
            
            frames.append({
                "file": file_name,
                "line": line_num,
                "func": f"{class_name.split('.')[-1]}.{method}",  # Simplified
            })
        
        failures.append({
            "test_name": test_name,
            "exception_type": exc_type,
            "exception_message": exc_message,
            "stack_trace": stack_trace[:5000],
            "frames": frames[:20],  # Limit frames
        })
    
    return failures


def main():
    """Main collection function."""
    print("=" * 60)
    print("Defects4J Failure Collection v2")
    print("=" * 60)
    
    os.system("java -version 2>&1 | head -1")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    projects_str = os.environ.get("PROJECTS", "")
    if projects_str:
        projects = {p: DEFECTS4J_PROJECTS.get(p, 10) for p in projects_str.split(",") if p in DEFECTS4J_PROJECTS}
    else:
        projects = DEFECTS4J_PROJECTS
    
    max_bugs = int(os.environ.get("MAX_BUGS", "100"))
    
    print(f"\nProjects: {', '.join(projects.keys())}")
    print(f"Max bugs per project: {max_bugs}")
    
    all_failures = []
    all_builds = []
    stats = {"bugs_processed": 0, "failures_captured": 0, "compile_errors": 0, "passed": 0}
    
    for project, total_bugs in projects.items():
        print(f"\n{'='*60}")
        print(f"Processing: {project} ({min(total_bugs, max_bugs)} bugs)")
        print(f"{'='*60}")
        
        project_failures = []
        
        for bug_id in range(1, min(total_bugs, max_bugs) + 1):
            print(f"\n  Bug {bug_id}: ", end="", flush=True)
            stats["bugs_processed"] += 1
            
            # Checkout
            if not checkout_bug(project, bug_id, WORK_DIR):
                print("checkout failed")
                continue
            
            # Compile
            if not compile_bug(project, bug_id, WORK_DIR):
                print("compile failed")
                stats["compile_errors"] += 1
                continue
            
            # Run tests and parse failures
            failures = run_tests(project, bug_id, WORK_DIR)
            
            if not failures:
                print("PASSED (no failures)")
                stats["passed"] += 1
                continue
            
            # Create failure instances
            for i, failure in enumerate(failures[:20]):  # Max 20 per bug
                build_id = f"defects4j_{project}_{bug_id}"
                
                failure_data = {
                    "build_id": build_id,
                    "failure_id": f"{build_id}::failure_{i}",
                    "test_name": failure["test_name"],
                    "exception_type": failure["exception_type"],
                    "exception_message": failure["exception_message"],
                    "stack_trace": failure["stack_trace"],
                    "frames": failure["frames"],
                    "stderr": "",
                    "stdout": "",
                    "project": project,
                    "bug_id": f"{project}-{bug_id}",
                    "timestamp": datetime.now().isoformat(),
                    "ground_truth": {
                        "root_cause_id": f"{project}-{bug_id}",
                        "changed_files": [],
                    },
                    "language": "java",
                }
                project_failures.append(failure_data)
                stats["failures_captured"] += 1
            
            print(f"✓ {len(failures)} failures (frames: {sum(len(f['frames']) for f in failures)})")
            
            # Cleanup
            bug_dir = WORK_DIR / f"{project}_{bug_id}"
            if bug_dir.exists():
                shutil.rmtree(bug_dir, ignore_errors=True)
        
        if project_failures:
            all_failures.extend(project_failures)
            
            # Create build entry
            unique_bugs = list(set(f["bug_id"] for f in project_failures))
            all_builds.append({
                "build_id": f"defects4j_{project}_all",
                "project": project,
                "bug_ids": unique_bugs,
                "num_failures": len(project_failures),
                "num_bugs": len(unique_bugs),
                "language": "java",
                "timestamp": datetime.now().isoformat(),
            })
    
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
    
    # Sample output
    if all_failures:
        print(f"\n{'='*60}")
        print("Sample failure:")
        print(f"{'='*60}")
        sample = all_failures[0]
        print(f"  test_name: {sample['test_name']}")
        print(f"  exception_type: {sample['exception_type']}")
        print(f"  exception_message: {sample['exception_message'][:100]}...")
        print(f"  frames: {len(sample['frames'])}")
        for frame in sample['frames'][:3]:
            print(f"    - {frame['file']}:{frame['func']}:{frame['line']}")


if __name__ == "__main__":
    main()

