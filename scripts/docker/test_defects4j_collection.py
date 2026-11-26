#!/usr/bin/env python3
"""Debug script to test Defects4J data collection on a single bug."""

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

WORK_DIR = Path("/tmp/defects4j_work")


def run_command(cmd, cwd=None, timeout=600):
    """Run a command and return output."""
    print(f"  Running: {cmd[:80]}...")
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
        return {"success": False, "stdout": "", "stderr": f"Timeout", "returncode": -1}
    except Exception as e:
        return {"success": False, "stdout": "", "stderr": str(e), "returncode": -1}


def main():
    print("=" * 60)
    print("Defects4J Collection Debug Test")
    print("=" * 60)
    
    # Check Java and defects4j
    os.system("java -version 2>&1 | head -1")
    os.system("which defects4j")
    
    # Test project and bug
    project = os.environ.get("PROJECT", "Lang")
    bug_id = int(os.environ.get("BUG_ID", "1"))
    
    print(f"\nTesting: {project} bug {bug_id}")
    
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    bug_dir = WORK_DIR / f"{project}_{bug_id}"
    
    # Clean up previous attempt
    if bug_dir.exists():
        shutil.rmtree(bug_dir, ignore_errors=True)
    bug_dir.mkdir(parents=True)
    
    # Step 1: Checkout
    print("\n1. CHECKOUT")
    result = run_command(f"defects4j checkout -p {project} -v {bug_id}b -w {bug_dir}")
    print(f"   Success: {result['success']}")
    if not result["success"]:
        print(f"   Error: {result['stderr'][:500]}")
        return
    
    # Step 2: Compile
    print("\n2. COMPILE")
    result = run_command("defects4j compile", cwd=str(bug_dir))
    print(f"   Success: {result['success']}")
    if not result["success"]:
        print(f"   Error: {result['stderr'][:500]}")
        return
    
    # Step 3: Run tests
    print("\n3. RUN TESTS (defects4j test)")
    result = run_command("defects4j test", cwd=str(bug_dir))
    print(f"   Return code: {result['returncode']}")
    print(f"   Success: {result['success']}")
    
    # Check failing_tests file
    failing_tests_file = bug_dir / "failing_tests"
    print(f"\n4. FAILING TESTS FILE")
    print(f"   Exists: {failing_tests_file.exists()}")
    
    if failing_tests_file.exists():
        with open(failing_tests_file) as f:
            content = f.read()
        print(f"   Content ({len(content)} chars):")
        print("   ---")
        for line in content.split('\n')[:10]:
            print(f"   {line}")
        print("   ---")
        
        # Parse test names
        failing_tests = [line.strip() for line in content.split('\n') if line.strip()]
        print(f"\n   Parsed {len(failing_tests)} failing tests:")
        for t in failing_tests[:5]:
            print(f"     - {t}")
    
    # Step 4: Get detailed output for first failing test
    if failing_tests_file.exists():
        with open(failing_tests_file) as f:
            test_lines = [l.strip() for l in f if l.strip()]
        
        if test_lines:
            test_name = test_lines[0]
            print(f"\n5. GETTING DETAILS FOR: {test_name}")
            
            # The format might be "--- package.Class::method"
            # We need to extract just "package.Class::method"
            if test_name.startswith("---"):
                test_name = test_name[3:].strip()
            
            print(f"   Clean test name: {test_name}")
            
            # Use proper escaping for shell
            # Quote the test name to handle special characters
            escaped_name = test_name.replace("'", "'\\''")
            result = run_command(
                f"defects4j test -t '{escaped_name}'",
                cwd=str(bug_dir)
            )
            
            print(f"\n   Return code: {result['returncode']}")
            print(f"\n   STDOUT (first 1500 chars):")
            print("   ---")
            print(result['stdout'][:1500])
            print("   ---")
            
            print(f"\n   STDERR (first 1500 chars):")
            print("   ---")
            print(result['stderr'][:1500])
            print("   ---")
            
            # Try to find stack trace
            combined = result['stdout'] + result['stderr']
            
            print(f"\n6. PARSING OUTPUT")
            
            # Look for "at " lines (stack frames)
            at_lines = re.findall(r'^\s*at .+$', combined, re.MULTILINE)
            print(f"   Found {len(at_lines)} 'at' lines (stack frames)")
            for line in at_lines[:5]:
                print(f"     {line.strip()}")
            
            # Look for exception patterns
            exc_pattern = r'(\w+(?:Error|Exception)):\s*(.+?)(?:\n|$)'
            exc_matches = re.findall(exc_pattern, combined)
            print(f"\n   Found {len(exc_matches)} exception patterns:")
            for exc_type, exc_msg in exc_matches[:3]:
                print(f"     {exc_type}: {exc_msg[:80]}")
            
            # Look for assertion failures
            if "expected:" in combined.lower():
                assert_match = re.search(r'expected:<(.+?)> but was:<(.+?)>', combined)
                if assert_match:
                    print(f"\n   AssertionError: expected:<{assert_match.group(1)[:50]}> but was:<{assert_match.group(2)[:50]}>")
    
    print("\n" + "=" * 60)
    print("Test complete!")


if __name__ == "__main__":
    main()


