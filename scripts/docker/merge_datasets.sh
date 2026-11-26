#!/bin/bash
# Merge all datasets (BugsInPy Python + Defects4J Java) into one
#
# Usage:
#   ./scripts/docker/merge_datasets.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_DIR="$PROJECT_ROOT/data/real_bugsinpy_dataset"

echo "============================================================"
echo "Merging All Datasets (BugsInPy + Defects4J)"
echo "============================================================"

# Initialize empty files
> "$OUTPUT_DIR/failure_instances.jsonl"
> "$OUTPUT_DIR/builds.jsonl"

TOTAL_FAILURES=0

# Add BugsInPy Python 3.8 failures
if [ -f "$OUTPUT_DIR/failure_instances_py38.jsonl" ]; then
    COUNT=$(wc -l < "$OUTPUT_DIR/failure_instances_py38.jsonl")
    cat "$OUTPUT_DIR/failure_instances_py38.jsonl" >> "$OUTPUT_DIR/failure_instances.jsonl"
    cat "$OUTPUT_DIR/builds_py38.jsonl" >> "$OUTPUT_DIR/builds.jsonl" 2>/dev/null
    echo "✓ Added BugsInPy (Python 3.8): $COUNT failures"
    TOTAL_FAILURES=$((TOTAL_FAILURES + COUNT))
fi

# Add BugsInPy Python 3.7 failures (if any)
if [ -f "$OUTPUT_DIR/failure_instances_py37.jsonl" ]; then
    COUNT=$(wc -l < "$OUTPUT_DIR/failure_instances_py37.jsonl")
    cat "$OUTPUT_DIR/failure_instances_py37.jsonl" >> "$OUTPUT_DIR/failure_instances.jsonl"
    cat "$OUTPUT_DIR/builds_py37.jsonl" >> "$OUTPUT_DIR/builds.jsonl" 2>/dev/null
    echo "✓ Added BugsInPy (Python 3.7): $COUNT failures"
    TOTAL_FAILURES=$((TOTAL_FAILURES + COUNT))
fi

# Add Defects4J Java failures
if [ -f "$OUTPUT_DIR/failure_instances_defects4j.jsonl" ]; then
    COUNT=$(wc -l < "$OUTPUT_DIR/failure_instances_defects4j.jsonl")
    cat "$OUTPUT_DIR/failure_instances_defects4j.jsonl" >> "$OUTPUT_DIR/failure_instances.jsonl"
    cat "$OUTPUT_DIR/builds_defects4j.jsonl" >> "$OUTPUT_DIR/builds.jsonl" 2>/dev/null
    echo "✓ Added Defects4J (Java): $COUNT failures"
    TOTAL_FAILURES=$((TOTAL_FAILURES + COUNT))
fi

if [ "$TOTAL_FAILURES" -eq 0 ]; then
    echo "✗ No failure files found!"
    exit 1
fi

echo ""
echo "============================================================"
echo "Dataset Summary"
echo "============================================================"
echo "Total failures: $TOTAL_FAILURES"
echo "Output: $OUTPUT_DIR/failure_instances.jsonl"
echo ""
echo "To run HyTri evaluation:"
echo "  export OPENAI_API_KEY='your-key'"
echo "  cd $PROJECT_ROOT"
echo "  .venv/bin/python scripts/run_evaluation.py --dataset-path data/real_bugsinpy_dataset/failure_instances.jsonl"

