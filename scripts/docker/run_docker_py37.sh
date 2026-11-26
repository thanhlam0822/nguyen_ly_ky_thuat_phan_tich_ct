#!/bin/bash
# Build and run Docker container for Python 3.7 projects (Keras, etc.)
#
# Usage:
#   ./scripts/docker/run_docker_py37.sh [max_bugs]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

MAX_BUGS="${1:-100}"

echo "============================================================"
echo "BugsInPy Docker Collection - Python 3.7 Projects"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Projects: keras (Python 3.7 required)"
echo "Max bugs per project: $MAX_BUGS"
echo ""

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/data/real_bugsinpy_dataset"
mkdir -p "$OUTPUT_DIR"

# Build Docker image
echo "Building Docker image (Python 3.7)..."
docker build \
    -f "$SCRIPT_DIR/Dockerfile.py37" \
    -t hytri-bugsinpy-py37 \
    "$PROJECT_ROOT"

echo ""
echo "Running collection in Docker container..."
echo ""

# Run container
docker run --rm \
    -v "$OUTPUT_DIR:/output" \
    -e "MAX_BUGS=$MAX_BUGS" \
    hytri-bugsinpy-py37

echo ""
echo "============================================================"
echo "Collection complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================"

if [ -f "$OUTPUT_DIR/failure_instances_py37.jsonl" ]; then
    FAILURES=$(wc -l < "$OUTPUT_DIR/failure_instances_py37.jsonl")
    echo "Captured $FAILURES failures (Python 3.7 projects)"
fi



