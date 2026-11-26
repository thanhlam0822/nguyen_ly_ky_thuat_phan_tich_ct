#!/bin/bash
# Build and run Docker container to collect real BugsInPy failures
#
# Usage:
#   ./scripts/docker/run_docker_collection.sh [projects] [max_bugs]
#
# Examples:
#   ./scripts/docker/run_docker_collection.sh                    # All projects, all bugs
#   ./scripts/docker/run_docker_collection.sh "thefuck,black" 10 # Specific projects, 10 bugs each

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PROJECTS="${1:-}"
MAX_BUGS="${2:-100}"

echo "============================================================"
echo "BugsInPy Docker Collection"
echo "============================================================"
echo "Project root: $PROJECT_ROOT"
echo "Projects: ${PROJECTS:-all}"
echo "Max bugs per project: $MAX_BUGS"
echo ""

# Create output directory
OUTPUT_DIR="$PROJECT_ROOT/data/real_bugsinpy_dataset"
mkdir -p "$OUTPUT_DIR"

# Force rebuild Docker image (to pick up Dockerfile changes)
echo "Building Docker image (force rebuild)..."
docker rmi hytri-bugsinpy 2>/dev/null || true
docker build \
    -f "$SCRIPT_DIR/Dockerfile.bugsinpy" \
    -t hytri-bugsinpy \
    "$PROJECT_ROOT"

echo ""
echo "Running collection in Docker container..."
echo ""

# Run container
docker run --rm \
    -v "$OUTPUT_DIR:/output" \
    -e "PROJECTS=$PROJECTS" \
    -e "MAX_BUGS=$MAX_BUGS" \
    hytri-bugsinpy

echo ""
echo "============================================================"
echo "Collection complete!"
echo "Output saved to: $OUTPUT_DIR"
echo "============================================================"

# Show results
if [ -f "$OUTPUT_DIR/failure_instances_py38.jsonl" ]; then
    FAILURES=$(wc -l < "$OUTPUT_DIR/failure_instances_py38.jsonl")
    echo "Captured $FAILURES failures (Python 3.8 projects)"
fi

