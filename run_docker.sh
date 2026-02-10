#!/bin/bash
# ============================================================================
# Unsloth GPU Benchmark — Docker Runner (Linux / WSL / DGX)
# ============================================================================
# Usage:
#   ./run_docker.sh --run <profile>            # Headless (SSH-friendly)
#   ./run_docker.sh --run <profile> --dgx      # ARM64 Blackwell (DGX Station GB300)
#   ./run_docker.sh --interactive <profile>     # Jupyter Lab mode
#
# Profiles: small | medium | large | xlarge | stress | all
#
# ARM64 Blackwell systems (DGX Station GB300) need a custom Dockerfile.
# All other GPUs (RTX 5090, RTX PRO 6000, etc.) use unsloth/unsloth from Docker Hub.
# ============================================================================

set -e

# ── Parse arguments ──────────────────────────────────────────────────────────
MODE="--run"
PROFILE="large"
DGX_STATION=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run)        MODE="--run";         shift ;;
        --interactive) MODE="--interactive"; shift ;;
        --dgx)        DGX_STATION=true;     shift ;;
        *)            PROFILE="$1";         shift ;;
    esac
done

CONTAINER_NAME="unsloth-benchmark"
WORK_DIR="$(pwd)/work"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$WORK_DIR"

# Copy benchmark script to work directory so it's available inside the container
cp "$SCRIPT_DIR/benchmark.py" "$WORK_DIR/benchmark.py"

# ── Select Docker image ─────────────────────────────────────────────────────
if [ "$DGX_STATION" = true ]; then
    IMAGE_NAME="unsloth-dgx"
    echo "============================================================"
    echo "  Unsloth GPU Benchmark — ARM64 Blackwell Mode (DGX)"
    echo "============================================================"
    echo "  Profile:   $PROFILE"
    echo "  Image:     $IMAGE_NAME (custom build)"
    echo "  Work Dir:  $WORK_DIR"
    echo "============================================================"

    # Build the DGX Station image if it doesn't exist
    if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
        echo ""
        echo "Building ARM64 Blackwell Docker image (first time only, ~20-30 min)..."
        echo "This builds Triton + xFormers from source for aarch64/Blackwell."
        echo ""
        docker build -f "$SCRIPT_DIR/Dockerfile.dgx_station" -t "$IMAGE_NAME" "$SCRIPT_DIR"
        echo ""
        echo "Build complete!"
    else
        echo ""
        echo "Using existing $IMAGE_NAME image."
    fi
else
    IMAGE_NAME="unsloth/unsloth"
    echo "============================================================"
    echo "  Unsloth GPU Benchmark — Standard GPU Mode"
    echo "============================================================"
    echo "  Profile:   $PROFILE"
    echo "  Image:     $IMAGE_NAME (Docker Hub)"
    echo "  Work Dir:  $WORK_DIR"
    echo "============================================================"
fi

# Stop any existing container with the same name
docker rm -f "$CONTAINER_NAME" 2>/dev/null || true

# ── Run ──────────────────────────────────────────────────────────────────────
if [ "$MODE" = "--run" ]; then
    echo ""
    echo "Running benchmark headless (profile: $PROFILE)..."
    echo "No GUI or Jupyter needed — pure SSH-friendly."
    echo ""
    docker run --rm \
        --name "$CONTAINER_NAME" \
        --entrypoint bash \
        -v "$WORK_DIR":/workspace/work \
        --gpus all \
        "$IMAGE_NAME" \
        -c "cd /workspace/work && python benchmark.py --profile $PROFILE"

    echo ""
    echo "============================================================"
    echo "  Done! Results saved in: $WORK_DIR/"
    ls -la "$WORK_DIR"/benchmark_*.json 2>/dev/null || echo "  (no result files found)"
    echo "============================================================"
else
    echo ""
    echo "Starting Unsloth container with Jupyter Lab..."
    echo ""
    docker run -d \
        --name "$CONTAINER_NAME" \
        -e JUPYTER_PASSWORD="benchmark" \
        -p 8888:8888 -p 2222:22 \
        -v "$WORK_DIR":/workspace/work \
        --gpus all \
        "$IMAGE_NAME"

    echo ""
    echo "============================================================"
    echo "  Container started!"
    echo "  Jupyter Lab:  http://localhost:8888  (password: benchmark)"
    echo "  SSH:          ssh -p 2222 unsloth@localhost"
    echo ""
    echo "  To run the benchmark inside the container:"
    echo "    docker exec -it $CONTAINER_NAME bash"
    echo "    cd /workspace/work && python benchmark.py --profile $PROFILE"
    echo ""
    echo "  Or run headless instead:"
    echo "    ./run_docker.sh --run $PROFILE"
    echo "============================================================"
fi
