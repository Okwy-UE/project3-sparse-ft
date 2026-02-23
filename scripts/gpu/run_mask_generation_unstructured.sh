#!/usr/bin/env bash
#
# Generate unstructured magnitude-based masks for all model-task pairs
# This script runs on GPU nodes to generate masks for use on Cerebras CS-3
#
# Usage:
#   bash run_mask_generation_unstructured.sh [options]
#
# Options:
#   --dry-run            Show what would be done without actually doing it
#   --cpu                Force CPU-only execution
#   --no-plot            Skip histogram generation
#   --allow-download     Allow downloading models from HuggingFace (default: use cached only)
#

set -euo pipefail

# ============================================================================
# Configuration
# ============================================================================

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Navigate to project root
cd "$PROJECT_ROOT"

# Output directory
MASKS_DIR="$PROJECT_ROOT/masks"
mkdir -p "$MASKS_DIR"

# ============================================================================
# Environment Setup
# ============================================================================

echo "========================================================================"
echo "Unstructured Mask Generation for All Model-Task Pairs"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Output directory: $MASKS_DIR"
echo ""

# Check if we're on GPU node
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
    DEVICE="cuda"
else
    echo "No GPU detected, using CPU"
    DEVICE="cpu"
fi

echo ""

# Check Python environment
echo "Python: $(which python3)"
echo "Python version: $(python3 --version)"
echo ""

# Check required packages
echo "Checking dependencies..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}')" || {
    echo "ERROR: PyTorch not installed"
    exit 1
}

python3 -c "import transformers; print(f'Transformers: {transformers.__version__}')" || {
    echo "WARNING: transformers not installed, installing..."
    pip install transformers --quiet
}

echo ""

# ============================================================================
# Parse Arguments
# ============================================================================

DRY_RUN=""
GPU_ID="${GPU_ID:-0}"
NO_PLOT=""
FORCE_CPU=""
ALLOW_DOWNLOAD=""
MIXTRAL_4BIT=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --no-plot)
            NO_PLOT="--no-plot"
            shift
            ;;
        --cpu)
            FORCE_CPU="--device cpu"
            DEVICE="cpu"
            shift
            ;;
        --gpu-id)
            GPU_ID="$2"
            shift 2
            ;;
        --allow-download)
            ALLOW_DOWNLOAD="--allow-download"
            shift
            ;;
        --mixtral-4bit)
            MIXTRAL_4BIT="--mixtral-4bit"
            shift
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --dry-run            Show what would be done without actually doing it"
            echo "  --no-plot            Skip histogram generation"
            echo "  --cpu                Force CPU-only execution"
            echo "  --gpu-id <id>        Single GPU index to use (default: 0)"
            echo "  --allow-download     Allow downloading models from HuggingFace (default: use cached only)"
            echo "  --mixtral-4bit       Load Mixtral in 4-bit to reduce GPU memory"
            echo "  --help               Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Force single-GPU, non-distributed runtime when using CUDA.
if [[ "$DEVICE" == "cuda" ]]; then
    unset LOCAL_RANK RANK WORLD_SIZE MASTER_ADDR MASTER_PORT
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$GPU_ID"
fi

# ============================================================================
# Run Mask Generation
# ============================================================================

echo "========================================================================"
echo "Starting mask generation..."
echo "========================================================================"
echo ""

# Models and tasks (Phoenix-compatible)
MODELS="mixtral"
TASKS="boolq hellaswag gsm8k"
SPARSITIES="0.25 0.50 0.75"

echo "Configuration:"
echo "  Models: $MODELS"
echo "  Tasks: $TASKS"
echo "  Sparsities: 25%, 50%, 75%"
echo "  Method: unstructured"
echo "  Importance: magnitude"
echo "  Device: $DEVICE"
echo "  Cache mode: local only (using previously downloaded models)"
echo ""

# Run the Python script
# Note: By default, this will use locally cached models only (no download)
# Add --allow-download flag to enable downloading from HuggingFace
python3 "$SCRIPT_DIR/generate_all_masks_unstructured.py" \
    --output_dir "$MASKS_DIR" \
    --models $MODELS \
    --tasks $TASKS \
    --sparsities $SPARSITIES \
    --device "$DEVICE" \
    $FORCE_CPU \
    $NO_PLOT \
    $ALLOW_DOWNLOAD \
    $MIXTRAL_4BIT \
    $DRY_RUN

EXIT_CODE=$?

# ============================================================================
# Summary
# ============================================================================

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "SUCCESS: Mask generation complete"
    echo "========================================================================"
    echo ""
    echo "Masks saved to: $MASKS_DIR"
    echo ""
    echo "Directory structure:"
    find "$MASKS_DIR" -type f -name "*.pt" -o -name "*.npz" -o -name "*.json" | sort
    echo ""
    echo "Next steps:"
    echo "  1. Verify masks with: python scripts/cs3/validate_sparsity.py"
    echo "  2. Transfer masks to Cerebras environment (already in git)"
    echo "  3. Run sparse training with: bash scripts/cs3/run_sparse_experiments.sh"
    echo ""
else
    echo ""
    echo "========================================================================"
    echo "ERROR: Mask generation failed with exit code $EXIT_CODE"
    echo "========================================================================"
    echo ""
    echo "Check the output above for error messages"
    echo ""
fi

exit $EXIT_CODE
