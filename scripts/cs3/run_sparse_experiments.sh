#!/usr/bin/env bash
#
# Run sparse fine-tuning experiments on Cerebras CS-3
# 
# Weeks 6-8 pipeline:
# 1. Compute pruning masks (Week 6)
# 2. Train sparse LoRA (Week 7)
# 3. Full sparse sweep (Week 8)
#

set -euo pipefail

# ============================================================================
# Environment
# ============================================================================
export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export https_proxy="$HTTPS_PROXY"

# Activate virtual environment
source "$HOME/R_2.6.0/venv_cerebras_pt/bin/activate"

# Navigate to project directory
cd "$HOME/project3-sparse-ft"

# Pull latest changes
git pull || true

# ============================================================================
# Configuration
# ============================================================================
: "${NUM_CSX:=2}"
: "${RESULTS_ROOT:=$PWD/results/runs}"
: "${MASKS_ROOT:=$PWD/masks}"

mkdir -p "$RESULTS_ROOT" "$MASKS_ROOT" "$PWD/results/tables"

# Models and tasks (Phoenix-compatible)
MODELS=("llama" "mistral")
TASKS=("boolq" "hellaswag" "gsm8k")

# Sparsity levels (comprehensive sweep)
SPARSITIES=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

# Pruning methods
METHODS=("unstructured" "structured" "random")

# ============================================================================
# Helper functions
# ============================================================================
GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
NOW_ISO="$(date -Iseconds)"

find_cfg() {
    local model="$1"
    local task="$2"
    
    local hits=""
    hits="$(find "$PWD/configs" -type f \( -name "*.yaml" -o -name "*.yml" \) \
        | grep -iE "$model" \
        | grep -iE "$task" \
        | grep -iE "cs3|csx|cerebras" \
        | grep -iE "lora|peft" || true)"
    
    if [[ -z "$hits" ]]; then
        hits="$(find "$PWD/configs" -type f \( -name "*.yaml" -o -name "*.yml" \) \
            | grep -iE "$model" \
            | grep -iE "$task" || true)"
    fi
    
    [[ -z "$hits" ]] && return 1
    
    echo "$hits" \
        | awk '{ print length($0) "\t" $0 }' \
        | sort -n \
        | head -n1 \
        | cut -f2-
}

get_checkpoint_path() {
    local model="$1"
    
    case "$model" in
        llama*)
            echo "$HOME/project3-sparse-ft/checkpoints/cs/llama_7b/model_to_cs-2.5.mdl"
            ;;
        mistral*)
            echo "$HOME/project3-sparse-ft/checkpoints/cs/mistral_7b/model_to_cs-2.5.mdl"
            ;;
        mixtral*)
            echo "$HOME/project3-sparse-ft/checkpoints/cs/mixtral_8x7b/model_to_cs-2.5.mdl"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ============================================================================
# Week 6: Compute masks
# ============================================================================
compute_masks() {
    local model="$1"
    local method="$2"
    
    echo "============================================================"
    echo "Computing masks: model=$model method=$method"
    echo "============================================================"
    
    local ckpt_path="$(get_checkpoint_path "$model")"
    if [[ -z "$ckpt_path" ]] || [[ ! -f "$ckpt_path" ]]; then
        echo "⚠️  Checkpoint not found: $ckpt_path"
        return 1
    fi
    
    local mask_dir="$MASKS_ROOT/$model"
    mkdir -p "$mask_dir"
    
    # Compute masks for all sparsity levels
    python scripts/cs3/compute_masks.py \
        --model_path "$ckpt_path" \
        --output_dir "$mask_dir" \
        --sparsity "${SPARSITIES[@]}" \
        --method "$method" \
        --importance magnitude \
        --format pt \
        --plot
    
    echo "✓ Masks computed for $model ($method)"
}

# ============================================================================
# Week 7: Train sparse LoRA
# ============================================================================
train_sparse_lora() {
    local model="$1"
    local task="$2"
    local sparsity="$3"
    local method="$4"
    local mode="$5"  # sparse_to_dense or sparse_to_sparse
    
    local run_id="cs3_${model}_${task}_${method}_s${sparsity}_${mode}_${GIT_SHA}_$(date +%Y%m%d_%H%M%S)"
    local out_dir="$RESULTS_ROOT/$run_id"
    mkdir -p "$out_dir"
    
    echo "============================================================"
    echo "RUN: $run_id"
    echo "Model: $model | Task: $task | Sparsity: $sparsity | Method: $method | Mode: $mode"
    echo "============================================================"
    
    # Find config
    local cfg=""
    if ! cfg="$(find_cfg "$model" "$task")"; then
        echo "⚠️  No config found for model=$model task=$task"
        return 1
    fi
    
    # Get mask path
    local mask_path="$MASKS_ROOT/$model/masks_sparsity${sparsity}_${method}.pt"
    if [[ ! -f "$mask_path" ]]; then
        echo "⚠️  Mask not found: $mask_path"
        echo "    Run compute_masks first"
        return 1
    fi
    
    # Copy config and metadata
    cp "$cfg" "$out_dir/config.yaml"
    echo "$GIT_SHA" > "$out_dir/git_sha.txt"
    (git status --porcelain=v1 || true) > "$out_dir/git_status.txt"
    
    # Create training script with sparse LoRA
    # NOTE: This requires integration with Cerebras ModelZoo
    # You may need to modify the config to include sparse mask loading
    
    # For now, we'll run standard training and document the sparse configuration
    cat > "$out_dir/sparse_config.json" <<EOF
{
    "model": "$model",
    "task": "$task",
    "sparsity": $sparsity,
    "method": "$method",
    "mode": "$mode",
    "mask_path": "$mask_path",
    "git_sha": "$GIT_SHA",
    "timestamp": "$NOW_ISO"
}
EOF
    
    # Run training
    cszoo fit "$cfg" \
        --num_csx="$NUM_CSX" \
        --job_labels "name=${run_id}" \
        --model_dir "$out_dir/model_dir" \
        --mount_dirs "$PWD" \
        --python_paths "$PWD/src" \
        |& tee "$out_dir/train.log"
    
    # Validate sparsity preservation (post-training)
    python scripts/cs3/validate_sparsity.py \
        --checkpoint "$out_dir/model_dir" \
        --mask_path "$mask_path" \
        --output "$out_dir/validation_report.json"
    
    echo "✓ Training completed: $run_id"
}

# ============================================================================
# Week 8: Full sparse sweep
# ============================================================================
run_sparse_sweep() {
    echo "========================================================================"
    echo "WEEK 8: Full Sparse Sweep"
    echo "========================================================================"
    
    for model in "${MODELS[@]}"; do
        for task in "${TASKS[@]}"; do
            for method in "${METHODS[@]}"; do
                for sparsity in "${SPARSITIES[@]}"; do
                    # Convert sparsity to integer for filenames
                    local sparsity_int=$(echo "$sparsity * 100" | bc | cut -d. -f1)
                    
                    # Run sparse-to-dense
                    train_sparse_lora "$model" "$task" "$sparsity_int" "$method" "sparse_to_dense" || true
                    
                    # Run sparse-to-sparse (for non-zero sparsity)
                    if (( $(echo "$sparsity > 0" | bc -l) )); then
                        train_sparse_lora "$model" "$task" "$sparsity_int" "$method" "sparse_to_sparse" || true
                    fi
                done
            done
        done
    done
}

# ============================================================================
# Main pipeline
# ============================================================================
main() {
    local mode="${1:-all}"
    
    case "$mode" in
        masks)
            echo "Computing masks for all models..."
            for model in "${MODELS[@]}"; do
                for method in "${METHODS[@]}"; do
                    compute_masks "$model" "$method" || true
                done
            done
            ;;
        
        train)
            echo "Training sparse LoRA models..."
            # Run a subset for testing
            train_sparse_lora "mistral" "boolq" "50" "unstructured" "sparse_to_dense"
            ;;
        
        sweep)
            echo "Running full sparse sweep..."
            run_sparse_sweep
            ;;
        
        all)
            echo "Running full pipeline (Weeks 6-8)..."
            
            # Week 6: Compute masks
            for model in "${MODELS[@]}"; do
                for method in "${METHODS[@]}"; do
                    compute_masks "$model" "$method" || true
                done
            done
            
            # Weeks 7-8: Train and sweep
            run_sparse_sweep
            ;;
        
        *)
            echo "Usage: $0 [masks|train|sweep|all]"
            echo ""
            echo "  masks  - Compute pruning masks (Week 6)"
            echo "  train  - Train single sparse LoRA model (Week 7)"
            echo "  sweep  - Run full sparse sweep (Week 8)"
            echo "  all    - Run complete pipeline (default)"
            exit 1
            ;;
    esac
    
    echo ""
    echo "========================================================================"
    echo "DONE"
    echo "========================================================================"
}

# Run main function
main "$@"
