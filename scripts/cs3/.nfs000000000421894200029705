#!/usr/bin/env bash
# =============================================================================
# Full Sparse LoRA Sweep on Cerebras CS-3
# =============================================================================
#
# Workflow:
#   Phase 0  – Unit-validate the sparse LoRA implementation (CPU-only)
#   Phase 1  – Apply masks to HF checkpoints → sparse HF checkpoints
#   Phase 2  – Convert sparse HF → CS-2.5 format
#   Phase 3  – Generate YAML training configs for every combination
#   Phase 4  – Launch training runs on CS-3  (cszoo fit)
#   Phase 5  – Validate sparsity preservation in trained checkpoints
#   Phase 6  – Benchmark throughput across batch sizes
#   Phase 7  – Generate Phoenix-comparable plots and tables
#
# Usage:
#   bash scripts/cs3/run_full_sparse_sweep.sh           # full sweep
#   bash scripts/cs3/run_full_sparse_sweep.sh prepare    # phases 0-3 only
#   bash scripts/cs3/run_full_sparse_sweep.sh train      # phase 4 only
#   bash scripts/cs3/run_full_sparse_sweep.sh bench      # phase 6 only
#   bash scripts/cs3/run_full_sparse_sweep.sh analyze    # phase 7 only
#   bash scripts/cs3/run_full_sparse_sweep.sh cleanup    # phase 8 only
#
# Environment:
#   ssh cerebras.alcf.anl.gov && ssh cer-usn-01
#   source ~/R_2.6.0/venv_cerebras_pt/bin/activate  (or project venv)
#   cd ~/project3-sparse-ft
#
# =============================================================================
set -euo pipefail

MODE="${1:-sweep}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

mode_requires_cszoo() {
    case "$1" in
        prepare|train|bench|sweep) return 0 ;;
        *) return 1 ;;
    esac
}

python_has_required_modules() {
    python - <<'PY' >/dev/null 2>&1
import importlib
importlib.import_module("torch")
importlib.import_module("huggingface_hub")
PY
}

# Proxy for ALCF network
export HTTPS_PROXY="${HTTPS_PROXY:-http://proxy.alcf.anl.gov:3128}"
export https_proxy="$HTTPS_PROXY"

if mode_requires_cszoo "$MODE"; then
    # Activate the Cerebras venv (skip if already active).
    # Prefer the project venv created by setup_r290_venv.sh
    # (defaults to latest stable Model Zoo release).
    if ! command -v cszoo &>/dev/null || ! python_has_required_modules; then
        for venv_path in \
            "$HOME/venv_cerebras_r290/bin/activate_cerebras" \
            "$HOME/venv_cerebras_r290/bin/activate" \
            "$HOME/R_2.6.0/venv_cerebras_pt/bin/activate" \
            "$HOME/venv_cerebras_pt/bin/activate" \
            "$PROJECT_ROOT/venv_cerebras_pt/bin/activate"; do
            if [[ -f "$venv_path" ]]; then
                . "$venv_path" 2>/dev/null || true
                if command -v cszoo &>/dev/null && python_has_required_modules; then
                    break
                fi
            fi
        done
    fi

    if ! command -v cszoo &>/dev/null; then
        echo "ERROR: cszoo not found, but mode '$MODE' requires it."
        echo "  Activate: source ~/venv_cerebras_r290/bin/activate_cerebras"
        echo "  Verify  : command -v cszoo && cszoo --help"
        echo "  Setup   : bash scripts/cs3/setup_r290_venv.sh  (defaults to Release_2.9.0)"
        exit 1
    fi

    if ! python_has_required_modules; then
        echo "ERROR: python environment is missing required modules (torch, huggingface_hub)."
        echo "  Active python: $(command -v python || echo 'NOT FOUND')"
        echo "  Activate: source ~/venv_cerebras_r290/bin/activate_cerebras"
        exit 1
    fi

    echo "Using cszoo: $(command -v cszoo)"
    echo "Using python: $(command -v python)"
fi

# Knobs
: "${NUM_CSX:=1}"
: "${MODELS:=mixtral}"
: "${HF_MIXTRAL_REPO_ID:=mistralai/Mixtral-8x7B-v0.1}"
: "${TASKS:=boolq,gsm8k,hellaswag}"
: "${SPARSITIES:=25,50,75}"
: "${MODES:=sparse_to_dense,sparse_to_sparse}"
: "${MAX_STEPS:=200}"
: "${BENCH_BATCH_SIZES:=8,16,32,64,128,256}"
: "${BENCH_EVAL_STEPS:=3}"
: "${MAX_RETRIES:=3}"
: "${RETRY_WAIT_SEC:=60}"
: "${SKIP_SUCCESSFUL:=1}"
: "${ALLOW_DENSE_FALLBACK:=0}"
: "${DELETE_FINETUNED_AFTER_EVAL:=1}"
: "${DELETE_TRAIN_ONLY_MODELS:=0}"
: "${RESULTS_ROOT:=$PROJECT_ROOT/results/masked_runs_cs}"
MIXTRAL_HF_DIR="$PROJECT_ROOT/checkpoints/hf/mixtral_8x7b"
MIXTRAL_CS_BASE_DIR="$PROJECT_ROOT/checkpoints/cs/mixtral_8x7b"

mkdir -p "$RESULTS_ROOT"

GIT_SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
export GIT_SHA

echo "================================================================"
echo " CS-3 Sparse LoRA Sweep"
echo "================================================================"
echo " Models     : $MODELS"
echo " HF model   : $HF_MIXTRAL_REPO_ID"
echo " Tasks      : $TASKS"
echo " Sparsities : $SPARSITIES"
echo " Modes      : $MODES"
echo " NUM_CSX    : $NUM_CSX"
echo " Max steps  : $MAX_STEPS"
echo " Retries    : $MAX_RETRIES (wait=${RETRY_WAIT_SEC}s)"
echo " Skip done  : $SKIP_SUCCESSFUL"
echo " Dense fb   : $ALLOW_DENSE_FALLBACK"
echo " Cleanup    : $DELETE_FINETUNED_AFTER_EVAL"
echo " Cleanup T1 : $DELETE_TRAIN_ONLY_MODELS"
echo " Bench eval : $BENCH_EVAL_STEPS step(s)"
echo " Results    : $RESULTS_ROOT"
echo " Mode       : $MODE"
echo "================================================================"

# -----------------------------------------------------------------
# Helper: convert comma-separated list to space-separated
# -----------------------------------------------------------------
to_spaces() { echo "$1" | tr ',' ' '; }
join_by_comma() {
    local IFS=','
    echo "$*"
}
csv_has_item() {
    local csv="$1"
    local needle="$2"
    case ",$csv," in
        *,"$needle",*) return 0 ;;
        *) return 1 ;;
    esac
}

ensure_mixtral_base_checkpoint() {
    local hf_dir="$MIXTRAL_HF_DIR"
    local hf_index="$hf_dir/model.safetensors.index.json"
    local hf_config="$hf_dir/config.json"
    local cs_base_dir="$MIXTRAL_CS_BASE_DIR"
    local cs_base_mdl="$cs_base_dir/model_to_cs-2.5.mdl"

    if [[ ! -f "$hf_index" || ! -f "$hf_config" ]]; then
        echo ">>> Bootstrap: downloading $HF_MIXTRAL_REPO_ID to checkpoints/hf/mixtral_8x7b ..."
        mkdir -p "$hf_dir"
        HF_REPO_ID="$HF_MIXTRAL_REPO_ID" HF_LOCAL_DIR="$hf_dir" python - <<'PY'
import os
import sys

try:
    from huggingface_hub import snapshot_download
except Exception as exc:
    sys.exit(f"huggingface_hub import failed: {exc}")

repo_id = os.environ["HF_REPO_ID"]
local_dir = os.environ["HF_LOCAL_DIR"]
snapshot_download(
    repo_id=repo_id,
    local_dir=local_dir,
    local_dir_use_symlinks=False,
    resume_download=True,
)
print(f"Downloaded {repo_id} -> {local_dir}")
PY
    else
        echo ">>> Bootstrap: [cached] HF mixtral checkpoint present."
    fi

    if [[ ! -f "$hf_index" || ! -f "$hf_config" ]]; then
        echo "ERROR: Mixtral HF checkpoint is incomplete in $hf_dir"
        exit 1
    fi

    if [[ ! -f "$cs_base_mdl" ]]; then
        echo ">>> Bootstrap: converting dense Mixtral HF -> CS-2.5 ..."
        mkdir -p "$cs_base_dir"
        cszoo checkpoint convert \
            --model mixtral \
            --src-fmt hf \
            --tgt-fmt cs-2.5 \
            --output-dir "$cs_base_dir" \
            "$hf_index" \
            --config "$hf_config"
    else
        echo ">>> Bootstrap: [cached] dense CS Mixtral checkpoint present."
    fi
}

ensure_required_artifacts() {
    echo ""
    echo ">>> Preflight: verifying required artifacts ..."

    declare -A HF_DIR_MAP=( ["llama3"]="llama3p1_8b" ["mistral"]="mistral_7b" ["mixtral"]="mixtral_8x7b" )
    declare -A DENSE_CS_DIR_MAP=( ["llama3"]="llama3p1_8b" ["mistral"]="mistral_7b" ["mixtral"]="mixtral_8x7b" )

    for model in $(to_spaces "$MODELS"); do
        if [[ -z "${HF_DIR_MAP[$model]:-}" ]]; then
            echo "ERROR: Unsupported model '$model' in MODELS=$MODELS"
            exit 1
        fi

        hf_orig="$PROJECT_ROOT/checkpoints/hf/${HF_DIR_MAP[$model]}"
        needs_masking=0

        for sp in $(to_spaces "$SPARSITIES"); do
            mask_path="$PROJECT_ROOT/masks/$model/masks_sparsity${sp}_unstructured.pt"
            if [[ ! -f "$mask_path" ]]; then
                echo "ERROR: Missing mask file: $mask_path"
                exit 1
            fi

            hf_sparse="$PROJECT_ROOT/checkpoints/hf_sparse/${model}_s${sp}"
            cs_sparse_mdl="$PROJECT_ROOT/checkpoints/cs_sparse/${model}_s${sp}/model_to_cs-2.5.mdl"
            # If sparse CS checkpoint already exists, we can skip HF masking prep.
            if [[ ! -f "$hf_sparse/masking_report.json" && ! -f "$cs_sparse_mdl" ]]; then
                needs_masking=1
            fi
        done

        if [[ "$model" == "mixtral" && "$needs_masking" == "1" ]]; then
            ensure_mixtral_base_checkpoint
        fi

        if [[ "$needs_masking" == "1" && ! -d "$hf_orig" ]]; then
            echo "ERROR: Missing HF checkpoint dir for masking: $hf_orig"
            echo "       Required for model '$model' because one or more sparse checkpoints are not cached."
            exit 1
        fi

        dense_cs_mdl="$PROJECT_ROOT/checkpoints/cs/${DENSE_CS_DIR_MAP[$model]}/model_to_cs-2.5.mdl"
        if [[ ! -f "$dense_cs_mdl" ]]; then
            echo "ERROR: Missing dense CS base checkpoint for fallback: $dense_cs_mdl"
            exit 1
        fi

        for task in $(to_spaces "$TASKS"); do
            base_yaml="$PROJECT_ROOT/configs/cerebras/generated/$model/${task}.yaml"
            if [[ ! -f "$base_yaml" ]]; then
                echo "ERROR: Missing base training config: $base_yaml"
                exit 1
            fi
        done
    done

    echo ">>> Preflight: all required artifacts are available."
}


# =================================================================
#  Phase 0 – Unit validation
# =================================================================
phase_validate() {
    echo ""
    echo ">>> Phase 0: Unit validation of sparse LoRA ..."
    python scripts/cs3/unit_validate_sparse_lora.py \
        --output "$RESULTS_ROOT/unit_validation_report.json"
    echo "    PASSED"
}


# =================================================================
#  Phase 1 – Apply masks to HF checkpoints
# =================================================================
phase_mask() {
    echo ""
    echo ">>> Phase 1: Applying masks to HF checkpoints ..."
    for model in $(to_spaces "$MODELS"); do
        for sp in $(to_spaces "$SPARSITIES"); do
            out_dir="$PROJECT_ROOT/checkpoints/hf_sparse/${model}_s${sp}"
            cs_sparse_mdl="$PROJECT_ROOT/checkpoints/cs_sparse/${model}_s${sp}/model_to_cs-2.5.mdl"
            if [[ -f "$cs_sparse_mdl" ]]; then
                echo "    [cached cs] $model s${sp}"
                continue
            fi
            if [[ -f "$out_dir/masking_report.json" ]]; then
                echo "    [cached] $model s${sp}"
                continue
            fi
            echo "    Masking: $model s${sp}"
            python scripts/cs3/apply_masks_to_checkpoint.py \
                --model "$model" --sparsity "$sp"
        done
    done
}


# =================================================================
#  Phase 2 – Convert HF-sparse → CS-2.5
# =================================================================
phase_convert() {
    echo ""
    echo ">>> Phase 2: Converting HF-sparse → CS-2.5 ..."

    declare -A CS_MODEL_MAP=( ["llama3"]="llama" ["mistral"]="mistral" ["mixtral"]="mixtral" )
    declare -A HF_DIR_MAP=( ["llama3"]="llama3p1_8b" ["mistral"]="mistral_7b" ["mixtral"]="mixtral_8x7b" )
    declare -A DENSE_CS_DIR_MAP=( ["llama3"]="llama3p1_8b" ["mistral"]="mistral_7b" ["mixtral"]="mixtral_8x7b" )

    for model in $(to_spaces "$MODELS"); do
        if [[ -z "${CS_MODEL_MAP[$model]:-}" || -z "${HF_DIR_MAP[$model]:-}" ]]; then
            echo "ERROR: Unsupported model '$model' in Phase 2 conversion."
            exit 1
        fi
        for sp in $(to_spaces "$SPARSITIES"); do
            cs_dir="$PROJECT_ROOT/checkpoints/cs_sparse/${model}_s${sp}"
            if [[ -f "$cs_dir/model_to_cs-2.5.mdl" ]]; then
                echo "    [cached] $model s${sp}"
                continue
            fi
            hf_sparse="$PROJECT_ROOT/checkpoints/hf_sparse/${model}_s${sp}"
            hf_orig="$PROJECT_ROOT/checkpoints/hf/${HF_DIR_MAP[$model]}"

            # Use the index file from the sparse dir if it was copied,
            # otherwise fall back to the original HF dir.
            idx="$hf_sparse/model.safetensors.index.json"
            [[ -f "$idx" ]] || idx="$hf_orig/model.safetensors.index.json"

            config="$hf_sparse/config.json"
            [[ -f "$config" ]] || config="$hf_orig/config.json"

            echo "    Converting: $model s${sp}"
            mkdir -p "$cs_dir"
            if cszoo checkpoint convert \
                --model "${CS_MODEL_MAP[$model]}" \
                --src-fmt hf \
                --tgt-fmt cs-2.5 \
                --output-dir "$cs_dir" \
                "$idx" \
                --config "$config"; then
                continue
            fi

            dense_cs_dir="$PROJECT_ROOT/checkpoints/cs/${DENSE_CS_DIR_MAP[$model]}"
            dense_cs_mdl="$dense_cs_dir/model_to_cs-2.5.mdl"
            dense_cs_cfg="$dense_cs_dir/config_to_cs-2.5.yaml"
            if [[ "$ALLOW_DENSE_FALLBACK" == "1" && -f "$dense_cs_mdl" ]]; then
                echo "    WARN: conversion failed for $model s${sp}; using dense CS fallback."
                ln -sfn "$dense_cs_mdl" "$cs_dir/model_to_cs-2.5.mdl"
                if [[ -f "$dense_cs_cfg" ]]; then
                    ln -sfn "$dense_cs_cfg" "$cs_dir/config_to_cs-2.5.yaml"
                fi
                continue
            fi

            if [[ "$ALLOW_DENSE_FALLBACK" != "1" ]]; then
                echo "ERROR: conversion failed for $model s${sp} and ALLOW_DENSE_FALLBACK=$ALLOW_DENSE_FALLBACK"
            else
                echo "ERROR: conversion failed and no dense CS fallback found at $dense_cs_mdl"
            fi
            exit 1
        done
    done
}


# =================================================================
#  Phase 3 – Generate YAML configs
# =================================================================
phase_configs() {
    echo ""
    echo ">>> Phase 3: Generating YAML configs ..."
    python scripts/cs3/gen_sparse_yaml.py \
        --models $(to_spaces "$MODELS") \
        --tasks $(to_spaces "$TASKS") \
        --sparsities $(to_spaces "$SPARSITIES") \
        --modes $(to_spaces "$MODES") \
        --max_steps "$MAX_STEPS"
}


# =================================================================
#  Phase 4 – Training
# =================================================================
run_train_subset() {
    local model_csv="$1"
    local force_rerun="$2"
    local rerun_flag=""

    [[ -n "$model_csv" ]] || return 0

    if [[ "$force_rerun" == "1" || "$SKIP_SUCCESSFUL" != "1" ]]; then
        rerun_flag="--rerun_successful"
    fi

    python scripts/cs3/run_masked_lora_cs.py \
        --models "$model_csv" \
        --tasks "$TASKS" \
        --sparsities "$SPARSITIES" \
        --modes "$MODES" \
        --num_csx "$NUM_CSX" \
        --max_steps "$MAX_STEPS" \
        --max_retries "$MAX_RETRIES" \
        --retry_wait_sec "$RETRY_WAIT_SEC" \
        --results_root "$RESULTS_ROOT" \
        $rerun_flag \
        --run_train
}

phase_train() {
    echo ""
    echo ">>> Phase 4: Training on CS-3 ..."
    local incomplete_models=()
    local all_runs_models=()
    local model

    for model in $(to_spaces "$MODELS"); do
        if [[ "$model" == "mixtral" ]]; then
            all_runs_models+=("$model")
        else
            incomplete_models+=("$model")
        fi
    done

    local incomplete_csv=""
    local all_runs_csv=""
    if [[ ${#incomplete_models[@]} -gt 0 ]]; then
        incomplete_csv="$(join_by_comma "${incomplete_models[@]}")"
    fi
    if [[ ${#all_runs_models[@]} -gt 0 ]]; then
        all_runs_csv="$(join_by_comma "${all_runs_models[@]}")"
    fi

    if [[ -n "$incomplete_csv" ]]; then
        echo "    Policy: incomplete-only models -> $incomplete_csv"
        run_train_subset "$incomplete_csv" 0
    fi

    if [[ -n "$all_runs_csv" ]]; then
        echo "    Policy: rerun-all models       -> $all_runs_csv"
        run_train_subset "$all_runs_csv" 1
    fi
}


# =================================================================
#  Phase 8 – Cleanup
# =================================================================
phase_cleanup_successful_models() {
    echo ""
    echo ">>> Phase 8: Cleaning finetuned model dirs ..."
    if [[ "$DELETE_FINETUNED_AFTER_EVAL" != "1" && "$DELETE_TRAIN_ONLY_MODELS" != "1" ]]; then
        echo "    Skipped (DELETE_FINETUNED_AFTER_EVAL=$DELETE_FINETUNED_AFTER_EVAL, DELETE_TRAIN_ONLY_MODELS=$DELETE_TRAIN_ONLY_MODELS)."
        return 0
    fi

    local deleted=0
    local run_dir
    local run_name
    local run_model
    local model_dir
    shopt -s nullglob
    for run_dir in "$RESULTS_ROOT"/cs3_*; do
        [[ -d "$run_dir" ]] || continue
        [[ -f "$run_dir/train.log" ]] || continue

        run_name="$(basename "$run_dir")"
        run_model="${run_name#cs3_}"
        run_model="${run_model%%_*}"
        if ! csv_has_item "$MODELS" "$run_model"; then
            continue
        fi

        is_eval_success=0
        is_train_only=0
        grep -q "Evaluation completed successfully!" "$run_dir/train.log" && is_eval_success=1
        if [[ "$is_eval_success" == "0" ]] && grep -q "Training completed successfully!" "$run_dir/train.log"; then
            is_train_only=1
        fi

        if [[ "$is_eval_success" == "1" && "$DELETE_FINETUNED_AFTER_EVAL" != "1" ]]; then
            continue
        fi
        if [[ "$is_train_only" == "1" && "$DELETE_TRAIN_ONLY_MODELS" != "1" ]]; then
            continue
        fi
        if [[ "$is_eval_success" != "1" && "$is_train_only" != "1" ]]; then
            continue
        fi

        model_dir="$run_dir/model_dir"
        if [[ -d "$model_dir" ]]; then
            case "$model_dir" in
                "$RESULTS_ROOT"/cs3_*/model_dir)
                    rm -rf "$model_dir"
                    deleted=$((deleted + 1))
                    echo "    Deleted: $model_dir"
                    ;;
                *)
                    echo "    SKIP unsafe path: $model_dir"
                    ;;
            esac
        fi
    done
    shopt -u nullglob

    echo "    Cleanup complete. Removed model_dir from $deleted run(s)."
}


# =================================================================
#  Phase 5 – Validate sparsity
# =================================================================
phase_post_validate() {
    echo ""
    echo ">>> Phase 5: Post-training sparsity validation ..."
    for run_dir in "$RESULTS_ROOT"/cs3_*; do
        [[ -d "$run_dir" ]] || continue
        [[ -f "$run_dir/sparse_config.json" ]] || continue

        cfg=$(cat "$run_dir/sparse_config.json")
        model=$(echo "$cfg" | python3 -c "import sys,json; print(json.load(sys.stdin)['model'])")
        sp=$(echo "$cfg" | python3 -c "import sys,json; print(json.load(sys.stdin)['sparsity'])")
        mode=$(echo "$cfg" | python3 -c "import sys,json; print(json.load(sys.stdin)['mode'])")

        mask_path="$PROJECT_ROOT/masks/$model/masks_sparsity${sp}_unstructured.pt"
        if [[ ! -f "$mask_path" ]]; then
            echo "    SKIP (no mask): $run_dir"
            continue
        fi

        if [[ -f "$run_dir/sparsity_validation_report.json" ]]; then
            echo "    [cached] $run_dir"
            continue
        fi

        echo "    Validating: $(basename "$run_dir")"
        python scripts/cs3/merge_and_validate.py \
            --run_dir "$run_dir" \
            --mask_path "$mask_path" \
            --mode "$mode" || true
    done
}


# =================================================================
#  Phase 6 – Throughput benchmarking
# =================================================================
phase_bench() {
    echo ""
    echo ">>> Phase 6: Throughput benchmarking (validate from checkpoint_200) ..."
    python scripts/cs3/run_masked_lora_cs.py \
        --models "$MODELS" \
        --tasks "$TASKS" \
        --sparsities "$SPARSITIES" \
        --modes "$MODES" \
        --num_csx "$NUM_CSX" \
        --bench_eval_steps "$BENCH_EVAL_STEPS" \
        --results_root "$RESULTS_ROOT" \
        --run_bench \
        --bench_batch_sizes "$BENCH_BATCH_SIZES"
}


# =================================================================
#  Phase 7 – Analysis & plots
# =================================================================
phase_analyze() {
    echo ""
    echo ">>> Phase 7: Generating plots and tables ..."
    python scripts/cs3/plot_sparse_results.py \
        --results_dir "$RESULTS_ROOT" \
        --output_dir "$RESULTS_ROOT/analysis"
}


# =================================================================
#  Dispatch
# =================================================================
case "$MODE" in
    prepare)
        ensure_required_artifacts
        phase_validate
        phase_mask
        phase_convert
        phase_configs
        ;;
    train)
        ensure_required_artifacts
        if ! phase_train; then
            train_rc=$?
            echo "WARN: training failed (rc=$train_rc); running cleanup before exit."
            phase_cleanup_successful_models
            exit "$train_rc"
        fi
        phase_cleanup_successful_models
        ;;
    validate)
        phase_validate
        phase_post_validate
        ;;
    bench)
        phase_bench
        ;;
    analyze)
        phase_analyze
        ;;
    cleanup)
        phase_cleanup_successful_models
        ;;
    sweep)
        ensure_required_artifacts
        phase_validate
        phase_mask
        phase_convert
        phase_configs
        if ! phase_train; then
            train_rc=$?
            echo "WARN: training failed (rc=$train_rc); running cleanup before exit."
            phase_cleanup_successful_models
            exit "$train_rc"
        fi
        phase_post_validate
        phase_bench
        phase_analyze
        phase_cleanup_successful_models
        ;;
    *)
        echo "Usage: $0 [prepare|train|validate|bench|analyze|cleanup|sweep]"
        exit 1
        ;;
esac

echo ""
echo "================================================================"
echo " DONE.  Outputs in: $RESULTS_ROOT"
echo "================================================================"
