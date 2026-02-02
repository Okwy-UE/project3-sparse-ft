#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/gpu/run_week4_dense_h100.sh
#
# Assumes:
#   - you are on a dgxh-* node with 8 GPUs
#   - your env has: torch, transformers, datasets, deepspeed (for Mixtral), accelerate
#   - HF_TOKEN set for gated models (Llama 3.1)

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

SEQ_LEN=2048
BATCH_POINTS=(1 2 4)   # >=3 points as requested
MAX_STEPS=200          # set to match your baseline budget
LR=2e-5
WARMUP=10
WD=0.0
GRAD_ACCUM=1
BF16=1
ATTN_IMPL=""           # optionally: flash_attention_2

# Model IDs (HF)
LLAMA="meta-llama/Llama-3.1-8B"
MISTRAL="mistralai/Mistral-7B-v0.1"
MIXTRAL="mistralai/Mixtral-8x7B-v0.1"

MODELS=("$LLAMA" "$MISTRAL" "$MIXTRAL")
TASKS=("boolq" "hellaswag" "gsm8k")

timestamp() { date +"%Y%m%d_%H%M%S"; }
mk_run_dir() {
  local tag="$1"
  local rid="${tag}_$(timestamp)_$RANDOM"
  local dir="results/runs/${rid}"
  mkdir -p "${dir}"
  echo "${dir}"
}

maybe_flag() {
  local name="$1"; local val="$2"
  if [[ -n "${val}" ]]; then echo "--${name} ${val}"; fi
}

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "===================================================="
    echo "W4: model=${model} task=${task}"
    echo "===================================================="

    # (A) Throughput scaling (one measured step per batch size)
    for bsz in "${BATCH_POINTS[@]}"; do
      run_dir="$(mk_run_dir "w4_bench")"
      echo "[bench] bsz=${bsz} -> ${run_dir}"

      torchrun --nproc_per_node=8 -m src.train.gpu_dense_sft \
        --run_dir "${run_dir}" \
        --model_name "${model}" \
        --task "${task}" \
        --seq_len "${SEQ_LEN}" \
        --per_device_train_batch_size "${bsz}" \
        --gradient_accumulation_steps 1 \
        --max_steps 1 \
        --learning_rate "${LR}" \
        --warmup_steps "${WARMUP}" \
        --weight_decay "${WD}" \
        --bf16 \
        $(maybe_flag "attn_implementation" "${ATTN_IMPL}") \
        --bench_only \
        --bench_warmup_steps 1 \
        --bench_measured_steps 1

      # simple tag file for downstream aggregation
      echo "${model}" > "${run_dir}/model_id.txt"
      echo "${task}" > "${run_dir}/task_id.txt"
    done

    # (B) Dense train run (choose a “best” batch size point; you can change)
    train_bsz="${BATCH_POINTS[-1]}"
    run_dir="$(mk_run_dir "w4_train")"
    echo "[train] bsz=${train_bsz} steps=${MAX_STEPS} -> ${run_dir}"

    ds_flag=""
    # Mixtral generally needs Zero-3 for dense fine-tune on 8 GPUs
    if [[ "${model}" == "${MIXTRAL}" ]]; then
      ds_flag="--deepspeed configs/gpu/deepspeed_zero3_bf16.json"
    fi

    torchrun --nproc_per_node=8 -m src.train.gpu_dense_sft \
      --run_dir "${run_dir}" \
      --model_name "${model}" \
      --task "${task}" \
      --seq_len "${SEQ_LEN}" \
      --per_device_train_batch_size "${train_bsz}" \
      --gradient_accumulation_steps "${GRAD_ACCUM}" \
      --max_steps "${MAX_STEPS}" \
      --learning_rate "${LR}" \
      --warmup_steps "${WARMUP}" \
      --weight_decay "${WD}" \
      --bf16 \
      --gradient_checkpointing \
      $(maybe_flag "attn_implementation" "${ATTN_IMPL}") \
      ${ds_flag}

    echo "${model}" > "${run_dir}/model_id.txt"
    echo "${task}" > "${run_dir}/task_id.txt"

    # (C) Identical eval harness (task accuracy)
    HF_OUT_DIR="$(python -c "import json; print(json.load(open('${run_dir}/checkpoint.json'))['hf_out_dir'])")"
    echo "[eval] hf_out=${HF_OUT_DIR}"
    python -m src.eval.eval_phoenix_tasks \
      --model_or_path "${HF_OUT_DIR}" \
      --task "${task}" \
      --out_json "${run_dir}/eval_metrics.json" \
      --bf16 \
      $( [[ -n "${ATTN_IMPL}" ]] && echo "--attn_implementation ${ATTN_IMPL}" )

  done
done

echo "DONE Week4 runs. Next: aggregate tables."
