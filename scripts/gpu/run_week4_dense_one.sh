#!/usr/bin/env bash
# scripts/gpu/run_week4_dense_one.sh
set -euo pipefail

# Usage:
#   MODEL_ID="meta-llama/Llama-3.1-8B" TASK="gsm8k" bash scripts/gpu/run_week4_dense_one.sh
#
# Required env:
#   HF_TOKEN (for gated models like Llama 3.1)
#
# Optional env:
#   BATCH_POINTS="8 16 32"
#   TRAIN_BSZ=16
#   SEQ_LEN=2048
#   MAX_STEPS=200
#   LR=2e-5
#   WARMUP=10
#   WD=0.0
#   GRAD_ACCUM=1
#   ATTN_IMPL=flash_attention_2

: "${MODEL_ID:?Set MODEL_ID}"
: "${TASK:?Set TASK}"

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export TOKENIZERS_PARALLELISM=false

SEQ_LEN="${SEQ_LEN:-2048}"
MAX_STEPS="${MAX_STEPS:-200}"
LR="${LR:-2e-5}"
WARMUP="${WARMUP:-10}"
WD="${WD:-0.0}"
GRAD_ACCUM="${GRAD_ACCUM:-1}"
ATTN_IMPL="${ATTN_IMPL:-}"
BATCH_POINTS=(${BATCH_POINTS:-"8 16 32"})
TRAIN_BSZ="${TRAIN_BSZ:-${BATCH_POINTS[-1]}}"

timestamp() { date +"%Y%m%d_%H%M%S"; }
slugify() { echo "$1" | tr '/: ' '___' | tr -cd '[:alnum:]_-' ; }

get_visible_gpu_count() {
  python - <<'PY'
import os, torch
cvd = os.environ.get("CUDA_VISIBLE_DEVICES","")
if cvd.strip():
    print(len([x for x in cvd.split(",") if x.strip()!=""]))
else:
    print(torch.cuda.device_count())
PY
}

NPROC="$(get_visible_gpu_count)"
if [[ "${NPROC}" -le 0 ]]; then
  echo "ERROR: no GPUs visible"
  exit 1
fi

MODEL_TAG="$(slugify "${MODEL_ID}")"
GROUP_ID="w4_${MODEL_TAG}_${TASK}_$(timestamp)_$RANDOM"

mk_run_dir() {
  local kind="$1"
  local dir="results/runs/${GROUP_ID}/${kind}"
  mkdir -p "${dir}"
  echo "${dir}"
}

maybe_flag() {
  local name="$1"; local val="$2"
  if [[ -n "${val}" ]]; then echo "--${name} ${val}"; fi
}

echo "[week4] MODEL_ID=${MODEL_ID}"
echo "[week4] TASK=${TASK}"
echo "[week4] GROUP_ID=${GROUP_ID}"
echo "[week4] visible_gpus=${NPROC}"
echo "[week4] seq_len=${SEQ_LEN} batch_points=${BATCH_POINTS[*]} train_bsz=${TRAIN_BSZ}"

# ---------------------------
# (A) Throughput scaling
# ---------------------------
for bsz in "${BATCH_POINTS[@]}"; do
  run_dir="$(mk_run_dir "bench_bsz${bsz}")"
  echo "[bench] bsz=${bsz} -> ${run_dir}"

  torchrun --nproc_per_node="${NPROC}" -m src.train.gpu_dense_sft \
    --run_dir "${run_dir}" \
    --group_id "${GROUP_ID}" \
    --model_name "${MODEL_ID}" \
    --task "${TASK}" \
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

  python -m src.utils.run_logging register --run_dir "${run_dir}"
done

# ---------------------------
# (B) Dense training run
# ---------------------------
run_dir="$(mk_run_dir "train")"
echo "[train] bsz=${TRAIN_BSZ} steps=${MAX_STEPS} -> ${run_dir}"

DS_FLAG=""
if [[ "${MODEL_ID}" == *"Mixtral-8x7B"* ]]; then
  DS_FLAG="--deepspeed configs/gpu/deepspeed_zero3_bf16.json"
fi

torchrun --nproc_per_node="${NPROC}" -m src.train.gpu_dense_sft \
  --run_dir "${run_dir}" \
  --group_id "${GROUP_ID}" \
  --model_name "${MODEL_ID}" \
  --task "${TASK}" \
  --seq_len "${SEQ_LEN}" \
  --per_device_train_batch_size "${TRAIN_BSZ}" \
  --gradient_accumulation_steps "${GRAD_ACCUM}" \
  --max_steps "${MAX_STEPS}" \
  --learning_rate "${LR}" \
  --warmup_steps "${WARMUP}" \
  --weight_decay "${WD}" \
  --bf16 \
  --gradient_checkpointing \
  $(maybe_flag "attn_implementation" "${ATTN_IMPL}") \
  ${DS_FLAG}

python -m src.utils.run_logging register --run_dir "${run_dir}"

# ---------------------------
# (C) Eval (identical harness)
# ---------------------------
HF_OUT_DIR="$(python - <<PY
import json
p="${run_dir}/checkpoint.json"
print(json.load(open(p))["hf_out_dir"])
PY
)"
echo "[eval] hf_out=${HF_OUT_DIR}"

eval_dir="$(mk_run_dir "eval")"
python -m src.eval.eval_phoenix_tasks \
  --model_or_path "${HF_OUT_DIR}" \
  --task "${TASK}" \
  --out_json "${eval_dir}/eval_metrics.json" \
  --bf16 \
  --max_length "${SEQ_LEN}" \
  $( [[ -n "${ATTN_IMPL}" ]] && echo "--attn_implementation ${ATTN_IMPL}" )

# Tie eval to the same group_id by copying meta
cp "${run_dir}/run_meta.json" "${eval_dir}/run_meta.json" || true
python -m src.utils.run_logging snapshot --run_dir "${eval_dir}"
python -m src.utils.run_logging register --run_dir "${eval_dir}"

echo "DONE: ${GROUP_ID}"
