#!/usr/bin/env bash
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TRANSFORMERS_CACHE="$HF_HOME"

MODEL="$1"   # meta-llama/Meta-Llama-3-8B
TASK="$2"    # gsm8k or boolq
OUT="$3"

python scripts/gpu/finetune_lora.py \
  --model "$MODEL" \
  --task "$TASK" \
  --out "$OUT" \
  --max_seq_len 2048 \
  --lr 2e-4 \
  --epochs 1 \
  --per_device_train_bs 1 \
  --grad_accum 16 \
  --lora_r 16 \
  --lora_alpha 64 |& tee "$OUT/stdout.log"
