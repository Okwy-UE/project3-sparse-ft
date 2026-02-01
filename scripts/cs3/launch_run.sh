#!/usr/bin/env bash
set -euo pipefail

TRAIN_YAML="${1:?path to generated train yaml}"
MODEL_TAG="${2:?llama3|mistral|mixtral}"
TASK="${3:?boolq|hellaswag|gsm8k}"
SYSTEM="${4:-CS-3}"

RUN_ID="${SYSTEM}_${MODEL_TAG}_${TASK}_$(date +%Y%m%d_%H%M%S)"
ART_DIR="$HOME/project3-sparse-ft/results/runs/$RUN_ID"
MODEL_DIR="$HOME/project3-sparse-ft/results/runs/$RUN_ID/model_dir"

mkdir -p "$ART_DIR"
mkdir -p "$MODEL_DIR"

cp "$TRAIN_YAML" "$ART_DIR/train.yaml"

# Run registry (append-only)
REG="$HOME/project3-sparse-ft/results/runs/run_registry.csv"
if [ ! -f "$REG" ]; then
  mkdir -p "$(dirname "$REG")"
  echo "run_id,date,hardware,system,model,task,method,sparsity,notes,artifact_path" > "$REG"
fi
echo "$RUN_ID,$(date -Iseconds),cerebras,$SYSTEM,$MODEL_TAG,$TASK,dense_lora,0,\"week2 baseline\",$ART_DIR" >> "$REG"

echo "[INFO] RUN_ID=$RUN_ID"
echo "[INFO] MODEL_DIR=$MODEL_DIR"
echo "[INFO] ART_DIR=$ART_DIR"

# Launch
cszoo fit "$TRAIN_YAML" \
  --job_labels "name=${RUN_ID}" \
  --model_dir "$MODEL_DIR" |& tee "$ART_DIR/train.log"

echo "[OK] Completed run: $RUN_ID"
