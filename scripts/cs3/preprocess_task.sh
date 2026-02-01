set -euo pipefail

TASK="${1:?boolq|hellaswag|gsm8k}"
JSONL_DIR="${2:?e.g., $HOME/project3-sparse-ft/data/sft/boolq}"
OUT_DIR="${3:?e.g., $HOME/project3-sparse-ft/data/cs_hdf5/boolq}"
TOKENIZER="${4:?e.g., meta-llama/Meta-Llama-3-8B}"
MSL="${5:-2048}"

TEMPLATE="configs/cerebras/data_preprocess_sft_template.yaml"
CFG_TMP="$(mktemp /tmp/data_preprocess_${TASK}.XXXX.yaml)"

sed \
  -e "s|__JSONL_DIR__|${JSONL_DIR}|g" \
  -e "s|__OUTPUT_DIR__|${OUT_DIR}|g" \
  -e "s|__TOKENIZER__|${TOKENIZER}|g" \
  -e "s|__MSL__|${MSL}|g" \
  "$TEMPLATE" > "$CFG_TMP"

echo "[INFO] Using config: $CFG_TMP"
cszoo data_preprocess run --config "$CFG_TMP"

echo "[OK] Preprocessed: $TASK -> $OUT_DIR"
