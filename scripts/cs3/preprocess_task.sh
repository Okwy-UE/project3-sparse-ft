set -euo pipefail

TAG="${1:?tag like llama3-boolq-train}"
JSONL_PATH="${2:?path to jsonl file, e.g. data/sft/boolq/train.jsonl}"
OUT_DIR="${3:?e.g. data/cs_hdf5/llama3/boolq/train}"
TOKENIZER="${4:?e.g. meta-llama/Meta-Llama-3-8B}"
MSL="${5:-2048}"

TEMPLATE="configs/cerebras/data_preprocess_sft_template.yaml"

# Persist rendered configs for reproducibility
GEN_DIR="configs/cerebras/preprocess/generated"
mkdir -p "$GEN_DIR"
CFG_OUT="$GEN_DIR/${TAG}.yaml"

sed \
  -e "s|__JSONL_PATH__|${JSONL_PATH}|g" \
  -e "s|__OUTPUT_DIR__|${OUT_DIR}|g" \
  -e "s|__TOKENIZER__|${TOKENIZER}|g" \
  -e "s|__MSL__|${MSL}|g" \
  "$TEMPLATE" > "$CFG_OUT"

echo "[INFO] Using config: $CFG_OUT"
cszoo data_preprocess run --config "$CFG_OUT"

echo "[OK] Preprocessed: ${TAG} -> ${OUT_DIR}"
