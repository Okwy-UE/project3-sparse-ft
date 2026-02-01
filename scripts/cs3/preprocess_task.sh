set -euo pipefail

TAG="${1:?tag (e.g., mistral-boolq-train)}"
INPUT="${2:?input dir OR jsonl file}"
OUT_DIR="${3:?output dir}"
TOKENIZER="${4:?hf tokenizer id}"
MSL="${5:-2048}"

# Template path (must exist)
TEMPLATE="configs/cerebras/preprocess/template_sft_prompt_completion.yaml"
# If your repo uses a different template path, update TEMPLATE above accordingly.

GEN_DIR="configs/cerebras/preprocess/generated"
mkdir -p "$GEN_DIR"
CFG="${GEN_DIR}/${TAG}.yaml"

TMPDIR=""
INPUT_DIR="$INPUT"

if [ -f "$INPUT" ]; then
  TMPDIR="$(mktemp -d /tmp/csdp_${TAG}.XXXX)"
  ln -sf "$(realpath "$INPUT")" "${TMPDIR}/data.jsonl"
  INPUT_DIR="$TMPDIR"
fi

mkdir -p "$OUT_DIR"

sed \
  -e "s|__INPUT_DIR__|${INPUT_DIR}|g" \
  -e "s|__OUTPUT_DIR__|${OUT_DIR}|g" \
  -e "s|__TOKENIZER__|${TOKENIZER}|g" \
  -e "s|__MSL__|${MSL}|g" \
  "$TEMPLATE" > "$CFG"

echo "[INFO] Using config: $CFG"
cszoo data_preprocess run --config "$CFG"

if [ -n "$TMPDIR" ]; then
  rm -rf "$TMPDIR"
fi

echo "[OK] Preprocessed -> ${OUT_DIR}"
