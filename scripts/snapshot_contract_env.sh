set -euo pipefail

CONTRACT_PATH="${1:?usage: snapshot_contract_env.sh <path_to_contract_yaml> <out_dir>}"
OUT_DIR="${2:?usage: snapshot_contract_env.sh <path_to_contract_yaml> <out_dir>}"

mkdir -p "${OUT_DIR}"

# Contract snapshot
cp "${CONTRACT_PATH}" "${OUT_DIR}/contract.yaml"

# Git snapshot (repo)
{
  echo "git_rev: $(git rev-parse HEAD 2>/dev/null || echo NA)"
  echo "git_status:"
  git status --porcelain 2>/dev/null || true
} > "${OUT_DIR}/git_state.txt"

# Basic env snapshot (portable)
{
  echo "date: $(date -Iseconds)"
  echo "hostname: $(hostname)"
  echo "whoami: $(whoami)"
  echo "pwd: $(pwd)"
  echo "python: $(python --version 2>/dev/null || true)"
  echo ""
  echo "pip_freeze:"
  python -m pip freeze 2>/dev/null || true
} > "${OUT_DIR}/env.txt"
