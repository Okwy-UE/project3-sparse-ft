#!/usr/bin/env bash
#
# CS-3 sparse LoRA launcher (thin wrapper around the full sweep).
#
# Usage:
#   bash scripts/cs3/run_sparse_experiments.sh validate
#   bash scripts/cs3/run_sparse_experiments.sh train
#   bash scripts/cs3/run_sparse_experiments.sh bench
#   bash scripts/cs3/run_sparse_experiments.sh sweep
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/run_full_sparse_sweep.sh" "${1:-sweep}"
