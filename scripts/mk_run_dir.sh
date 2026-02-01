set -euo pipefail

# usage:
#   mk_run_dir.sh <hardware> <model> <task> <method>
HARDWARE="${1:?}"
MODEL="${2:?}"
TASK="${3:?}"
METHOD="${4:?}"

TS="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${HARDWARE}_${MODEL}_${TASK}_${METHOD}_${TS}"

RUN_DIR="results/runs/${RUN_ID}"
mkdir -p "${RUN_DIR}"

echo "${RUN_ID}"
