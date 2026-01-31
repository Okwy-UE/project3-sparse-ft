
#!/usr/bin/env bash

set -euo pipefail



REGISTRY="${REGISTRY:-$HOME/project3/results/runs/run_registry.csv}"

mkdir -p "$(dirname "$REGISTRY")"



if [ ! -f "$REGISTRY" ]; then

  echo "run_id,date,hardware,system,model,task,method,sparsity,seed,max_seq_len,global_batch,lr,notes,artifact_path" > "$REGISTRY"

fi



run_id="$1"; hardware="$2"; system="$3"; model="$4"; task="$5"; method="$6"; sparsity="$7"

seed="${8:-1}"; max_seq_len="${9:-2048}"; global_batch="${10:-NA}"; lr="${11:-NA}"

notes="${12:-}"; artifact_path="${13:-}"



echo "${run_id},$(date -Iseconds),${hardware},${system},${model},${task},${method},${sparsity},${seed},${max_seq_len},${global_batch},${lr},\"${notes}\",${artifact_path}" >> "$REGISTRY"

