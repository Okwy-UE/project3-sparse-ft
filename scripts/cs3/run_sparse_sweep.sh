#!/usr/bin/env bash
set -euo pipefail

export HTTPS_PROXY="http://proxy.alcf.anl.gov:3128"
export https_proxy="$HTTPS_PROXY"
source "$HOME/R_2.6.0/venv_cerebras_pt/bin/activate"

cd "$HOME/project3-sparse-ft"
git pull || true

: "${NUM_CSX:=2}"
: "${RESULTS_ROOT:=$PWD/results/runs}"

MODELS=("llama3" "mistral")
TASKS=("boolq" "hellaswag" "gsm8k")

# "all percentages" => define a grid you can defend.
# If you truly mean every integer percent, set e.g. seq 10 1 90 (but that's huge).
SPARSITIES=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
TYPES=("unstructured_task" "random" "structured_nm")

# For N:M (structured)
NM_N=2
NM_M=4

find_cfg() {
  local model="$1"
  local task="$2"
  local hits=""
  hits="$(find "$PWD/configs" -type f \( -name "*.yaml" -o -name "*.yml" \) \
          | grep -iE "$model" | grep -iE "$task" | head -n1 || true)"
  [[ -z "$hits" ]] && return 1
  echo "$hits"
}

run_one() {
  local model="$1"
  local task="$2"
  local cfg="$3"
  local mask_type="$4"
  local sparsity="$5"

  local run_id="cs3_${model}_${task}_${mask_type}_p${sparsity}_$(date +%Y%m%d_%H%M%S)"
  local out_dir="$RESULTS_ROOT/$run_id"
  mkdir -p "$out_dir"

  cp "$cfg" "$out_dir/train.yaml"

  # Patch YAML ckpt_path to masked CS checkpoint (you will create it)
  local masked_ckpt="$HOME/project3-sparse-ft/checkpoints/cs_masked/${model}/${task}/${mask_type}/p${sparsity}/model.mdl"
  python "$PWD/scripts/cs3/patch_yaml_for_sparse.py" \
    --yaml "$out_dir/train.yaml" \
    --ckpt_path "$masked_ckpt" \
    --mask_type "$mask_type" \
    --sparsity "$sparsity" \
    --notes "sparse sweep"

  echo "[INFO] RUN_ID=$run_id"
  echo "[INFO] CFG=$out_dir/train.yaml"
  echo "[INFO] MASKED_CKPT=$masked_ckpt"

  cszoo fit "$out_dir/train.yaml" \
    --num_csx="$NUM_CSX" \
    --job_labels "name=${run_id}" \
    --model_dir "$out_dir/model_dir" \
    --mount_dirs "$PWD" \
    --python_paths "$PWD/src" \
    |& tee "$out_dir/train.log"
}

for model in "${MODELS[@]}"; do
  for task in "${TASKS[@]}"; do
    cfg="$(find_cfg "$model" "$task")" || { echo "SKIP: no cfg for $model $task"; continue; }
    for mask_type in "${TYPES[@]}"; do
      for sp in "${SPARSITIES[@]}"; do
        run_one "$model" "$task" "$cfg" "$mask_type" "$sp"
      done
    done
  done
done

echo "[DONE] sparse sweep"
