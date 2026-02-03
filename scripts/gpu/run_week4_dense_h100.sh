#!/usr/bin/env bash
set -euo pipefail

# Usage (examples):
#   sbatch scripts/gpu/run_week4_dense_h100.sh llama3.1-8b boolq
#   sbatch scripts/gpu/run_week4_dense_h100.sh mixtral-8x7b gsm8k

MODEL="${1:?model alias required}"
TASK="${2:?task required}"

#SBATCH --job-name=wk4_dense
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=results/runs/slurm-%j.out

# ---- environment (customize to OSU HPC module stack)
# module purge
# module load cuda/12.1
# module load gcc/...

export PROJECT_HOME="$HOME/project3-sparse-ft"
cd "$PROJECT_HOME"

# Put caches on NVME/scratch if your site provides it:
export HF_HOME="${SCRATCH:-/tmp}/hf"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_HOME"

# Recommended for throughput
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate your env
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate cerebras-project3

python -m pip install -q -U "lm_eval[hf]" accelerate peft datasets transformers

python scripts/gpu/run_week4_dense.py \
  --model "$MODEL" \
  --task "$TASK" \
  --contract configs/contracts/week4_dense_h100.yaml
