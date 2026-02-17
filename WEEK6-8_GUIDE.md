# Weeks 6-8: Sparse Fine-Tuning Pipeline

This guide covers the implementation and execution of the sparse fine-tuning pipeline for Weeks 6-8 of the project.

## Overview

### Week 6: Pruning Pipeline Implementation
- Compute offline masks with task-aware importance
- Validate mask correctness
- Generate per-layer sparsity histograms

### Week 7: Sparse LoRA Implementation
- Implement baseline dense LoRA adapters
- Implement masked LoRA: (B@A) ⊙ M
- Validate sparsity preservation after merge
- Run experiments across all models and tasks

### Week 8: Full Sparse Sweep on Cerebras
- Run sparse-to-dense and sparse-to-sparse experiments
- Test LLaMA and Mistral models
- Collect throughput vs batch size metrics
- Generate comprehensive results datasets

## Implementation Structure

```
project3-sparse-ft/
├── src/
│   ├── pruning/
│   │   ├── __init__.py
│   │   ├── importance.py         # Importance scoring (magnitude, gradient, Taylor)
│   │   ├── mask_generator.py     # Mask generation (unstructured, structured, random)
│   │   ├── mask_ops.py          # Mask I/O, validation, statistics
│   │   └── sparse_lora.py       # Sparse LoRA implementation
│   ├── models/
│   │   ├── __init__.py
│   │   └── sparse_model_wrapper.py  # Model wrapper for sparse training
│   └── utils/
│       └── cerebras_sparse_callback.py  # Cerebras integration
├── scripts/cs3/
│   ├── compute_masks.py         # Generate pruning masks
│   ├── validate_sparsity.py     # Validate sparsity preservation
│   ├── analyze_results.py       # Aggregate and analyze results
│   └── run_sparse_experiments.sh # Main experiment runner
└── masks/                       # Generated masks (not in git)
    ├── llama/
    ├── mistral/
    └── mixtral/
```

## Quick Start

### 1. Environment Setup

```bash
# On Cerebras CS-3
ssh ebereuwadoka@cerebras.alcf.anl.gov
ssh cer-usn-01

# Activate environment
source ~/R_2.6.0/venv_cerebras_pt/bin/activate

# Navigate to project
cd ~/project3-sparse-ft

# Pull latest code
git pull
```

### 2. Week 6: Compute Masks

```bash
# Compute masks for all models and methods
bash scripts/cs3/run_sparse_experiments.sh masks
```

This will:
- Generate masks for LLaMA, Mistral at sparsities: 0%, 10%, ..., 90%
- Use three methods: unstructured, structured, random
- Save masks to `./masks/{model}/masks_sparsity{XX}_{method}.pt`
- Generate validation reports and histograms

**Manual mask computation:**

```bash
# Single model, single method
python scripts/cs3/compute_masks.py \
    --model_path ~/project3-sparse-ft/checkpoints/cs/mistral_7b/model_to_cs-2.5.mdl \
    --output_dir ./masks/mistral \
    --sparsity 0.5 0.7 0.9 \
    --method unstructured \
    --importance magnitude \
    --format pt \
    --plot
```

### 3. Week 7: Train Sparse LoRA

```bash
# Single training run (for testing)
bash scripts/cs3/run_sparse_experiments.sh train
```

**Manual training:**

```bash
# The training script is integrated into run_sparse_experiments.sh
# You can customize specific runs by editing the script
```

### 4. Week 8: Full Sparse Sweep

```bash
# Run full experimental sweep
bash scripts/cs3/run_sparse_experiments.sh sweep
```

This will:
- Train models at all sparsity levels (0-90%)
- Test all pruning methods (unstructured, structured, random)
- Run both sparse-to-dense and sparse-to-sparse modes
- Validate sparsity preservation for each run

### 5. Analyze Results

```bash
python scripts/cs3/analyze_results.py \
    --results_dir ./results/runs \
    --output_dir ./results/analysis \
    --plot
```

This generates:
- `summary.csv`: All runs with metadata
- `grouped_summary.csv`: Aggregated by model/task/method/sparsity
- `sparsity_vs_performance.csv`: Performance metrics per sparsity level
- Plots: sparsity vs loss, sparsity vs throughput, comparison heatmaps

## Mask Generation Details

### Importance Metrics

**Magnitude (default):**
- Score: |W|
- Fast, no data required
- Works well for pretrained models

**Gradient:**
- Score: |∇W|
- Requires forward/backward passes
- More task-aware

**Taylor (first-order):**
- Score: |W · ∇W|
- Approximates loss change
- Best task-aware metric

### Pruning Methods

**Unstructured:**
- Element-wise pruning
- Highest compression potential
- Requires sparse kernels for speedup

**Structured:**
- Channel/filter pruning
- Hardware-friendly
- Lower compression ratio

**Random:**
- Baseline for comparison
- No importance computation

### Sparsity Levels

Testing 0%, 10%, 20%, ..., 90% sparsity to create a comprehensive performance curve.

## Sparse LoRA Details

### Architecture

```
Standard LoRA:   y = W·x + (B @ A)·x
Sparse LoRA:     y = (W ⊙ M)·x + (B @ A)·x
Merge:          W' = (W + B @ A) ⊙ M
```

Where:
- W: base weights (frozen)
- M: binary mask
- A, B: trainable LoRA matrices
- ⊙: element-wise multiplication

### Training Modes

**Sparse-to-Dense:**
- Start with sparse base weights
- Train dense LoRA adapters
- Merge: re-apply mask after merging
- Use case: maximum expressiveness during fine-tuning

**Sparse-to-Sparse:**
- Start with sparse base weights
- Train dense LoRA adapters
- Enforce sparsity after each step
- Merge: re-apply mask
- Use case: maintain sparsity throughout training

## Validation

### Sparsity Preservation Check

After training and merge:

```bash
python scripts/cs3/validate_sparsity.py \
    --checkpoint ./results/runs/{run_id}/model_dir \
    --mask_path ./masks/mistral/masks_sparsity50_unstructured.pt \
    --output ./results/runs/{run_id}/validation_report.json
```

Checks:
1. Where M = 0, W must be 0 (within tolerance)
2. Overall sparsity matches expected
3. Per-layer sparsity statistics

### Expected Outcomes

**Week 6 Deliverable:**
- Masks for all models at all sparsity levels
- Validation reports showing correct sparsity
- Histograms showing per-layer sparsity distribution

**Week 7 Deliverable:**
- Correctness memo proving sparsity preservation
- LoRA baseline results (dense adapters)
- Sparse LoRA results with validation
- No significant regression beyond tolerance

**Week 8 Deliverable:**
- Complete results dataset for Cerebras runs
- Performance curves: accuracy vs sparsity
- Throughput analysis: samples/sec vs sparsity
- Comparison plots vs Phoenix baseline

## Phoenix Compatibility

### Tasks

Using Phoenix-compatible tasks:
- **BoolQ**: Boolean question answering
- **HellaSwag**: Commonsense NLI
- **GSM8K**: Grade school math

### Models

- **LLaMA-7B**
- **Mistral-7B**
- **Mixtral-8x7B** (optional, for MOE comparison)

### Metrics

Tracking same metrics as Phoenix:
- Final loss
- Accuracy/F1 (task-dependent)
- Throughput (samples/sec)
- Training time
- Memory usage

## Troubleshooting

### Mask not found error

```bash
# Re-generate masks
python scripts/cs3/compute_masks.py --model_path ... --output_dir ./masks/...
```

### Sparsity validation fails

Check:
1. Mask was loaded correctly
2. LoRA merge applied mask
3. Checkpoint saved after mask application

Debug:
```python
# In validation script, add:
print(f"Weight[mask==0]: min={weight[mask==0].min()}, max={weight[mask==0].max()}")
```

### Training OOM on Cerebras

Reduce batch size in config:
```yaml
trainer:
  fit:
    train_dataloader:
      batch_size: 128  # Reduce from 256
```

### Config not found

Ensure configs exist:
```bash
find ./configs -name "*.yaml" | grep -i mistral | grep -i gsm8k
```

Generate if missing:
```bash
python scripts/cs3/gen_train_config.py --model mistral --task gsm8k
```

## Performance Tips

### Parallel Mask Generation

```bash
# Generate masks in parallel for different models
for model in llama mistral; do
    python scripts/cs3/compute_masks.py \
        --model_path ./checkpoints/cs/${model}_7b/model_to_cs-2.5.mdl \
        --output_dir ./masks/$model \
        --sparsity 0.5 0.7 0.9 \
        --method unstructured &
done
wait
```

### Cerebras Compilation Cache

Set `COMPILE_ONCE=1` to reuse compilations:
```bash
export COMPILE_ONCE=1
bash scripts/cs3/run_sparse_experiments.sh sweep
```

### Batch Size Tuning

For throughput analysis (Week 8), test multiple batch sizes:
```yaml
batch_sizes: [32, 64, 128, 256, 512]
```

## Next Steps

After completing Weeks 6-8:

1. **Analysis**: Compare against Phoenix baseline
2. **Visualization**: Create plots for paper/presentation
3. **Documentation**: Write up findings in `EXPERIMENTS.md`
4. **Optimization**: Identify best sparsity-performance tradeoffs

## References

- [Cerebras ModelZoo](https://github.com/Cerebras/modelzoo)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Magnitude Pruning](https://arxiv.org/abs/1506.02626)
- [Phoenix Paper](https://dl.acm.org/doi/10.1145/3731599.3767395)
