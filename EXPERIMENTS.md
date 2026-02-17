# Sparse Fine-Tuning Experiments

**Project**: Sparse-to-Dense Fine-Tuning on Cerebras CS-3  
**Comparison**: Phoenix-compatible baseline  
**Timeline**: Weeks 6-8 (Milestone 3)

---

## Experimental Design

### Objective

Evaluate sparse-to-dense and sparse-to-sparse fine-tuning strategies on Cerebras CS-3, comparing against GPU baselines and Phoenix paper results.

### Research Questions

1. **RQ1**: How does sparsity level affect fine-tuning performance?
2. **RQ2**: Which pruning method (unstructured, structured, random) is most effective?
3. **RQ3**: Does sparse-to-sparse training provide benefits over sparse-to-dense?
4. **RQ4**: How does Cerebras CS-3 compare to GPUs for sparse fine-tuning?

---

## Experimental Matrix

### Models

- **LLaMA-7B**: Decoder-only transformer, 7B parameters
- **Mistral-7B**: Decoder-only with sliding window attention
- **Mixtral-8x7B**: Mixture-of-experts (optional)

### Tasks (Phoenix-compatible)

- **BoolQ**: Boolean question answering (Yes/No)
- **HellaSwag**: Commonsense reasoning (multiple choice)
- **GSM8K**: Grade school math problems

### Sparsity Levels

- **0%**: Dense baseline (no pruning)
- **10%, 20%, ..., 90%**: Progressive sparsity
- Target: comprehensive sparsity curve

### Pruning Methods

1. **Unstructured**: Element-wise pruning based on magnitude
   - Highest compression potential
   - Requires sparse kernels for speedup
   
2. **Structured**: Channel/filter pruning
   - Hardware-friendly (no special kernels needed)
   - Lower compression but maintains dense compute
   
3. **Random**: Random element pruning
   - Baseline to assess importance of importance metrics

### Training Modes

1. **Dense Baseline**: No pruning, standard LoRA
2. **Sparse-to-Dense**: 
   - Initialize with sparse weights
   - Train dense LoRA adapters
   - Merge and re-apply sparsity
3. **Sparse-to-Sparse**:
   - Initialize with sparse weights
   - Train dense LoRA adapters
   - Enforce sparsity after each step
   - Merge and re-apply sparsity

---

## Week-by-Week Plan

### Week 6: Pruning Pipeline Implementation

**Goals:**
- Implement offline mask computation
- Validate mask correctness
- Generate sparsity histograms

**Deliverables:**
- [ ] Mask generation script (`compute_masks.py`)
- [ ] Mask validation script (`validate_sparsity.py`)
- [ ] Masks for all models at all sparsity levels
- [ ] Per-layer sparsity histograms
- [ ] "Pruning + mask spec" document

**Status:** ✅ **COMPLETE**

**Artifacts:**
- Masks: `./masks/{model}/masks_sparsity{XX}_{method}.pt`
- Statistics: `./masks/{model}/stats_sparsity{XX}_{method}.json`
- Histograms: `./masks/{model}/histogram_sparsity{XX}_{method}.png`

---

### Week 7: Sparse LoRA Implementation

**Goals:**
- Implement LoRA baseline (dense adapters)
- Implement masked LoRA: (B@A) ⊙ M
- Validate sparsity preservation after merge
- Run experiments for all models and tasks

**Deliverables:**
- [ ] Sparse LoRA implementation (`sparse_lora.py`)
- [ ] Cerebras integration callback
- [ ] Validation of sparsity preservation
- [ ] Correctness memo: "sparsity preserved + no regression"
- [ ] Results for all model-task pairs at key sparsity points (0%, 50%, 90%)

**Status:** 🔄 **IN PROGRESS**

**Key Metrics:**
- Sparsity preservation: zero violations within tolerance (1e-6)
- Performance: within 5% of dense baseline at low sparsity
- Validation: checksum match on masks

---

### Week 8: Full Sparse Sweep on Cerebras

**Goals:**
- Run comprehensive sparse experiments
- Collect throughput vs batch size data
- Generate performance curves
- Compare against Phoenix results

**Deliverables:**
- [ ] Full experimental sweep (all sparsity levels, all methods)
- [ ] Throughput analysis (samples/sec vs sparsity)
- [ ] Performance curves (accuracy vs sparsity)
- [ ] Preliminary plots and tables
- [ ] Cerebras sparse results dataset

**Status:** ⏳ **PENDING**

**Target Runs:**
- Models: 2 (LLaMA, Mistral)
- Tasks: 3 (BoolQ, HellaSwag, GSM8K)
- Methods: 3 (unstructured, structured, random)
- Sparsity: 10 levels (0%-90%)
- Modes: 2 (sparse-to-dense, sparse-to-sparse)
- **Total**: 2 × 3 × 3 × 10 × 2 = **360 runs**

---

## Experimental Protocol

### 1. Mask Generation (Week 6)

```bash
# For each model and method
python scripts/cs3/compute_masks.py \
    --model_path ./checkpoints/cs/{model}/model_to_cs-2.5.mdl \
    --output_dir ./masks/{model} \
    --sparsity 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 \
    --method {unstructured|structured|random} \
    --importance magnitude \
    --format pt \
    --plot
```

**Validation:**
- Check mask is binary (only 0s and 1s)
- Verify actual sparsity matches target (±1%)
- Inspect per-layer sparsity distribution

### 2. Training (Week 7)

```bash
# For each model, task, sparsity, method, mode
bash scripts/cs3/run_sparse_experiments.sh train
```

**Configuration:**
- Optimizer: AdamW (β1=0.9, β2=0.95, weight_decay=0.1)
- Learning rate: 3e-4 (cosine decay to 3e-5)
- Warmup: 2000 steps
- Max steps: 200 (for smoke testing, increase for full runs)
- Batch size: 256 (tune based on memory)
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05

**Validation:**
- After training, validate sparsity preserved
- Check zero violations < tolerance
- Log sparsity statistics

### 3. Full Sweep (Week 8)

```bash
# Run complete experimental matrix
bash scripts/cs3/run_sparse_experiments.sh sweep
```

**Analysis:**
- Aggregate results across all runs
- Generate performance curves
- Compute throughput statistics
- Create comparison tables

---

## Metrics

### Primary Metrics

1. **Accuracy/F1**: Task-specific performance
   - BoolQ: Accuracy
   - HellaSwag: Accuracy
   - GSM8K: Exact match accuracy

2. **Sparsity**: Fraction of zero weights
   - Global sparsity (across all layers)
   - Per-layer sparsity distribution

3. **Throughput**: Training speed
   - Samples/second
   - Tokens/second

### Secondary Metrics

4. **Training Time**: Wall-clock time to convergence
5. **Memory Usage**: Peak memory during training
6. **Compilation Time**: Cerebras graph compilation time
7. **Loss**: Training and validation loss

---

## Validation Criteria

### Mask Correctness

- ✅ Mask is binary (values ∈ {0, 1})
- ✅ Actual sparsity within 1% of target
- ✅ Per-layer sparsity reasonable (no layer 100% sparse)

### Sparsity Preservation

- ✅ After LoRA merge, weights at masked positions are zero
- ✅ Violations within numerical tolerance (1e-6)
- ✅ Checksum matches original mask

### Performance

- ✅ At 0% sparsity: matches dense baseline (±2%)
- ✅ At 50% sparsity: within 10% of dense baseline
- ✅ At 90% sparsity: degrades gracefully (not random)

---

## Results Structure

```
results/
├── runs/
│   └── cs3_{model}_{task}_{method}_s{sparsity}_{mode}_{sha}_{timestamp}/
│       ├── config.yaml               # Training config
│       ├── sparse_config.json        # Sparse-specific config
│       ├── train.log                 # Training log
│       ├── validation_report.json    # Sparsity validation
│       ├── model_dir/                # Checkpoints
│       └── git_sha.txt               # Git commit
├── analysis/
│   ├── summary.csv                   # All runs
│   ├── grouped_summary.csv           # Aggregated by hyperparams
│   ├── sparsity_vs_performance.csv   # Sparsity curves
│   └── plots/
│       ├── sparsity_vs_loss.png
│       ├── sparsity_vs_throughput.png
│       └── method_comparison_heatmap.png
└── tables/
    ├── main_results.tex              # LaTeX table for paper
    └── phoenix_comparison.tex        # Comparison with Phoenix
```

---

## Baseline Comparisons

### Dense Baseline (Week 4-5)

Prior work established dense LoRA fine-tuning baselines:

- **LLaMA-7B + BoolQ**: Loss=X.XX, Accuracy=XX.X%
- **Mistral-7B + HellaSwag**: Loss=X.XX, Accuracy=XX.X%
- **Mistral-7B + GSM8K**: Loss=X.XX, Accuracy=XX.X%

(See `results/run_registry.csv` for full results)

### Phoenix Comparison

Phoenix paper (ACM '24) reports:
- Models: Similar scale (7B parameters)
- Tasks: BoolQ, HellaSwag, GSM8K
- Sparsity: Up to 90%
- Hardware: NVIDIA GPUs

**Goal**: Match or exceed Phoenix performance on Cerebras CS-3.

---

## Known Issues & Limitations

### Current Limitations

1. **Mask Application**: Currently applied via PyTorch hooks. May need Cerebras-specific implementation for optimal performance.

2. **Sparse Kernels**: CS-3 may not accelerate unstructured sparsity without custom kernels.

3. **Gradient Accumulation**: Sparse-to-sparse mode requires mask application after each gradient step, which may slow training.

4. **Memory**: Very high sparsity (>90%) may cause numerical instability.

### Workarounds

- Use structured pruning for better hardware utilization
- Apply masks only at initialization and after merge (sparse-to-dense)
- Monitor training curves closely at high sparsity

---

## Analysis Plan

### Week 8 Analysis

1. **Aggregate Results**
   ```bash
   python scripts/cs3/analyze_results.py \
       --results_dir ./results/runs \
       --output_dir ./results/analysis \
       --plot
   ```

2. **Generate Plots**
   - Sparsity vs accuracy (all methods)
   - Sparsity vs throughput (Cerebras speedup analysis)
   - Method comparison heatmaps

3. **Statistical Tests**
   - Compare sparse-to-dense vs sparse-to-sparse (paired t-test)
   - Compare methods (ANOVA)
   - Significance testing vs Phoenix baseline

4. **Insights**
   - Identify optimal sparsity-performance tradeoff
   - Determine best pruning method per model/task
   - Quantify Cerebras vs GPU advantages

---

## References

- **Phoenix Paper**: [ACM DL](https://dl.acm.org/doi/10.1145/3731599.3767395)
- **LoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Magnitude Pruning**: [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- **Cerebras ModelZoo**: [GitHub](https://github.com/Cerebras/modelzoo)

---

## Team & Timeline

**Investigator**: Ebere Uwadoka  
**Advisor**: [Your Advisor]  
**Duration**: Weeks 6-8 (Feb 2026)  
**Hardware**: Cerebras CS-3 at ALCF

---

## Changelog

- **2026-02-16**: Created experiment plan for Weeks 6-8
- **2026-02-16**: Implemented pruning pipeline (Week 6)
- **2026-02-16**: Implemented Sparse LoRA (Week 7)

---

**Next Update**: End of Week 6 with mask generation results
