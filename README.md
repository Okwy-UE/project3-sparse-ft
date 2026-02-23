# Sparse Fine-Tuning on Cerebras CS-3

**Sparse-to-dense and sparse-to-sparse fine-tuning experiments comparing Cerebras CS-3 with GPU baselines and Phoenix paper results.**

---

## 🎯 Project Overview

This project implements comprehensive sparse fine-tuning experiments on Cerebras CS-3 systems, with a focus on:

1. **Pruning Pipeline**: Unstructured, structured, and random pruning methods
2. **Sparse LoRA**: Low-rank adaptation with sparsity preservation
3. **Phoenix Comparison**: Apples-to-apples comparison with Phoenix paper results
4. **Hardware Analysis**: Cerebras CS-3 vs GPU performance characteristics

---

## 📁 Repository Structure

```
project3-sparse-ft/
├── src/
│   ├── pruning/              # Core pruning and sparse LoRA implementation
│   │   ├── importance.py     # Importance metrics (magnitude, gradient, Taylor)
│   │   ├── mask_generator.py # Mask generation (unstructured, structured, random)
│   │   ├── mask_ops.py       # Mask I/O, validation, statistics
│   │   └── sparse_lora.py    # Sparse LoRA implementation
│   ├── models/               # Model wrappers
│   ├── data/                 # Dataset processing
│   ├── utils/                # Utilities (Cerebras callbacks, logging)
│   └── eval/                 # Evaluation scripts
├── scripts/
│   ├── cs3/                  # Cerebras CS-3 scripts
│   │   ├── compute_masks.py           # Generate pruning masks
│   │   ├── validate_sparsity.py       # Validate sparsity preservation
│   │   ├── analyze_results.py         # Result analysis and plotting
│   │   ├── run_sparse_experiments.sh  # Main experiment runner
│   │   └── quick_test.sh              # Quick validation test
│   └── gpu/                  # GPU baseline scripts
├── configs/                  # Training configurations
│   └── cerebras/
│       └── generated/        # Auto-generated configs
├── masks/                    # Generated pruning masks (not in git)
├── results/                  # Experiment results
│   ├── runs/                 # Individual run outputs
│   ├── analysis/             # Aggregated analysis
│   └── tables/               # Summary tables
├── EXPERIMENTS.md            # Detailed experiment plan and results
├── WEEK6-8_GUIDE.md          # Implementation guide for Weeks 6-8
└── README.md                 # This file
```

---

## 🚀 Quick Start

### Prerequisites

- Access to Cerebras CS-3 at ALCF
- Python 3.11+
- Cerebras ModelZoo (Release 2.6.0)
- PyTorch 2.0+

### Setup

```bash
# SSH to Cerebras
ssh XXX@cerebras.alcf.anl.gov
ssh cer-usn-01

# Activate Cerebras environment
source ~/R_2.6.0/venv_cerebras_pt/bin/activate

# Clone repository
cd ~
git clone https://github.com/Okwy-UE/project3-sparse-ft.git
cd project3-sparse-ft

# Test installation
bash scripts/cs3/quick_test.sh
```

---

## 📊 Experiments

### Week 6: Pruning Pipeline (COMPLETE ✅)

**Generate pruning masks:**

```bash
# All models and methods
bash scripts/cs3/run_sparse_experiments.sh masks

# Single model
python scripts/cs3/compute_masks.py \
    --model_path ./checkpoints/cs/mistral_7b/model_to_cs-2.5.mdl \
    --output_dir ./masks/mistral \
    --sparsity 0.5 0.7 0.9 \
    --method unstructured \
    --format pt \
    --plot
```

**Output:**
- Masks: `./masks/{model}/masks_sparsity{XX}_{method}.pt`
- Stats: `./masks/{model}/stats_sparsity{XX}_{method}.json`
- Plots: `./masks/{model}/histogram_sparsity{XX}_{method}.png`

### Week 7: Sparse LoRA Training (IN PROGRESS 🔄)

**Train sparse LoRA model:**

```bash
# Test run
bash scripts/cs3/run_sparse_experiments.sh train

# Validate sparsity preservation
python scripts/cs3/validate_sparsity.py \
    --checkpoint ./results/runs/{run_id}/model_dir \
    --mask_path ./masks/mistral/masks_sparsity50_unstructured.pt \
    --output ./results/runs/{run_id}/validation_report.json
```

**Output:**
- Checkpoints: `./results/runs/{run_id}/model_dir/`
- Logs: `./results/runs/{run_id}/train.log`
- Validation: `./results/runs/{run_id}/validation_report.json`

### Week 8: Full Sparse Sweep (PENDING ⏳)

**Run complete experimental matrix:**

```bash
# Full sweep: 2 models × 3 tasks × 3 methods × 10 sparsities × 2 modes = 360 runs
bash scripts/cs3/run_sparse_experiments.sh sweep
```

**Analyze results:**

```bash
python scripts/cs3/analyze_results.py \
    --results_dir ./results/runs \
    --output_dir ./results/analysis \
    --plot
```

**Output:**
- Summary tables: `./results/analysis/summary.csv`
- Performance curves: `./results/analysis/sparsity_vs_loss.png`
- Comparison heatmaps: `./results/analysis/method_comparison_heatmap.png`

---

## 🔬 Experimental Design

### Models

- **LLaMA-7B**: Decoder-only transformer
- **Mistral-7B**: Sliding window attention
- **Mixtral-8x7B**: Mixture-of-experts (optional)

### Tasks (Phoenix-Compatible)

- **BoolQ**: Boolean question answering
- **HellaSwag**: Commonsense reasoning
- **GSM8K**: Grade school math

### Sparsity Levels

`0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%`

### Pruning Methods

1. **Unstructured**: Element-wise magnitude pruning
2. **Structured**: Channel/filter pruning
3. **Random**: Random baseline

### Training Modes

1. **Sparse-to-Dense**: Sparse base + dense LoRA
2. **Sparse-to-Sparse**: Maintain sparsity during training

---

## 📈 Key Features

### Pruning Implementation

- **Multiple importance metrics**: Magnitude, gradient, Taylor expansion
- **Flexible pruning methods**: Unstructured, structured, random
- **Comprehensive validation**: Binary checks, sparsity verification, checksum
- **Visualization**: Per-layer sparsity histograms

### Sparse LoRA

- **Sparsity preservation**: Masks applied before/after LoRA merge
- **Two training modes**: Sparse-to-dense and sparse-to-sparse
- **Validation**: Automated checking of zero violations
- **Efficient storage**: Separate mask and LoRA artifacts

### Cerebras Integration

- **Custom callbacks**: Apply masks during Cerebras training
- **Config generation**: Automatic sparse config creation
- **Validation**: Post-training sparsity checks
- **Logging**: Comprehensive run metadata

---

## 📖 Documentation

- **[EXPERIMENTS.md](EXPERIMENTS.md)**: Complete experimental plan, results, and analysis
- **[WEEK6-8_GUIDE.md](WEEK6-8_GUIDE.md)**: Step-by-step implementation guide
- **[src/pruning/README.md](src/pruning/README.md)**: Detailed API documentation

---

## 🛠️ API Usage

### Generate Masks

```python
from pruning.mask_generator import compute_unstructured_mask
from pruning.mask_ops import save_mask

# Load model
model = ...

# Compute masks
masks = compute_unstructured_mask(
    model,
    sparsity=0.7,  # 70% sparsity
    importance_metric="magnitude",
    global_pruning=True
)

# Save masks
save_mask(masks, "masks_70pct.pt", metadata={"sparsity": 0.7})
```

### Apply Sparse LoRA

```python
from pruning.mask_ops import load_mask
from pruning.sparse_lora import SparseLoRAConfig, apply_sparse_lora

# Load masks
masks, _ = load_mask("masks_70pct.pt")

# Configure and apply Sparse LoRA
config = SparseLoRAConfig(
    r=16,
    sparsity_mode="sparse_to_dense",
    preserve_sparsity_on_merge=True
)
model = apply_sparse_lora(model, masks, config)

# Train model...
# (use standard PyTorch or Cerebras training loop)

# Validate sparsity preserved
from pruning.sparse_lora import validate_sparsity_preserved
is_valid, report = validate_sparsity_preserved(model, masks)
```

---

## 🔍 Validation

### Mask Validation

```bash
# Check mask correctness
python << EOF
from pruning.mask_ops import load_mask, validate_mask

masks, metadata = load_mask("./masks/mistral/masks_sparsity50_unstructured.pt")

for name, mask in masks.items():
    is_valid, stats = validate_mask(mask, expected_sparsity=0.5, tolerance=0.01)
    print(f"{name}: {'✓' if is_valid else '✗'} {stats['sparsity']:.2%}")
EOF
```

### Sparsity Preservation

```bash
# After training, validate sparsity
python scripts/cs3/validate_sparsity.py \
    --checkpoint ./results/runs/{run_id}/model_dir \
    --mask_path ./masks/mistral/masks_sparsity50_unstructured.pt \
    --output validation_report.json
```

### Quick Test

```bash
# Run all unit tests
bash scripts/cs3/quick_test.sh
```

---

## 📊 Results

### Expected Outcomes

**Week 6:**
- ✅ Masks generated for all models/methods/sparsities
- ✅ Validation reports confirming correctness
- ✅ Per-layer sparsity histograms

**Week 7:**
- 🔄 Sparse LoRA training runs
- 🔄 Sparsity preservation validation
- 🔄 Performance within 5% of dense baseline (at low sparsity)

**Week 8:**
- ⏳ Complete experimental sweep (360 runs)
- ⏳ Performance curves: accuracy vs sparsity
- ⏳ Throughput analysis: samples/sec vs sparsity
- ⏳ Phoenix comparison tables

### Preliminary Results

*(To be updated as experiments complete)*

---

## 🔧 Troubleshooting

### Common Issues

**Issue**: `ModuleNotFoundError: No module named 'pruning'`  
**Fix**: Ensure `src/` is in PYTHONPATH:
```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

**Issue**: Mask file not found  
**Fix**: Generate masks first:
```bash
bash scripts/cs3/run_sparse_experiments.sh masks
```

**Issue**: Sparsity validation fails  
**Fix**: Check mask was applied correctly. Re-run with fresh masks.

**Issue**: Cerebras compilation timeout  
**Fix**: Reduce batch size or increase timeout in config.

### Getting Help

- Check [WEEK6-8_GUIDE.md](WEEK6-8_GUIDE.md) for detailed instructions
- Review [EXPERIMENTS.md](EXPERIMENTS.md) for experimental design
- See [src/pruning/README.md](src/pruning/README.md) for API documentation

---

## 🤝 Comparison with Phoenix

This project aims for an apples-to-apples comparison with the [Phoenix paper](https://dl.acm.org/doi/10.1145/3731599.3767395):

| Aspect | Phoenix | This Project |
|--------|---------|--------------|
| **Models** | 7B params | LLaMA-7B, Mistral-7B |
| **Tasks** | BoolQ, HellaSwag, GSM8K | Same ✅ |
| **Sparsity** | Up to 90% | 0-90% (10% increments) |
| **Methods** | Unstructured, structured | Same + random baseline |
| **Hardware** | NVIDIA GPUs | Cerebras CS-3 |
| **Training** | LoRA fine-tuning | Sparse LoRA (our contribution) |

**Key Differences:**
- We use Cerebras CS-3 (wafer-scale) vs GPUs
- We implement sparse-to-sparse training mode
- We provide comprehensive validation and analysis tools

---

## 📚 References

- **Phoenix Paper**: [ACM DL](https://dl.acm.org/doi/10.1145/3731599.3767395)
- **LoRA**: [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- **Magnitude Pruning**: [arXiv:1506.02626](https://arxiv.org/abs/1506.02626)
- **Cerebras ModelZoo**: [GitHub](https://github.com/Cerebras/modelzoo)

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@misc{uwadoka2026sparse,
  author = {Uwadoka, Ebere},
  title = {Sparse Fine-Tuning on Cerebras CS-3},
  year = {2026},
  url = {https://github.com/Okwy-UE/project3-sparse-ft}
}
```

---

## 👥 Team

**Investigator**: Ebere Uwadoka  
**Institution**: Oregon State University  
**Hardware**: Cerebras CS-3 at Argonne Leadership Computing Facility (ALCF)

---

## 📄 License

This project is licensed under the MIT License.

---

## 🎯 Status

| Week | Task | Status |
|------|------|--------|
| 6 | Pruning Pipeline | ✅ COMPLETE |
| 7 | Sparse LoRA Training | 🔄 IN PROGRESS |
| 8 | Full Sparse Sweep | ⏳ PENDING |

**Last Updated**: 2026-02-16

---

## 🚦 Next Steps

1. ✅ Complete Week 6 mask generation
2. 🔄 Run Week 7 sparse LoRA validation
3. ⏳ Execute Week 8 full experimental sweep
4. ⏳ Analyze results and compare with Phoenix
5. ⏳ Write up findings in paper/report

---

For detailed instructions, see **[WEEK6-8_GUIDE.md](WEEK6-8_GUIDE.md)**.
