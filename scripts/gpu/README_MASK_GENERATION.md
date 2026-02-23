# GPU Mask Generation Guide

This guide explains how to generate pruning masks on GPU nodes for use on Cerebras CS-3.

## Quick Start

### 1. Ensure you're on a GPU node

Since you mentioned you're currently on a GPU node, you're ready to go. The scripts will automatically detect GPU availability.

### 2. Run the mask generation script

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft

# Full run (all models, all tasks, all sparsity levels)
bash scripts/gpu/run_mask_generation_unstructured.sh
```

### 3. Verify the output

After completion, check the generated masks:

```bash
ls -lh masks/
```

Expected output structure:
```
masks/
├── llama3/
│   ├── masks_sparsity25_unstructured.pt      # PyTorch format (for Cerebras)
│   ├── masks_sparsity25_unstructured.npz     # NumPy format (for GPU)
│   ├── stats_sparsity25_unstructured.json    # Statistics and metadata
│   ├── histogram_sparsity25_unstructured.png # Visualization
│   ├── masks_sparsity50_unstructured.pt
│   ├── masks_sparsity50_unstructured.npz
│   ├── stats_sparsity50_unstructured.json
│   ├── histogram_sparsity50_unstructured.png
│   ├── masks_sparsity75_unstructured.pt
│   ├── masks_sparsity75_unstructured.npz
│   ├── stats_sparsity75_unstructured.json
│   └── histogram_sparsity75_unstructured.png
├── mistral/
│   └── (same structure as llama3)
├── mixtral/
│   └── (same structure as llama3)
└── mask_registry.json                         # Master registry
```

---

## Options

### Dry Run (see what would be done)

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --dry-run
```

### CPU-only execution

If you want to test on CPU first:

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --cpu
```

### Skip histogram plots

If matplotlib is not available or you want to skip plotting:

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --no-plot
```

---

## Advanced Usage

### Generate masks for specific models only

```bash
python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models llama3 mistral \
    --tasks boolq gsm8k \
    --sparsities 0.25 0.50 \
    --device cuda
```

### Generate masks with custom sparsity levels

```bash
python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models mistral \
    --tasks boolq \
    --sparsities 0.1 0.3 0.5 0.7 0.9 \
    --device cuda
```

---

## Configuration Details

### Models Processed
- **llama3**: Meta-Llama-3-8B (8B parameters)
- **mistral**: Mistral-7B-v0.1 (7B parameters)
- **mixtral**: Mixtral-8x7B-v0.1 (47B parameters, MoE)

### Tasks (Phoenix-Compatible)
- **boolq**: Boolean question answering
- **hellaswag**: Commonsense reasoning
- **gsm8k**: Grade school math

### Sparsity Levels
- **25%**: Light pruning (75% of weights kept)
- **50%**: Moderate pruning (50% of weights kept)
- **75%**: Aggressive pruning (25% of weights kept)

### Pruning Method
- **Method**: Unstructured (element-wise)
- **Importance**: Magnitude-based (|weight|)
- **Scope**: Global (threshold computed across all layers)

---

## Output Formats

### PyTorch Format (`.pt`)
Used by Cerebras CS-3 training scripts. Contains:
- `masks`: Dictionary mapping layer names to binary masks
- `metadata`: Complete configuration and statistics

Load with:
```python
from pruning.mask_ops import load_mask
masks, metadata = load_mask("masks/mistral/masks_sparsity50_unstructured.pt")
```

### NumPy Format (`.npz`)
Used by GPU training scripts. Contains:
- Individual mask arrays for each layer
- Separate `.json` file with metadata

Load with:
```python
import numpy as np
masks_npz = np.load("masks/mistral/masks_sparsity50_unstructured.npz")
```

### Statistics JSON (`.json`)
Human-readable statistics including:
- Global sparsity (actual vs target)
- Per-layer sparsity distribution
- Validation results
- Metadata (creation time, config, etc.)

### Histogram PNG (`.png`)
Visual representation of per-layer sparsity distribution.

---

## Validation

### Automatic Validation

The script automatically validates masks during generation:
- ✓ Binary check (all values are 0 or 1)
- ✓ Sparsity check (within 5% tolerance of target)
- ✓ Statistics computation (global and per-layer)

### Manual Validation

Verify masks after generation:

```bash
python3 << 'EOF'
from pruning.mask_ops import load_mask, validate_mask, compute_sparsity_stats

# Load masks
masks, metadata = load_mask("masks/mistral/masks_sparsity50_unstructured.pt")

# Validate
for name, mask in list(masks.items())[:5]:  # Show first 5 layers
    is_valid, stats = validate_mask(mask, expected_sparsity=0.5, tolerance=0.05)
    print(f"{name}: {'✓' if is_valid else '✗'} sparsity={stats['sparsity']:.2%}")

# Global stats
stats = compute_sparsity_stats(masks)
print(f"\nGlobal sparsity: {stats['global']['sparsity']:.2%}")
print(f"Number of layers: {stats['global']['num_layers']}")
EOF
```

---

## Troubleshooting

### Issue: Out of memory on GPU

**Solution 1**: Process models one at a time
```bash
python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models llama3 \
    --device cuda
```

**Solution 2**: Use CPU
```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --cpu
```

### Issue: Hugging Face model download fails

**Solution**: Pre-download models or use cached versions
```bash
# Set HF cache directory
export HF_HOME=/path/to/cache
export TRANSFORMERS_CACHE=/path/to/cache

# Pre-download
python3 -c "from transformers import AutoModel; AutoModel.from_pretrained('mistralai/Mistral-7B-v0.1')"
```

### Issue: `transformers` not installed

**Solution**: Install transformers
```bash
pip install transformers torch
```

### Issue: Import errors for `pruning` module

**Solution**: Ensure you're in project root and `src/` is in PYTHONPATH
```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python3 scripts/gpu/generate_all_masks_unstructured.py
```

---

## Expected Runtime

Approximate times on different hardware:

| Model    | GPU (A100)  | GPU (V100)  | CPU (32 cores) |
|----------|-------------|-------------|----------------|
| llama3   | ~5 min      | ~10 min     | ~30 min        |
| mistral  | ~5 min      | ~10 min     | ~30 min        |
| mixtral  | ~15 min     | ~30 min     | ~2 hours       |

**Total for all pairs**: ~30-60 minutes on GPU, 3-6 hours on CPU

---

## Phoenix Compatibility

These masks are designed to be compatible with the Phoenix paper methodology:

| Aspect              | Phoenix                    | Our Implementation         |
|---------------------|----------------------------|----------------------------|
| **Models**          | 7B LLMs                    | LLaMA-7B, Mistral-7B ✓     |
| **Tasks**           | BoolQ, HellaSwag, GSM8K    | Same ✓                     |
| **Pruning Method**  | Magnitude-based            | Same ✓                     |
| **Scope**           | Global pruning             | Same ✓                     |
| **Format**          | Custom                     | PyTorch + NumPy ✓          |

---

## Next Steps

After generating masks:

1. **Commit to git** (masks are in shared repo):
   ```bash
   cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
   git add masks/
   git commit -m "Add unstructured magnitude-based masks (25%, 50%, 75%)"
   git push
   ```

2. **Transfer to Cerebras node**:
   ```bash
   # On Cerebras node (cer-usn-01)
   ssh cer-usn-01
   source ~/R_2.6.0/venv_cerebras_pt/bin/activate
   cd ~/project3-sparse-ft
   git pull
   ```

3. **Run sparse training**:
   ```bash
   # On Cerebras node
   bash scripts/cs3/run_sparse_experiments.sh train
   ```

4. **Run full sweep**:
   ```bash
   # On Cerebras node
   bash scripts/cs3/run_sparse_experiments.sh sweep
   ```

---

## Documentation

- **Mask format specification**: `src/pruning/README.md`
- **API documentation**: `src/pruning/mask_ops.py`
- **Experimental design**: `EXPERIMENTS.md`
- **Week 6-8 guide**: `WEEK6-8_GUIDE.md`

---

## Support

For questions or issues:
1. Check `WEEK6-8_GUIDE.md` for detailed instructions
2. Review `EXPERIMENTS.md` for experimental design
3. Inspect generated `mask_registry.json` for configuration details
