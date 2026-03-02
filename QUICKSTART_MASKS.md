# Quick Start: Generate Unstructured Masks

**Goal**: Generate unstructured magnitude-based pruning masks for all model-task pairs at 25%, 50%, and 75% sparsity.

---

## Prerequisites

You mentioned you're currently on a GPU node. Perfect! That's all you need.

---

## Step-by-Step Instructions

### 1. Navigate to project directory

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
```

### 2. Run the mask generation script

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh
```

That's it! The script will:
- ✓ Automatically detect your GPU
- ✓ Load models from Hugging Face
- ✓ Generate masks for all 9 model-task pairs
- ✓ Generate masks at 3 sparsity levels (25%, 50%, 75%)
- ✓ Save in both PyTorch format (for Cerebras) and NumPy format (for GPU)
- ✓ Create validation statistics and histograms
- ✓ Generate a master registry

**Total: 27 mask files (9 pairs × 3 sparsities)**

---

## What Gets Generated

```
masks/
├── llama3/
│   ├── masks_sparsity25_unstructured.pt      # For Cerebras
│   ├── masks_sparsity25_unstructured.npz     # For GPU
│   ├── stats_sparsity25_unstructured.json    # Statistics
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
│   └── (same structure)
├── mixtral/
│   └── (same structure)
└── mask_registry.json                         # Master index
```

---

## Expected Runtime

- **LLaMA-3** + **Mistral**: ~10-15 minutes each on GPU
- **Mixtral**: ~20-30 minutes on GPU (larger model)
- **Total**: ~45-60 minutes for all pairs

---

## Optional: Dry Run First

If you want to see what will be done without actually doing it:

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --dry-run
```

---

## Verify Output

After completion:

```bash
# Check files were created
ls -lh masks/*/

# View registry
cat masks/mask_registry.json

# Check a sample stats file
cat masks/mistral/stats_sparsity50_unstructured.json | head -30
```

---

## Next Steps (After Mask Generation)

### 1. Commit to git

```bash
git add masks/
git commit -m "Add unstructured magnitude-based masks (25%, 50%, 75% sparsity)"
git push
```

### 2. Move to Cerebras node for training

On your Cerebras node:

```bash
ssh cer-usn-01
source ~/R_2.6.0/venv_cerebras_pt/bin/activate
cd ~/project3-sparse-ft
git pull  # Get the masks you just generated
```

### 3. Run sparse training experiments

```bash
# Test with single run
bash scripts/cs3/run_sparse_experiments.sh train

# Or run full sweep
bash scripts/cs3/run_sparse_experiments.sh sweep
```

---

## Troubleshooting

### Out of memory?

Process one model at a time:

```bash
python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models llama3 \
    --device cuda

python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models mistral \
    --device cuda

python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models mixtral \
    --device cuda
```

### No transformers package?

```bash
pip install transformers --user
```

### Need to use CPU instead?

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --cpu
```

---

## What This Does Under the Hood

For each model-task pair:

1. **Load model** from Hugging Face (e.g., `meta-llama/Meta-Llama-3-8B`)
2. **Compute importance** using magnitude-based scoring: `importance = |weight|`
3. **Apply global threshold** to achieve target sparsity across all layers
4. **Generate binary masks** (0 = prune, 1 = keep)
5. **Validate** masks are binary and have correct sparsity
6. **Save** in multiple formats (PyTorch for Cerebras, NumPy for GPU)
7. **Generate statistics** and visualizations

---

## Configuration

The script uses these settings (matching Phoenix paper):

- **Models**: `llama3`, `mistral`, `mixtral`
- **Tasks**: `boolq`, `hellaswag`, `gsm8k`
- **Sparsities**: `25%`, `50%`, `75%`
- **Method**: Unstructured (element-wise)
- **Importance**: Magnitude-based (|weight|)
- **Scope**: Global (single threshold across all layers)

---

## Questions?

See detailed documentation:
- `scripts/gpu/README_MASK_GENERATION.md` - Comprehensive guide
- `WEEK6-8_GUIDE.md` - Week 6-8 implementation guide
- `EXPERIMENTS.md` - Full experimental design

---

**Ready? Run this now:**

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
bash scripts/gpu/run_mask_generation_unstructured.sh
```
