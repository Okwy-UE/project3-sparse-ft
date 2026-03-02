# START HERE: Generate Masks for Week 6-8

## 🎯 What You Need to Do

Generate unstructured magnitude-based pruning masks for all model-task pairs at 25%, 50%, and 75% sparsity.

---

## ✅ BEST OPTION: Use Your Cached Models

You already have all models downloaded! No authentication needed.

### Run This Command NOW:

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
./RUN_WITH_CACHE.sh
```

**That's it!** 🎉

This will:
- Use models already in your cache (`/nfs/hpc/share/uwadokae/.cache/huggingface/`)
- Generate **27 mask configurations** (3 models × 3 tasks × 3 sparsities)
- Complete in ~45-60 minutes
- Require **ZERO** authentication or downloads

---

## 📊 What You're Generating

### Models (3)
- ✓ **LLaMA 3.1** (8B) - already cached
- ✓ **Mistral** (7B) - already cached  
- ✓ **Mixtral** (8x7B) - already cached

### Tasks (3)
- **BoolQ** - Boolean question answering
- **HellaSwag** - Commonsense reasoning
- **GSM8K** - Grade school math

### Sparsity Levels (3)
- **25%** - Light pruning
- **50%** - Moderate pruning
- **75%** - Aggressive pruning

### Method
- **Unstructured** magnitude-based pruning (matching Phoenix paper)

---

## 📁 Expected Output

```
masks/
├── llama3/
│   ├── masks_sparsity25_unstructured.pt    ← For Cerebras
│   ├── masks_sparsity25_unstructured.npz   ← For GPU
│   ├── masks_sparsity50_unstructured.pt
│   ├── masks_sparsity50_unstructured.npz
│   ├── masks_sparsity75_unstructured.pt
│   └── masks_sparsity75_unstructured.npz
├── mistral/
│   └── (same structure)
├── mixtral/
│   └── (same structure)
└── mask_registry.json
```

---

## ⏱️ Timeline

- **LLaMA 3.1**: ~10-15 minutes
- **Mistral**: ~10-15 minutes
- **Mixtral**: ~20-30 minutes
- **Total**: ~45-60 minutes

---

## 🔧 What Was Fixed

### Issue 1: Import Error ✅
```
ImportError: cannot import name 'SparseLoRA'
```
**Fixed**: Updated `src/pruning/__init__.py` to import `SparseLoRALayer`

### Issue 2: Authentication Error ✅
```
GatedRepoError: Access to model meta-llama/Meta-Llama-3-8B is restricted
```
**Fixed**: 
- Updated to use your cached `Llama-3.1-8B`
- Added `--local-only` flag to use cached models
- No authentication needed!

---

## 📚 Documentation

- **`USE_CACHED_MODELS.md`** - Detailed guide on using cached models
- **`HF_AUTH_GUIDE.md`** - Authentication guide (not needed if using cache)
- **`FIXES_APPLIED.md`** - Complete summary of all fixes
- **`QUICKSTART_MASKS.md`** - Quick start guide
- **`RUN_THIS_NOW.md`** - Simple instructions

---

## 🚀 Alternative Run Options

### If You Want Different Behavior:

#### Skip Gated Models:
```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --skip-gated
```

#### Use Open Alternatives:
```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --use-alternatives
```

#### Custom Models/Tasks:
```bash
python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models mistral \
    --tasks boolq gsm8k \
    --sparsities 0.25 0.50 0.75 \
    --local-only \
    --device cuda
```

---

## ✅ After Completion

### 1. Verify Output
```bash
ls -lh masks/
cat masks/mask_registry.json | head -20
```

### 2. Check Stats
```bash
cat masks/mistral/stats_sparsity50_unstructured.json
```

### 3. Commit to Git
```bash
git add masks/ src/pruning/__init__.py scripts/gpu/ *.md *.sh
git commit -m "Add unstructured magnitude masks using cached models"
git push
```

### 4. Move to Cerebras for Training
```bash
# On Cerebras node
ssh ebereuwadoka@cerebras.alcf.anl.gov
ssh cer-usn-01
source ~/R_2.6.0/venv_cerebras_pt/bin/activate
cd ~/project3-sparse-ft
git pull
```

### 5. Run Sparse Training
```bash
# Test run
bash scripts/cs3/run_sparse_experiments.sh train

# Or full sweep
bash scripts/cs3/run_sparse_experiments.sh sweep
```

---

## 🆘 Quick Troubleshooting

### Problem: Import Error
**Solution**: Already fixed! Just run the command.

### Problem: Model Not Found in Cache
**Solution**: Your models are there! The script will find them at:
```
/nfs/hpc/share/uwadokae/.cache/huggingface/hub/
```

### Problem: Out of Memory
**Solution**: Process one model at a time:
```bash
python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models llama3 --local-only --device cuda

python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models mistral --local-only --device cuda

python3 scripts/gpu/generate_all_masks_unstructured.py \
    --models mixtral --local-only --device cuda
```

---

## 📌 Key Points

- ✅ **No authentication needed** - using your cached models
- ✅ **No downloads needed** - everything is already cached
- ✅ **Import errors fixed** - script updated
- ✅ **Phoenix-compatible** - uses same methodology
- ✅ **Dual format output** - PyTorch for Cerebras, NumPy for GPU
- ✅ **Full validation** - automatic correctness checks

---

## 🎯 THE COMMAND TO RUN RIGHT NOW

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
./RUN_WITH_CACHE.sh
```

**Then wait ~45-60 minutes for completion.**

That's it! 🚀

---

## 📞 Need Help?

See detailed guides:
- **Can't find models?** → `USE_CACHED_MODELS.md`
- **Authentication issues?** → `HF_AUTH_GUIDE.md`
- **Import errors?** → `FIXES_APPLIED.md`
- **Quick reference?** → `QUICKSTART_MASKS.md`

---

## ✨ Summary

1. **You have all models cached** ✓
2. **Import errors are fixed** ✓
3. **Script is updated** ✓
4. **Just run**: `./RUN_WITH_CACHE.sh` ✓

**Ready? Execute the command above!** 🎉
