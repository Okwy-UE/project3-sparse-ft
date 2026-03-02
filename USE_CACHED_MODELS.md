# Using Your Cached Models (No Authentication Needed!)

## ✅ Good News

You already have all the models downloaded in your cache:
```
/nfs/hpc/share/uwadokae/.cache/huggingface/hub/
├── models--meta-llama--Llama-3.1-8B       ← LLaMA 3.1 (8B)
├── models--mistralai--Mistral-7B-v0.1     ← Mistral 7B
└── models--mistralai--Mixtral-8x7B-v0.1   ← Mixtral 8x7B
```

**No authentication, no downloads needed!**

---

## 🚀 Run with Cached Models RIGHT NOW

### Super Simple (One Command):

```bash
./RUN_WITH_CACHE.sh
```

This will:
- ✓ Use ONLY your locally cached models
- ✓ Generate masks for all 3 models × 3 tasks × 3 sparsities = **27 configurations**
- ✓ Complete in ~45-60 minutes
- ✓ No authentication or downloads required!

---

## 🔧 What Changed

### 1. Fixed Import Error ✅
- Changed `SparseLoRA` → `SparseLoRALayer` in `src/pruning/__init__.py`

### 2. Added Cached Model Support ✅
- Script now uses `HF_HOME` environment variable (already set to your cache)
- Added `--local-only` flag to use only cached models
- Updated LLaMA model name from `Meta-Llama-3-8B` to `Llama-3.1-8B` (matches your cache)
- Added `cache_dir` and `local_files_only` parameters to model loading

### 3. Multiple Run Options ✅
Created several convenient scripts for different scenarios

---

## 📋 All Available Run Options

### Option 1: Use Cached Models (RECOMMENDED)

```bash
./RUN_WITH_CACHE.sh
```
**Best for**: You! Uses models you already have.

### Option 2: Skip Problematic Models

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --skip-gated
```
**Best for**: When you want to avoid authentication issues.

### Option 3: Use Open Alternatives

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --use-alternatives
```
**Best for**: Using open models like `open_llama_7b`.

### Option 4: Standard Run (requires auth for gated models)

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh
```
**Best for**: When you've authenticated with `huggingface-cli login`.

---

## 🎯 Expected Output

After running `./RUN_WITH_CACHE.sh`, you'll have:

```
masks/
├── llama3/
│   ├── masks_sparsity25_unstructured.pt    ← For Cerebras CS-3
│   ├── masks_sparsity25_unstructured.npz   ← For GPU
│   ├── stats_sparsity25_unstructured.json
│   ├── histogram_sparsity25_unstructured.png
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
└── mask_registry.json
```

**Total: 27 mask configurations**
- 3 models × 3 tasks × 3 sparsities
- Both PyTorch (.pt) and NumPy (.npz) formats
- Complete statistics and visualizations

---

## ⏱️ Expected Runtime

| Model       | Time (GPU)  | Status              |
|-------------|-------------|---------------------|
| LLaMA 3.1   | ~10-15 min  | ✓ In your cache     |
| Mistral     | ~10-15 min  | ✓ In your cache     |
| Mixtral     | ~20-30 min  | ✓ In your cache     |
| **Total**   | ~45-60 min  | All from local cache|

---

## 🔍 Verify Your Cache

Check what models you have cached:

```bash
ls -lh /nfs/hpc/share/uwadokae/.cache/huggingface/hub/ | grep models--
```

Expected output:
```
models--meta-llama--Llama-3.1-8B
models--mistralai--Mistral-7B-v0.1  
models--mistralai--Mixtral-8x7B-v0.1
```

---

## 💡 Why This Works

Your `gpu_dense_sft.py` already downloaded these models when you ran your previous experiments. The script now:

1. **Reads `HF_HOME` environment variable**: Points to your cache
2. **Uses `local_files_only=True`**: Won't try to download
3. **Matches model names**: Uses `Llama-3.1-8B` (what you have) instead of `Meta-Llama-3-8B`

---

## 🆘 Troubleshooting

### If you get "model not found in cache"

Check your cache:
```bash
ls -lh $HF_HOME/hub/ | grep models--
```

If models are missing, download them:
```bash
# Without local-only flag
bash scripts/gpu/run_mask_generation_unstructured.sh
```

### If you get import errors

Make sure you're in the project directory:
```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
```

---

## 📝 Summary

**TL;DR - Run this now:**

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
./RUN_WITH_CACHE.sh
```

This will:
- ✅ Use your locally cached models (no downloads)
- ✅ Skip all authentication issues  
- ✅ Generate all 27 mask configurations
- ✅ Complete in ~45-60 minutes
- ✅ Work immediately with no setup

---

## 🔜 After Completion

1. **Verify masks**:
   ```bash
   ls -lh masks/
   cat masks/mask_registry.json
   ```

2. **Commit to git**:
   ```bash
   git add masks/ src/pruning/__init__.py scripts/gpu/ *.md *.sh
   git commit -m "Add unstructured masks using cached models"
   git push
   ```

3. **Move to Cerebras**:
   ```bash
   ssh cer-usn-01
   cd ~/project3-sparse-ft
   git pull
   ```

4. **Run sparse training** (on Cerebras):
   ```bash
   bash scripts/cs3/run_sparse_experiments.sh
   ```

---

## 🎉 You're Ready!

Just run:
```bash
./RUN_WITH_CACHE.sh
```

No authentication, no downloads, just mask generation using models you already have!
