# RUN THIS NOW: Generate Unstructured Masks

## Current Status
You are on a **GPU node** and want to generate unstructured magnitude-based masks for all model-task pairs.

---

## Commands to Run RIGHT NOW

### Step 1: Navigate to project

```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
```

### Step 2: Run mask generation

**IMPORTANT**: LLaMA-3 requires Hugging Face authentication. Choose ONE option:

#### Option A: Skip LLaMA-3 (Fastest - Recommended)

Generate masks for **Mistral** and **Mixtral** only:

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --skip-gated
```

This gives you **18 mask configurations** (2 models × 3 tasks × 3 sparsities) in ~30 minutes.

#### Option B: Use open alternative for LLaMA-3

Use `open_llama_7b` instead of official LLaMA-3:

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --use-alternatives
```

#### Option C: Authenticate and run all models

If you have Hugging Face access to LLaMA-3:

```bash
huggingface-cli login  # Enter your token
bash scripts/gpu/run_mask_generation_unstructured.sh
```

**See `HF_AUTH_GUIDE.md` for detailed authentication instructions.**

---

**RECOMMENDED**: Use **Option A** (skip LLaMA-3) for fastest results, handle LLaMA-3 later if needed.

---

## What This Does

Generates masks for:

### Models (3)
- `llama3` (Meta-Llama-3-8B)
- `mistral` (Mistral-7B-v0.1)
- `mixtral` (Mixtral-8x7B-v0.1)

### Tasks (3)
- `boolq`
- `hellaswag`
- `gsm8k`

### Sparsity Levels (3)
- 25% (light pruning)
- 50% (moderate pruning)
- 75% (aggressive pruning)

### Method
- **Unstructured** (element-wise)
- **Magnitude-based** importance

**Total**: 9 model-task pairs × 3 sparsities = **27 mask configurations**

---

## Expected Timeline

- **LLaMA-3**: ~10-15 minutes
- **Mistral**: ~10-15 minutes
- **Mixtral**: ~20-30 minutes
- **TOTAL**: ~45-60 minutes

---

## What Gets Created

```
masks/
├── llama3/
│   ├── masks_sparsity25_unstructured.pt      ← For Cerebras
│   ├── masks_sparsity25_unstructured.npz     ← For GPU
│   ├── stats_sparsity25_unstructured.json
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

## After Completion

### 1. Verify masks were created

```bash
ls -lh masks/
cat masks/mask_registry.json
```

### 2. Commit to git

```bash
git add masks/ scripts/gpu/generate_all_masks_unstructured.py scripts/gpu/run_mask_generation_unstructured.sh scripts/gpu/README_MASK_GENERATION.md QUICKSTART_MASKS.md RUN_THIS_NOW.md
git commit -m "Add unstructured mask generation pipeline and masks"
git push
```

### 3. Switch to Cerebras node for training

**On Cerebras:**

```bash
ssh ebereuwadoka@cerebras.alcf.anl.gov
ssh cer-usn-01
source ~/R_2.6.0/venv_cerebras_pt/bin/activate
cd ~/project3-sparse-ft
git pull  # Get the masks you generated
```

### 4. Run sparse training

```bash
# Single test run
bash scripts/cs3/run_sparse_experiments.sh train

# Or full sweep (all combinations)
bash scripts/cs3/run_sparse_experiments.sh sweep
```

---

## Optional: Test First (Dry Run)

Want to see what will happen before running?

```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --dry-run
```

---

## Troubleshooting

### If out of memory
Process one model at a time:
```bash
python3 scripts/gpu/generate_all_masks_unstructured.py --models llama3 --device cuda
python3 scripts/gpu/generate_all_masks_unstructured.py --models mistral --device cuda
python3 scripts/gpu/generate_all_masks_unstructured.py --models mixtral --device cuda
```

### If transformers not installed
```bash
pip install transformers --user
```

### If want to use CPU instead
```bash
bash scripts/gpu/run_mask_generation_unstructured.sh --cpu
```

---

## Documentation

- **Quick start**: `QUICKSTART_MASKS.md`
- **Detailed guide**: `scripts/gpu/README_MASK_GENERATION.md`
- **Week 6-8 guide**: `WEEK6-8_GUIDE.md`

---

## Summary

**YOU ARE HERE**: GPU node, ready to generate masks ✓

**RUN THIS NOW**:
```bash
cd /nfs/stak/users/uwadokae/guille/cs599/cerebras-project/project3-sparse-ft
bash scripts/gpu/run_mask_generation_unstructured.sh
```

**THEN**: Commit, push, pull on Cerebras, run training

---

**Ready? Execute the command above!** 🚀
