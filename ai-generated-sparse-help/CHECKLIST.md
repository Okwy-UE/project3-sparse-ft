# Weeks 6-8 Checklist

## Week 6: Pruning Pipeline ✅

### Implementation
- [x] importance.py (magnitude, gradient, Taylor)
- [x] mask_generator.py (unstructured, structured, random)
- [x] mask_ops.py (I/O, validation, stats)
- [x] compute_masks.py script
- [x] Documentation

### Execution
- [ ] Run: `bash scripts/cs3/run_sparse_experiments.sh masks`
- [ ] Verify masks created in `./masks/`
- [ ] Check statistics files
- [ ] Review histograms

### Deliverables
- [ ] Masks for all models (LLaMA, Mistral)
- [ ] All sparsity levels (0-90%)
- [ ] All methods (unstructured, structured, random)
- [ ] Validation reports
- [ ] Histograms

---

## Week 7: Sparse LoRA 🔄

### Implementation
- [x] sparse_lora.py (SparseLoRALayer, config)
- [x] sparse_model_wrapper.py
- [x] cerebras_sparse_callback.py
- [x] validate_sparsity.py script
- [x] Documentation

### Execution
- [ ] Run: `bash scripts/cs3/run_sparse_experiments.sh train`
- [ ] Monitor training logs
- [ ] Validate sparsity preservation
- [ ] Check performance metrics

### Deliverables
- [ ] Sparse LoRA training runs complete
- [ ] Validation reports (sparsity preserved)
- [ ] Correctness memo
- [ ] Performance within tolerance

---

## Week 8: Full Sweep ⏳

### Implementation
- [x] run_sparse_experiments.sh (full sweep)
- [x] analyze_results.py
- [x] Documentation

### Execution
- [ ] Run: `bash scripts/cs3/run_sparse_experiments.sh sweep`
- [ ] Monitor 360 runs
- [ ] Analyze results
- [ ] Generate plots

### Deliverables
- [ ] All 360 runs complete
- [ ] Summary tables (CSV)
- [ ] Performance curves
- [ ] Throughput analysis
- [ ] Phoenix comparison

---

## Final Deliverables

- [ ] Pruning + mask spec (WEEK6-8_GUIDE.md)
- [ ] Correctness memo (validation reports)
- [ ] Sparse results dataset (results/analysis/)
- [ ] Preliminary plots
- [ ] Phoenix comparison tables

---

**Status**: Implementation ✅ | Testing ⏳  
**Next**: Run quick_test.sh on Cerebras
