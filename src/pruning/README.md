# Sparse Pruning and LoRA Implementation

Complete implementation of pruning and sparse LoRA for Cerebras CS-3 experiments.

## Components

### 1. Importance Metrics (`importance.py`)

Compute importance scores for pruning:

```python
from pruning.importance import magnitude_importance, taylor_importance

# Magnitude-based (fast, no data needed)
importance = magnitude_importance(layer)

# Taylor expansion (task-aware)
importance = taylor_importance(layer)  # Requires gradients
```

### 2. Mask Generation (`mask_generator.py`)

Generate pruning masks:

```python
from pruning.mask_generator import MaskGenerator, compute_unstructured_mask

# Option 1: Using MaskGenerator
generator = MaskGenerator(
    model=model,
    sparsity=0.5,
    method="unstructured",
    importance_metric="magnitude"
)
masks = generator.generate_masks()

# Option 2: Direct function
masks = compute_unstructured_mask(
    model=model,
    sparsity=0.5,
    importance_metric="magnitude"
)
```

Supported methods:
- **Unstructured**: Element-wise pruning (highest compression)
- **Structured**: Channel/filter pruning (hardware-friendly)
- **Random**: Random pruning (baseline)

### 3. Mask Operations (`mask_ops.py`)

Validate, save, and analyze masks:

```python
from pruning.mask_ops import (
    validate_mask,
    save_mask,
    load_mask,
    compute_sparsity_stats,
    plot_sparsity_histogram
)

# Validate mask
is_valid, stats = validate_mask(mask, expected_sparsity=0.5, tolerance=0.01)

# Save mask
save_mask(masks, "masks.pt", metadata={"sparsity": 0.5}, format="pt")

# Load mask
masks, metadata = load_mask("masks.pt", format="pt")

# Compute statistics
stats = compute_sparsity_stats(masks)
print(f"Global sparsity: {stats['global']['sparsity']:.2%}")

# Plot histogram
plot_sparsity_histogram(masks, save_path="histogram.png")
```

### 4. Sparse LoRA (`sparse_lora.py`)

Implement sparse LoRA fine-tuning:

```python
from pruning.sparse_lora import (
    SparseLoRAConfig,
    apply_sparse_lora,
    merge_all_lora_weights,
    validate_sparsity_preserved
)

# Configure Sparse LoRA
config = SparseLoRAConfig(
    r=16,                    # LoRA rank
    alpha=32,                # Scaling factor
    dropout=0.05,            # Dropout
    target_modules=["Linear"],
    sparsity_mode="sparse_to_dense",
    initial_sparsity=0.5,
    preserve_sparsity_on_merge=True
)

# Apply to model
model = apply_sparse_lora(model, masks, config)

# Train model (using standard PyTorch training loop)
# ...

# Merge LoRA weights back
model = merge_all_lora_weights(model, preserve_sparsity=True)

# Validate sparsity preserved
is_valid, report = validate_sparsity_preserved(model, original_masks)
print(f"Sparsity preserved: {is_valid}")
```

## Usage Examples

### Example 1: Generate Masks

```python
import torch
import torch.nn as nn
from pruning.mask_generator import compute_unstructured_mask
from pruning.mask_ops import save_mask, compute_sparsity_stats

# Load model
model = ...  # Your model

# Compute masks
masks = compute_unstructured_mask(
    model,
    sparsity=0.7,  # 70% sparsity
    importance_metric="magnitude",
    global_pruning=True
)

# Compute stats
stats = compute_sparsity_stats(masks)
print(f"Generated masks with {stats['global']['sparsity']:.2%} sparsity")

# Save masks
save_mask(
    masks,
    "masks_70pct.pt",
    metadata={
        "sparsity": 0.7,
        "method": "unstructured",
        "importance": "magnitude"
    }
)
```

### Example 2: Sparse-to-Dense Training

```python
from pruning.mask_ops import load_mask
from pruning.sparse_lora import SparseLoRAConfig, apply_sparse_lora

# Load masks
masks, metadata = load_mask("masks_70pct.pt")
print(f"Loaded masks: {metadata}")

# Apply sparse LoRA (sparse-to-dense mode)
config = SparseLoRAConfig(
    r=16,
    sparsity_mode="sparse_to_dense",
    preserve_sparsity_on_merge=True
)

model = apply_sparse_lora(model, masks, config)

# Base weights are sparse, LoRA adapters are dense
# After training, merge will re-apply sparsity
```

### Example 3: Sparse-to-Sparse Training

```python
# Apply sparse LoRA (sparse-to-sparse mode)
config = SparseLoRAConfig(
    r=16,
    sparsity_mode="sparse_to_sparse",
    maintain_sparsity=True,  # Enforce sparsity during training
    preserve_sparsity_on_merge=True
)

model = apply_sparse_lora(model, masks, config)

# Sparsity is maintained throughout training
# Masks are applied after each gradient update
```

## Cerebras Integration

### Using Sparse Masks with Cerebras Training

```python
from utils.cerebras_sparse_callback import SparseMaskCallback, create_sparse_config

# Create callback
callback = SparseMaskCallback(
    mask_path="masks_70pct.pt",
    apply_every_step=False,  # Set True for sparse-to-sparse
)

# Modify Cerebras config
base_config = {...}  # Your base config
sparse_config = create_sparse_config(
    base_config,
    mask_path="masks_70pct.pt",
    lora_config={
        "r": 16,
        "alpha": 32,
        "dropout": 0.05
    },
    sparse_mode="sparse_to_dense"
)

# Train with cszoo
# cszoo fit sparse_config.yaml ...
```

## Validation

### Validate Masks

```python
from pruning.mask_ops import validate_mask

for name, mask in masks.items():
    is_valid, stats = validate_mask(
        mask,
        expected_sparsity=0.5,
        tolerance=0.01
    )
    
    if not is_valid:
        print(f"Invalid mask: {name}")
        print(f"  Error: {stats['error']}")
    else:
        print(f"Valid: {name} ({stats['sparsity']:.2%} sparsity)")
```

### Validate Sparsity Preservation After Training

```python
from pruning.sparse_lora import validate_sparsity_preserved

# After merging LoRA weights
is_valid, report = validate_sparsity_preserved(
    model,
    original_masks,
    tolerance=1e-6
)

if is_valid:
    print("✓ Sparsity preserved after LoRA merge")
else:
    print("✗ Sparsity NOT preserved")
    print(f"  Violations: {report['global']['total_violations']}")
    print(f"  Rate: {report['global']['violation_rate']:.2%}")
```

## File Formats

### Mask Storage

Supports three formats:

**PyTorch (.pt)**
```python
save_mask(masks, "masks.pt", format="pt")
masks, metadata = load_mask("masks.pt", format="pt")
```

**NumPy (.npz)**
```python
save_mask(masks, "masks.npz", format="npz")
masks, metadata = load_mask("masks.npz", format="npz")
```

**SafeTensors (.safetensors)**
```python
save_mask(masks, "masks.safetensors", format="safetensors")
masks, metadata = load_mask("masks.safetensors", format="safetensors")
```

## Statistics and Visualization

### Compute Comprehensive Stats

```python
from pruning.mask_ops import compute_sparsity_stats

stats = compute_sparsity_stats(masks)

# Global stats
print(f"Global sparsity: {stats['global']['sparsity']:.2%}")
print(f"Mean layer sparsity: {stats['global']['mean_layer_sparsity']:.2%}")
print(f"Std layer sparsity: {stats['global']['std_layer_sparsity']:.2%}")

# Per-layer stats
for layer, layer_stats in stats['per_layer'].items():
    print(f"{layer}: {layer_stats['sparsity']:.2%}")
```

### Visualize Sparsity Distribution

```python
from pruning.mask_ops import plot_sparsity_histogram

plot_sparsity_histogram(masks, save_path="sparsity_hist.png")
```

## Advanced Features

### Custom Importance Metric

```python
from pruning.importance import compute_layer_importance

def custom_importance(module, name):
    # Your custom importance computation
    weight = module.weight
    importance = torch.abs(weight) ** 2  # L2 norm example
    return importance

# Register custom metric
importance_dict = compute_layer_importance(
    model,
    method="custom"  # Will fall back to magnitude
)
```

### Layer-Specific Sparsity

```python
# Different sparsity per layer
layer_sparsities = {
    "layer1": 0.3,
    "layer2": 0.5,
    "layer3": 0.7
}

masks = {}
for name, module in model.named_modules():
    if name in layer_sparsities:
        sparsity = layer_sparsities[name]
        # Compute mask for this layer
        ...
```

### Gradual Pruning

```python
# Iterative pruning schedule
sparsity_schedule = [0.0, 0.2, 0.4, 0.6, 0.8]

for sparsity in sparsity_schedule:
    # Compute masks
    masks = compute_unstructured_mask(model, sparsity)
    
    # Apply masks
    apply_masks_to_model(model, masks)
    
    # Fine-tune for N steps
    train(model, num_steps=1000)
```

## Performance Considerations

### Memory-Efficient Mask Generation

```python
# Generate masks without loading full model
from pruning.mask_generator import compute_masks_for_state_dict

state_dict = torch.load("model.pt", map_location="cpu")
masks = compute_masks_for_state_dict(
    state_dict,
    sparsity=0.5,
    method="unstructured"
)
```

### Distributed Mask Computation

For very large models, compute masks per-layer in parallel:

```python
import multiprocessing as mp

def compute_layer_mask(layer_name, layer_weight, sparsity):
    # Compute mask for single layer
    ...

with mp.Pool(processes=8) as pool:
    results = pool.starmap(compute_layer_mask, layer_tasks)
```

## Testing

Run tests to verify implementation:

```bash
# Unit tests for mask generation
pytest src/pruning/tests/test_mask_generator.py

# Integration tests for sparse LoRA
pytest src/pruning/tests/test_sparse_lora.py

# End-to-end tests
pytest src/pruning/tests/test_e2e.py
```

## References

- [Lottery Ticket Hypothesis](https://arxiv.org/abs/1803.03635)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [Magnitude Pruning](https://arxiv.org/abs/1506.02626)
- [Structured Pruning](https://arxiv.org/abs/1608.08710)
