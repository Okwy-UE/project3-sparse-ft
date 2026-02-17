#!/usr/bin/env bash
#
# Quick test script to verify sparse pipeline works
#

set -euo pipefail

echo "========================================"
echo "Quick Test: Sparse Pipeline"
echo "========================================"

cd "$(dirname "$0")/../.."

# Test 1: Import modules
echo -e "\n[1/5] Testing Python imports..."
python3 << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

try:
    from pruning import MaskGenerator, compute_unstructured_mask
    from pruning.sparse_lora import SparseLoRAConfig, SparseLoRALayer
    from pruning.mask_ops import save_mask, load_mask
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)
EOF

# Test 2: Generate dummy masks
echo -e "\n[2/5] Testing mask generation..."
python3 << 'EOF'
import sys
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from pruning.mask_generator import compute_unstructured_mask
from pruning.mask_ops import save_mask, validate_mask, compute_sparsity_stats

# Create dummy model
class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
    
    def forward(self, x):
        return self.fc2(self.fc1(x))

model = DummyModel()

# Generate masks
masks = compute_unstructured_mask(model, sparsity=0.5)

print(f"✓ Generated {len(masks)} masks")

# Validate
for name, mask in masks.items():
    is_valid, stats = validate_mask(mask, expected_sparsity=0.5, tolerance=0.1)
    if is_valid:
        print(f"  ✓ {name}: {stats['sparsity']:.2%} sparsity")
    else:
        print(f"  ✗ {name}: validation failed")

# Save masks
save_mask(masks, "/tmp/test_masks.pt", metadata={"test": True})
print("✓ Masks saved to /tmp/test_masks.pt")
EOF

# Test 3: Load masks
echo -e "\n[3/5] Testing mask loading..."
python3 << 'EOF'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from pruning.mask_ops import load_mask

masks, metadata = load_mask("/tmp/test_masks.pt")
print(f"✓ Loaded {len(masks)} masks")
print(f"  Metadata: {metadata}")
EOF

# Test 4: Sparse LoRA
echo -e "\n[4/5] Testing Sparse LoRA..."
python3 << 'EOF'
import sys
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from pruning.sparse_lora import SparseLoRAConfig, SparseLoRALayer
from pruning.mask_ops import load_mask

# Load masks
masks, _ = load_mask("/tmp/test_masks.pt")

# Create dummy layer
base_layer = nn.Linear(128, 256)
mask = torch.ones_like(base_layer.weight)
mask[mask > 0.5] = 0  # 50% sparsity

# Create Sparse LoRA layer
config = SparseLoRAConfig(r=8, alpha=16)
lora_layer = SparseLoRALayer(base_layer, mask, config)

# Test forward pass
x = torch.randn(32, 128)
y = lora_layer(x)

print(f"✓ Sparse LoRA forward pass: {x.shape} -> {y.shape}")

# Test merge
lora_layer.merge_weights(preserve_sparsity=True)
sparsity = lora_layer.get_sparsity()
print(f"✓ Merged weights, sparsity: {sparsity:.2%}")
EOF

# Test 5: Validation
echo -e "\n[5/5] Testing sparsity validation..."
python3 << 'EOF'
import sys
import torch
import torch.nn as nn
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))

from pruning.mask_ops import validate_mask

# Create mask and test validation
mask = torch.zeros(128, 256)
mask[:64, :] = 1  # 50% sparsity

is_valid, stats = validate_mask(mask, expected_sparsity=0.5, tolerance=0.01)

if is_valid:
    print(f"✓ Validation passed")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    print(f"  Zeros: {stats['num_zeros']}, Ones: {stats['num_ones']}")
else:
    print(f"✗ Validation failed: {stats.get('error', 'Unknown error')}")
EOF

echo -e "\n========================================"
echo "✓ All tests passed!"
echo "========================================"
echo ""
echo "Next steps:"
echo "  1. Compute masks: bash scripts/cs3/run_sparse_experiments.sh masks"
echo "  2. Train model: bash scripts/cs3/run_sparse_experiments.sh train"
echo "  3. Full sweep: bash scripts/cs3/run_sparse_experiments.sh sweep"
