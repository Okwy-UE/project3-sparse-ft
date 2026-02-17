"""
Pruning and sparsity modules for sparse-to-dense fine-tuning.

Supports:
- Unstructured pruning (magnitude-based, gradient-based)
- Structured pruning (channel/head pruning)
- Random pruning
- Mask computation, storage, and application
"""

from .mask_generator import (
    compute_unstructured_mask,
    compute_structured_mask,
    compute_random_mask,
    MaskGenerator
)
from .mask_ops import (
    apply_mask,
    validate_mask,
    compute_sparsity_stats,
    save_mask,
    load_mask
)
from .importance import (
    magnitude_importance,
    gradient_importance,
    taylor_importance
)
from .sparse_lora import (
    SparseLoRA,
    SparseLoRAConfig,
    apply_sparse_lora
)

__all__ = [
    'compute_unstructured_mask',
    'compute_structured_mask',
    'compute_random_mask',
    'MaskGenerator',
    'apply_mask',
    'validate_mask',
    'compute_sparsity_stats',
    'save_mask',
    'load_mask',
    'magnitude_importance',
    'gradient_importance',
    'taylor_importance',
    'SparseLoRA',
    'SparseLoRAConfig',
    'apply_sparse_lora',
]
