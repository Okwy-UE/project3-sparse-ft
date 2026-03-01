"""
Wrapper for sparse models compatible with Cerebras training.

Provides:
  - SparseModelWrapper  : wraps a base model + masks, applies masks on forward
  - prepare_sparse_model: high-level helper for the three modes (inference,
    sparse_to_dense, sparse_to_sparse)
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.mask_ops import apply_mask, load_mask
from pruning.sparse_lora import (
    SparseLoRAConfig,
    apply_sparse_lora,
    merge_all_lora_weights,
    validate_sparsity_preserved,
)


class SparseModelWrapper(nn.Module):
    """
    Wraps a base model with sparsity masks.

    On each forward pass (if ``apply_on_forward`` is True) the masks are
    re-applied so that the masked weights stay exactly zero.  This is the
    correct behaviour for sparse-to-sparse training.

    For sparse-to-dense the base weights are frozen and sparse already;
    no per-forward re-masking is necessary, so set ``apply_on_forward=False``.
    """

    def __init__(
        self,
        base_model: nn.Module,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        mask_path: Optional[str] = None,
        apply_on_forward: bool = True,
    ):
        super().__init__()
        self.base_model = base_model
        self.apply_on_forward = apply_on_forward

        if masks is None and mask_path is not None:
            masks, _ = load_mask(mask_path)

        self.masks = masks or {}
        self._register_masks()

        if self.masks:
            self.apply_masks()

    def _register_masks(self):
        for name, mask in self.masks.items():
            buf_name = f"mask_{name.replace('.', '_')}"
            self.register_buffer(buf_name, mask)

    def apply_masks(self):
        for name, module in self.base_model.named_modules():
            if name in self.masks:
                if hasattr(module, "weight") and module.weight is not None:
                    m = self.masks[name].to(module.weight.device)
                    module.weight.data = apply_mask(module.weight.data, m)

    def forward(self, *args, **kwargs):
        if self.apply_on_forward and self.masks:
            self.apply_masks()
        return self.base_model(*args, **kwargs)

    def get_sparsity_stats(self) -> Dict:
        from pruning.mask_ops import compute_sparsity_stats
        return compute_sparsity_stats(self.masks, self.base_model)


def prepare_sparse_model(
    model: nn.Module,
    mask_path: str,
    lora_config: Optional[SparseLoRAConfig] = None,
    mode: Literal["inference", "sparse_to_dense", "sparse_to_sparse"] = "sparse_to_dense",
) -> nn.Module:
    """
    Prepare a model for sparse (LoRA) fine-tuning.

    Args:
        model:      Base model.
        mask_path:  Path to the ``.pt`` mask file.
        lora_config: Sparse LoRA config (if None, uses defaults).
        mode:
            ``inference``       – just mask the weights, no training.
            ``sparse_to_dense`` – mask weights + apply dense LoRA.
            ``sparse_to_sparse``– mask weights + apply masked LoRA (delta ⊙ M).

    Returns:
        Prepared model (may be a SparseModelWrapper or LoRA-wrapped).
    """
    masks, metadata = load_mask(mask_path)
    print(f"[prepare_sparse_model] Loaded {len(masks)} masks from {mask_path}")
    sparsity = metadata.get("sparsity", metadata.get("stats", {}).get("global", {}).get("sparsity", "?"))
    print(f"[prepare_sparse_model] Global sparsity: {sparsity}")

    if mode == "inference":
        return SparseModelWrapper(model, masks=masks, apply_on_forward=True)

    if mode not in ("sparse_to_dense", "sparse_to_sparse"):
        raise ValueError(f"Unknown mode: {mode}")

    if lora_config is None:
        lora_config = SparseLoRAConfig()

    lora_config.sparsity_mode = mode
    lora_config.maintain_sparsity = mode == "sparse_to_sparse"
    lora_config.masked_lora_update = mode == "sparse_to_sparse"

    for name, module in model.named_modules():
        if name in masks:
            if hasattr(module, "weight") and module.weight is not None:
                m = masks[name].to(module.weight.device)
                module.weight.data = apply_mask(module.weight.data, m)

    model = apply_sparse_lora(model, masks, lora_config)
    print(f"[prepare_sparse_model] Applied Sparse LoRA ({mode})")
    return model
