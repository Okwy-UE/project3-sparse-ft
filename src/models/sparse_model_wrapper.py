"""
Wrapper for sparse models compatible with Cerebras training.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Literal
from pathlib import Path
import sys

# Add parent to path to import pruning modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.mask_ops import apply_mask, load_mask
from pruning.sparse_lora import (
    SparseLoRAConfig,
    apply_sparse_lora,
    merge_all_lora_weights,
    validate_sparsity_preserved
)


class SparseModelWrapper(nn.Module):
    """
    Wrapper that applies sparsity masks to a model.
    
    Compatible with Cerebras training pipeline.
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        masks: Optional[Dict[str, torch.Tensor]] = None,
        mask_path: Optional[str] = None,
        apply_on_forward: bool = True,
    ):
        """
        Initialize sparse model wrapper.
        
        Args:
            base_model: Base model to wrap
            masks: Dictionary of sparsity masks
            mask_path: Path to load masks from (if masks not provided)
            apply_on_forward: If True, apply masks on every forward pass
        """
        super().__init__()
        
        self.base_model = base_model
        self.apply_on_forward = apply_on_forward
        
        # Load or set masks
        if masks is None and mask_path is not None:
            masks, _ = load_mask(mask_path)
        
        self.masks = masks or {}
        
        # Register masks as buffers
        self._register_masks()
        
        # Apply masks initially
        if self.masks:
            self.apply_masks()
    
    def _register_masks(self):
        """Register masks as buffers so they move with the model."""
        for name, mask in self.masks.items():
            # Replace dots with underscores for buffer names
            buffer_name = f"mask_{name.replace('.', '_')}"
            self.register_buffer(buffer_name, mask)
    
    def apply_masks(self):
        """Apply masks to model weights."""
        for name, module in self.base_model.named_modules():
            if name in self.masks:
                if hasattr(module, 'weight') and module.weight is not None:
                    mask = self.masks[name].to(module.weight.device)
                    module.weight.data = apply_mask(module.weight.data, mask)
    
    def forward(self, *args, **kwargs):
        """Forward pass with optional mask application."""
        if self.apply_on_forward and self.masks:
            self.apply_masks()
        
        return self.base_model(*args, **kwargs)
    
    def get_sparsity_stats(self) -> Dict:
        """Get current sparsity statistics."""
        from pruning.mask_ops import compute_sparsity_stats
        return compute_sparsity_stats(self.masks, self.base_model)


def prepare_sparse_model(
    model: nn.Module,
    mask_path: str,
    lora_config: Optional[SparseLoRAConfig] = None,
    mode: Literal["inference", "sparse_to_dense", "sparse_to_sparse"] = "sparse_to_dense",
) -> nn.Module:
    """
    Prepare a model for sparse fine-tuning.
    
    Args:
        model: Base model
        mask_path: Path to sparsity masks
        lora_config: Configuration for Sparse LoRA
        mode: Training mode
            - "inference": Just apply masks, no training
            - "sparse_to_dense": Start sparse, train dense LoRA
            - "sparse_to_sparse": Maintain sparsity during training
    
    Returns:
        Prepared model
    """
    # Load masks
    masks, metadata = load_mask(mask_path)
    print(f"Loaded masks from {mask_path}")
    print(f"Mask metadata: {metadata}")
    
    if mode == "inference":
        # Just wrap with masks
        model = SparseModelWrapper(model, masks=masks, apply_on_forward=True)
    
    elif mode in ["sparse_to_dense", "sparse_to_sparse"]:
        # Apply Sparse LoRA
        if lora_config is None:
            lora_config = SparseLoRAConfig()
        
        lora_config.sparsity_mode = mode
        lora_config.maintain_sparsity = (mode == "sparse_to_sparse")
        
        # Apply masks first
        for name, module in model.named_modules():
            if name in masks:
                if hasattr(module, 'weight') and module.weight is not None:
                    mask = masks[name].to(module.weight.device)
                    module.weight.data = apply_mask(module.weight.data, mask)
        
        # Apply LoRA
        model = apply_sparse_lora(model, masks, lora_config)
        print(f"Applied Sparse LoRA with mode: {mode}")
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return model
