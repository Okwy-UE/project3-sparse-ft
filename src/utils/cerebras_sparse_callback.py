"""
Cerebras callback for sparse training.

Integrates sparsity masks with Cerebras ModelZoo training pipeline.
"""

import torch
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from pruning.mask_ops import load_mask, apply_mask


class SparseMaskCallback:
    """
    Callback to apply sparsity masks during Cerebras training.
    
    This callback ensures masks are applied:
    - After model initialization
    - After each training step (optional, for sparse-to-sparse training)
    - Before checkpoint saving
    """
    
    def __init__(
        self,
        mask_path: str,
        apply_every_step: bool = False,
        device: str = "cpu",
    ):
        """
        Initialize sparse mask callback.
        
        Args:
            mask_path: Path to mask file
            apply_every_step: If True, apply masks after each training step
            device: Device for masks
        """
        self.mask_path = mask_path
        self.apply_every_step = apply_every_step
        self.device = device
        
        # Load masks
        self.masks, self.metadata = load_mask(mask_path)
        print(f"[SparseMaskCallback] Loaded masks from {mask_path}")
        print(f"[SparseMaskCallback] Sparsity: {self.metadata.get('sparsity', 'unknown')}")
        print(f"[SparseMaskCallback] Method: {self.metadata.get('method', 'unknown')}")
    
    def on_train_start(self, trainer, model):
        """Apply masks when training starts."""
        print("[SparseMaskCallback] Applying masks at training start")
        self._apply_masks(model)
    
    def on_train_step_end(self, trainer, model, outputs):
        """Optionally apply masks after each training step."""
        if self.apply_every_step:
            self._apply_masks(model)
    
    def on_save_checkpoint(self, trainer, model, checkpoint_path):
        """Apply masks before saving checkpoint."""
        print(f"[SparseMaskCallback] Applying masks before saving to {checkpoint_path}")
        self._apply_masks(model)
    
    def _apply_masks(self, model):
        """Apply masks to model weights."""
        for name, module in model.named_modules():
            if name in self.masks:
                if hasattr(module, 'weight') and module.weight is not None:
                    mask = self.masks[name].to(module.weight.device)
                    module.weight.data = apply_mask(module.weight.data, mask)


def create_sparse_config(
    base_config: dict,
    mask_path: str,
    lora_config: Optional[dict] = None,
    sparse_mode: str = "sparse_to_dense",
) -> dict:
    """
    Create a Cerebras config with sparse training settings.
    
    Args:
        base_config: Base config dictionary
        mask_path: Path to sparsity masks
        lora_config: LoRA configuration
        sparse_mode: Training mode ("sparse_to_dense", "sparse_to_sparse")
    
    Returns:
        Modified config dictionary
    """
    config = base_config.copy()
    
    # Add sparse mask callback
    if 'trainer' not in config:
        config['trainer'] = {}
    if 'init' not in config['trainer']:
        config['trainer']['init'] = {}
    if 'callbacks' not in config['trainer']['init']:
        config['trainer']['init']['callbacks'] = []
    
    # Add sparse mask callback
    sparse_callback = {
        'SparseMask': {
            'mask_path': mask_path,
            'apply_every_step': (sparse_mode == "sparse_to_sparse"),
        }
    }
    config['trainer']['init']['callbacks'].append(sparse_callback)
    
    # Add or update LoRA config
    if lora_config is not None:
        lora_callback = {'Lora': {'lora_params': lora_config}}
        
        # Check if LoRA already exists
        has_lora = False
        for cb in config['trainer']['init']['callbacks']:
            if 'Lora' in cb:
                cb['Lora']['lora_params'].update(lora_config)
                has_lora = True
                break
        
        if not has_lora:
            config['trainer']['init']['callbacks'].append(lora_callback)
    
    return config
