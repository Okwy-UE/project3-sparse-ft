"""
Sparse LoRA implementation.

Implements LoRA (Low-Rank Adaptation) with sparsity constraints:
- Sparse-to-dense: Start sparse, fine-tune densely
- Sparse-to-sparse: Maintain sparsity during fine-tuning
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Literal
from dataclasses import dataclass
import copy


@dataclass
class SparseLoRAConfig:
    """Configuration for Sparse LoRA."""
    
    # LoRA hyperparameters
    r: int = 16  # Rank of LoRA matrices
    alpha: float = 32.0  # Scaling factor
    dropout: float = 0.05  # Dropout probability
    target_modules: List[str] = None  # Which modules to apply LoRA to
    
    # Sparsity settings
    sparsity_mode: Literal["none", "sparse_to_dense", "sparse_to_sparse"] = "none"
    initial_sparsity: float = 0.5  # Initial sparsity level
    maintain_sparsity: bool = False  # Whether to enforce sparsity during training
    
    # Merge settings
    merge_at_end: bool = True  # Whether to merge LoRA weights back
    preserve_sparsity_on_merge: bool = True  # Whether to apply mask after merge
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["Linear"]


class SparseLoRALayer(nn.Module):
    """
    Sparse LoRA layer: combines a frozen sparse base layer with trainable LoRA adapters.
    
    Forward: output = (W_base ⊙ M) @ x + (B @ A) @ x
    Merge: W_merged = (W_base + B @ A) ⊙ M
    
    Where:
    - W_base: frozen base weights
    - M: sparsity mask
    - A, B: trainable LoRA matrices
    """
    
    def __init__(
        self,
        base_layer: nn.Module,
        mask: Optional[torch.Tensor],
        config: SparseLoRAConfig,
    ):
        super().__init__()
        
        self.config = config
        self.base_layer = base_layer
        
        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
        
        # Apply mask to base layer if provided
        if mask is not None:
            self.register_buffer('mask', mask)
            with torch.no_grad():
                if hasattr(self.base_layer, 'weight'):
                    self.base_layer.weight.data *= self.mask
        else:
            self.mask = None
        
        # Initialize LoRA matrices
        if isinstance(base_layer, nn.Linear):
            in_features = base_layer.in_features
            out_features = base_layer.out_features
            
            # LoRA: W + B @ A
            # A: (in_features, r)
            # B: (r, out_features) transposed -> (out_features, r)
            self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))
            
            # Initialize A with Kaiming, B with zeros (standard LoRA init)
            nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
            nn.init.zeros_(self.lora_B)
            
            self.scaling = config.alpha / config.r
        else:
            raise NotImplementedError(f"SparseLoRA not implemented for {type(base_layer)}")
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        
        self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: base output + LoRA output.
        """
        # Base layer output (already masked)
        base_out = self.base_layer(x)
        
        # LoRA output: (B @ A) @ x
        lora_out = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        lora_out = self.dropout(lora_out)
        
        return base_out + lora_out
    
    def merge_weights(self, preserve_sparsity: bool = True) -> None:
        """
        Merge LoRA weights back into base layer: W = (W_base + B @ A) ⊙ M
        
        Args:
            preserve_sparsity: If True, apply mask after merge
        """
        if self.merged:
            return
        
        if isinstance(self.base_layer, nn.Linear):
            # Compute LoRA delta: B @ A
            lora_delta = (self.lora_B @ self.lora_A) * self.scaling
            
            # Merge: W_new = W_base + delta
            with torch.no_grad():
                self.base_layer.weight.data += lora_delta
                
                # Re-apply mask if preserving sparsity
                if preserve_sparsity and self.mask is not None:
                    self.base_layer.weight.data *= self.mask
        
        self.merged = True
    
    def unmerge_weights(self) -> None:
        """
        Unmerge LoRA weights (subtract delta from base).
        """
        if not self.merged:
            return
        
        if isinstance(self.base_layer, nn.Linear):
            lora_delta = (self.lora_B @ self.lora_A) * self.scaling
            
            with torch.no_grad():
                self.base_layer.weight.data -= lora_delta
        
        self.merged = False
    
    def get_sparsity(self) -> float:
        """
        Compute current sparsity of base weights.
        """
        if hasattr(self.base_layer, 'weight'):
            weight = self.base_layer.weight
            num_zeros = (weight == 0).sum().item()
            total = weight.numel()
            return num_zeros / total
        return 0.0


def apply_sparse_lora(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    config: SparseLoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Apply Sparse LoRA to a model.
    
    Replaces target modules with SparseLoRALayer instances.
    
    Args:
        model: Base model
        masks: Dictionary of sparsity masks
        config: Sparse LoRA configuration
        target_modules: List of module types to apply LoRA to (e.g., ["Linear"])
    
    Returns:
        Model with Sparse LoRA layers
    """
    if target_modules is None:
        target_modules = config.target_modules or ["Linear"]
    
    # Convert target modules to types
    target_types = []
    for module_name in target_modules:
        if module_name == "Linear":
            target_types.append(nn.Linear)
        elif module_name == "Conv1d":
            target_types.append(nn.Conv1d)
        elif module_name == "Conv2d":
            target_types.append(nn.Conv2d)
        else:
            print(f"Warning: Unknown module type {module_name}")
    
    target_types = tuple(target_types)
    
    # Replace modules
    def _replace_module(parent_module, child_name, child_module):
        mask = masks.get(f"{parent_module.__class__.__name__}.{child_name}", None)
        
        # Try to find mask with different naming schemes
        if mask is None:
            for mask_key in masks.keys():
                if child_name in mask_key:
                    mask = masks[mask_key]
                    break
        
        sparse_lora_layer = SparseLoRALayer(child_module, mask, config)
        setattr(parent_module, child_name, sparse_lora_layer)
    
    # Recursively replace modules
    for name, module in model.named_children():
        if isinstance(module, target_types):
            _replace_module(model, name, module)
        else:
            # Recurse
            apply_sparse_lora(module, masks, config, target_modules)
    
    return model


def merge_all_lora_weights(
    model: nn.Module,
    preserve_sparsity: bool = True,
) -> nn.Module:
    """
    Merge all LoRA weights in a model.
    
    Args:
        model: Model with Sparse LoRA layers
        preserve_sparsity: Whether to preserve sparsity after merge
    
    Returns:
        Model with merged weights
    """
    for module in model.modules():
        if isinstance(module, SparseLoRALayer):
            module.merge_weights(preserve_sparsity=preserve_sparsity)
    
    return model


def validate_sparsity_preserved(
    model: nn.Module,
    original_masks: Dict[str, torch.Tensor],
    tolerance: float = 1e-6,
) -> Tuple[bool, Dict[str, any]]:
    """
    Validate that sparsity is preserved after LoRA merge.
    
    Args:
        model: Model after merge
        original_masks: Original sparsity masks
        tolerance: Numerical tolerance for zero check
    
    Returns:
        (is_valid, validation_report)
    """
    report = {
        'per_layer': {},
        'global': {'all_valid': True, 'total_violations': 0},
    }
    
    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALayer):
            mask = module.mask
            if mask is None:
                continue
            
            weight = module.base_layer.weight.data
            
            # Check: where mask is 0, weight should be 0
            violations = torch.abs(weight[mask == 0]) > tolerance
            num_violations = violations.sum().item()
            
            layer_report = {
                'num_violations': num_violations,
                'total_masked': (mask == 0).sum().item(),
                'violation_rate': num_violations / max((mask == 0).sum().item(), 1),
                'max_violation': torch.abs(weight[mask == 0]).max().item() if (mask == 0).any() else 0.0,
            }
            
            report['per_layer'][name] = layer_report
            report['global']['total_violations'] += num_violations
            
            if num_violations > 0:
                report['global']['all_valid'] = False
    
    return report['global']['all_valid'], report


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract only LoRA parameters from a model.
    
    Args:
        model: Model with Sparse LoRA layers
    
    Returns:
        State dict containing only LoRA parameters
    """
    lora_state_dict = {}
    
    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALayer):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data
    
    return lora_state_dict


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
) -> nn.Module:
    """
    Load LoRA parameters into a model.
    
    Args:
        model: Model with Sparse LoRA layers
        lora_state_dict: State dict of LoRA parameters
    
    Returns:
        Model with loaded LoRA parameters
    """
    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALayer):
            if f"{name}.lora_A" in lora_state_dict:
                module.lora_A.data = lora_state_dict[f"{name}.lora_A"]
            if f"{name}.lora_B" in lora_state_dict:
                module.lora_B.data = lora_state_dict[f"{name}.lora_B"]
    
    return model
