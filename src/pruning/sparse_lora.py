"""
Sparse LoRA implementation.

Two training modes:
  sparse_to_dense  – sparse base + dense LoRA update  (Phoenix baseline)
  sparse_to_sparse – sparse base + masked LoRA update  (B@A) ⊙ M

Forward:
  dense  :  output = W_base·x + scaling·(B @ A)·x
  masked :  output = W_base·x + scaling·((B @ A) ⊙ M)·x

Merge:
  W_merged = (W_base + scaling·B @ A) ⊙ M    [preserve_sparsity=True]
  W_merged =  W_base + scaling·B @ A          [preserve_sparsity=False]
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Literal, Tuple
from dataclasses import dataclass, field


@dataclass
class SparseLoRAConfig:
    """Configuration for Sparse LoRA."""

    r: int = 16
    alpha: float = 32.0
    dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: ["Linear"])

    sparsity_mode: Literal["none", "sparse_to_dense", "sparse_to_sparse"] = "none"
    initial_sparsity: float = 0.5
    maintain_sparsity: bool = False
    masked_lora_update: bool = False

    merge_at_end: bool = True
    preserve_sparsity_on_merge: bool = True


class SparseLoRALayer(nn.Module):
    """
    Drop-in replacement for nn.Linear that adds a low-rank adapter
    on top of a frozen (optionally sparse) base weight.
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

        for param in self.base_layer.parameters():
            param.requires_grad = False

        if mask is not None:
            self.register_buffer("mask", mask.clone())
            with torch.no_grad():
                if hasattr(self.base_layer, "weight"):
                    self.base_layer.weight.data *= self.mask
        else:
            self.mask = None

        if not isinstance(base_layer, nn.Linear):
            raise NotImplementedError(
                f"SparseLoRA not implemented for {type(base_layer)}"
            )

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(config.r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, config.r))

        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

        self.scaling = config.alpha / config.r
        self.dropout = (
            nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        )
        self.merged = False

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def _compute_lora_delta(self) -> torch.Tensor:
        """Return the LoRA weight delta, optionally masked."""
        delta = (self.lora_B @ self.lora_A) * self.scaling
        if self.config.masked_lora_update and self.mask is not None:
            delta = delta * self.mask
        return delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base_layer(x)

        if self.config.masked_lora_update and self.mask is not None:
            delta = self._compute_lora_delta()
            lora_out = self.dropout(x) @ delta.T
        else:
            lora_out = (
                self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling
            )

        return base_out + lora_out

    # ------------------------------------------------------------------
    # Merge / un-merge
    # ------------------------------------------------------------------
    def merge_weights(self, preserve_sparsity: bool = True) -> None:
        if self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        with torch.no_grad():
            self.base_layer.weight.data += delta
            if preserve_sparsity and self.mask is not None:
                self.base_layer.weight.data *= self.mask
        self.merged = True

    def unmerge_weights(self) -> None:
        if not self.merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        with torch.no_grad():
            self.base_layer.weight.data -= delta
        self.merged = False

    def get_sparsity(self) -> float:
        if hasattr(self.base_layer, "weight"):
            w = self.base_layer.weight
            return (w == 0).sum().item() / w.numel()
        return 0.0


# ======================================================================
# Model-level helpers
# ======================================================================

_TARGET_TYPE_MAP = {
    "Linear": nn.Linear,
    # "Conv1d": nn.Conv1d,
    # "Conv2d": nn.Conv2d,
}


def _resolve_target_types(names: List[str]) -> Tuple[type, ...]:
    types = []
    for n in names:
        if n in _TARGET_TYPE_MAP:
            types.append(_TARGET_TYPE_MAP[n])
        else:
            print(f"Warning: Unknown target module type '{n}'")
    return tuple(types)


def apply_sparse_lora(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    config: SparseLoRAConfig,
    target_modules: Optional[List[str]] = None,
) -> nn.Module:
    """
    Replace target modules in *model* with SparseLoRALayer instances.

    Mask lookup order for a module at path ``parent.child``:
      1. ``parent.child``           (exact module path)
      2. any mask key containing ``child``

    Returns the mutated model.
    """
    target_types = _resolve_target_types(
        target_modules or config.target_modules or ["Linear"]
    )

    def _find_mask(full_name: str, child_name: str):
        if full_name in masks:
            return masks[full_name]
        for mask_key, mask_tensor in masks.items():
            if full_name.endswith(mask_key) or mask_key.endswith(full_name):
                return mask_tensor
        for mask_key, mask_tensor in masks.items():
            if child_name in mask_key:
                return mask_tensor
        return None

    def _recurse(module: nn.Module, prefix: str = ""):
        for child_name, child_module in module.named_children():
            full_name = f"{prefix}.{child_name}" if prefix else child_name
            if isinstance(child_module, target_types):
                mask = _find_mask(full_name, child_name)
                lora_layer = SparseLoRALayer(child_module, mask, config)
                setattr(module, child_name, lora_layer)
            else:
                _recurse(child_module, full_name)

    _recurse(model)
    return model


def merge_all_lora_weights(
    model: nn.Module,
    preserve_sparsity: bool = True,
) -> nn.Module:
    """Merge all SparseLoRALayer deltas back into their base layers."""
    for module in model.modules():
        if isinstance(module, SparseLoRALayer):
            module.merge_weights(preserve_sparsity=preserve_sparsity)
    return model


def validate_sparsity_preserved(
    model: nn.Module,
    original_masks: Optional[Dict[str, torch.Tensor]] = None,
    tolerance: float = 1e-6,
) -> Tuple[bool, dict]:
    """
    Check that every SparseLoRALayer's base weight is zero wherever
    the mask says it should be.

    Returns (all_valid, report_dict).
    """
    report: dict = {"per_layer": {}, "global": {"all_valid": True, "total_violations": 0}}

    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALayer):
            mask = module.mask
            if mask is None:
                continue
            weight = module.base_layer.weight.data
            zero_region = mask == 0
            violations = (torch.abs(weight[zero_region]) > tolerance).sum().item()
            total_masked = zero_region.sum().item()
            layer_report = {
                "num_violations": int(violations),
                "total_masked": int(total_masked),
                "violation_rate": violations / max(total_masked, 1),
                "max_violation": (
                    torch.abs(weight[zero_region]).max().item()
                    if zero_region.any()
                    else 0.0
                ),
            }
            report["per_layer"][name] = layer_report
            report["global"]["total_violations"] += int(violations)
            if violations > 0:
                report["global"]["all_valid"] = False

    return report["global"]["all_valid"], report


def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Extract only LoRA parameters from a model."""
    lora_sd: Dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALayer):
            lora_sd[f"{name}.lora_A"] = module.lora_A.data
            lora_sd[f"{name}.lora_B"] = module.lora_B.data
    return lora_sd


def load_lora_state_dict(
    model: nn.Module,
    lora_state_dict: Dict[str, torch.Tensor],
) -> nn.Module:
    """Load LoRA parameters into a model."""
    for name, module in model.named_modules():
        if isinstance(module, SparseLoRALayer):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"
            if a_key in lora_state_dict:
                module.lora_A.data.copy_(lora_state_dict[a_key])
            if b_key in lora_state_dict:
                module.lora_B.data.copy_(lora_state_dict[b_key])
    return model
