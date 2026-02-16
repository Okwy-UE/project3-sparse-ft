from __future__ import annotations
from typing import Dict, Tuple, Optional
import torch

def apply_mask_to_state_dict(
    state_dict: Dict[str, torch.Tensor],
    masks: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in state_dict.items():
        if k in masks:
            m = masks[k].to(v.device, dtype=v.dtype)
            out[k] = v * m
        else:
            out[k] = v
    return out

def count_zeros(t: torch.Tensor) -> Tuple[int, int]:
    numel = t.numel()
    zeros = int((t == 0).sum().item())
    return zeros, numel
