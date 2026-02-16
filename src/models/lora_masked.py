from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Dict, Tuple

import torch


@dataclass
class MergeReport:
    param: str
    numel: int
    zeros_mask: int
    zeros_merged: int
    preserved: bool


def mask_checksum(mask: torch.Tensor) -> str:
    m = mask.detach().to(torch.uint8).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(m).hexdigest()


@torch.no_grad()
def merge_lora_with_mask(
    base_w: torch.Tensor,
    mask: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    *,
    alpha: float,
    r: int,
) -> torch.Tensor:
    """
    Merge: W' = (W ⊙ M) + ((B@A) ⊙ M) * (alpha/r)
    """
    maskb = mask.to(torch.bool) if mask.dtype != torch.bool else mask
    scale = alpha / float(r)
    delta = (B @ A) * scale
    return (base_w * maskb) + (delta * maskb)


@torch.no_grad()
def sparsity_preserved_check(
    merged: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[bool, int, int]:
    """
    True iff merged == 0 wherever mask == 0.
    """
    maskb = mask.to(torch.bool) if mask.dtype != torch.bool else mask
    mask0 = ~maskb
    zeros_mask = int(mask0.sum().item())
    zeros_merged = int((merged[mask0] == 0).sum().item())
    preserved = zeros_merged == zeros_mask
    return preserved, zeros_mask, zeros_merged


@torch.no_grad()
def run_unit_preservation_suite(
    *,
    d_out: int = 64,
    d_in: int = 128,
    r: int = 8,
    sparsity: float = 0.5,
    seed: int = 123,
) -> Dict[str, object]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)

    W = torch.randn((d_out, d_in), generator=g)
    A = torch.randn((r, d_in), generator=g)
    B = torch.randn((d_out, r), generator=g)

    u = torch.rand_like(W, generator=g)
    k = int(round((1.0 - sparsity) * u.numel()))
    k = max(0, min(u.numel(), k))
    flat = u.flatten()
    thr = torch.kthvalue(flat, flat.numel() - k + 1).values if k > 0 else torch.tensor(float("inf"))
    M = (u >= thr).to(torch.bool)

    merged = merge_lora_with_mask(W, M, A, B, alpha=16.0, r=r)
    preserved, zeros_mask, zeros_merged = sparsity_preserved_check(merged, M)

    return {
        "preserved": preserved,
        "zeros_mask": zeros_mask,
        "zeros_merged_on_mask0": zeros_merged,
        "mask_checksum": mask_checksum(M),
    }
