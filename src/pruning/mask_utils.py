from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import torch


@dataclass(frozen=True)
class MaskSpec:
    model_id: str
    task: str
    sparsity: float  # e.g., 0.5 for 50%
    sparsity_type: str  # unstructured | structured_block | random
    importance: str  # magnitude | wanda_like
    seed: int
    git_sha: str = "nogit"
    include_regex: str = r".*"
    exclude_regex: str = r"(?:^$)"  # exclude nothing by default
    module_types: Tuple[str, ...] = ("Linear",)  # default to Linear weights
    created_at: Optional[str] = None
    notes: str = ""


def _ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)


def _mask_checksum(mask: torch.Tensor) -> str:
    # Stable checksum of mask bytes (bool/uint8)
    m = mask.detach().to(torch.uint8).contiguous().cpu().numpy().tobytes()
    return hashlib.sha256(m).hexdigest()


def save_mask(path: str | Path, masks: Dict[str, torch.Tensor], spec: MaskSpec) -> None:
    path = Path(path)
    _ensure_dir(path)
    meta = dataclasses.asdict(spec)
    # Add per-tensor checksum map for debugging / correctness memo
    meta["checksums"] = {k: _mask_checksum(v) for k, v in masks.items()}
    torch.save({"masks": masks, "meta": meta}, path)


def load_mask(path: str | Path) -> Tuple[Dict[str, torch.Tensor], dict]:
    obj = torch.load(Path(path), map_location="cpu")
    return obj["masks"], obj.get("meta", {})


def mask_sparsity_report(masks: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    rep: Dict[str, dict] = {}
    for name, m in masks.items():
        mm = m.detach()
        if mm.dtype != torch.bool:
            mm = mm.to(torch.bool)
        nnz = int(mm.sum().item())
        numel = int(mm.numel())
        rep[name] = {
            "numel": numel,
            "nnz": nnz,
            "sparsity": 1.0 - (nnz / max(1, numel)),
        }
    return rep


def _name_ok(name: str, include: re.Pattern, exclude: re.Pattern) -> bool:
    return bool(include.search(name)) and not bool(exclude.search(name))


def apply_masks_inplace(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
    device: Optional[torch.device] = None,
) -> Dict[str, str]:
    """
    Applies masks to matching parameters: p <- p * mask.
    Returns a dict param_name -> status ("applied"|"missing"|"shape_mismatch").
    """
    status: Dict[str, str] = {}
    named_params = dict(model.named_parameters())
    for pname, mask in masks.items():
        if pname not in named_params:
            status[pname] = "missing"
            if strict:
                raise KeyError(f"Mask param {pname} not found in model.named_parameters()")
            continue
        p = named_params[pname]
        if p.shape != mask.shape:
            status[pname] = "shape_mismatch"
            if strict:
                raise ValueError(
                    f"Shape mismatch for {pname}: param={tuple(p.shape)} mask={tuple(mask.shape)}"
                )
            continue
        m = mask.to(device=p.device if device is None else device)
        if m.dtype != torch.bool:
            m = m.to(torch.bool)
        with torch.no_grad():
            p.mul_(m)
        status[pname] = "applied"
    return status


def enforce_masks_after_step(
    model: torch.nn.Module,
    masks: Dict[str, torch.Tensor],
    *,
    strict: bool = False,
) -> None:
    """
    Call after optimizer.step() to prevent regrowth (sparse-to-sparse).
    """
    _ = apply_masks_inplace(model, masks, strict=strict)


def build_param_filter(include_regex: str, exclude_regex: str) -> Tuple[re.Pattern, re.Pattern]:
    return re.compile(include_regex), re.compile(exclude_regex)


def default_exclude_for_llms() -> str:
    # Avoid masking norms, embeddings, and biases by default
    # (You can override via CLI)
    return r"(?:\.bias$)|(?:embed)|(?:norm)|(?:layernorm)|(?:ln_)"
