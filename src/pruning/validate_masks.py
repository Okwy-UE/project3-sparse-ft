from __future__ import annotations
from typing import Dict, Any, List, Tuple
import csv
import os
import torch

def mask_sparsity(mask: torch.Tensor) -> float:
    # sparsity = fraction of zeros
    return float((mask == 0).float().mean().item())

def validate_masks(
    masks: Dict[str, torch.Tensor],
    out_csv: str,
) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    rows: List[Tuple[str, float, int]] = []
    for name, m in masks.items():
        m = m.detach().cpu()
        sp = mask_sparsity(m)
        rows.append((name, sp, m.numel()))
    rows.sort(key=lambda x: x[0])

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["param_name", "sparsity", "numel"])
        for r in rows:
            w.writerow(list(r))

    # global weighted sparsity
    total = sum(r[2] for r in rows)
    weighted = sum(r[1] * r[2] for r in rows) / max(total, 1)
    return {"global_weighted_sparsity": float(weighted), "num_masked_params": len(rows)}
