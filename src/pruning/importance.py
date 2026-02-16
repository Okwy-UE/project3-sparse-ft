from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import torch

from .mask_utils import build_param_filter


@dataclass
class ImportanceConfig:
    importance: str  # magnitude | wanda_like
    include_regex: str = r".*"
    exclude_regex: str = r"(?:^$)"
    only_linear_weights: bool = True
    block_size: Optional[Tuple[int, int]] = None  # for structured_block (rows, cols)
    per_layer: bool = True  # if False, global threshold across all selected tensors


@torch.no_grad()
def _collect_linear_inputs_rms(
    model: torch.nn.Module,
    dataloader: Iterable[dict],
    *,
    device: torch.device,
    max_batches: int,
) -> Dict[str, torch.Tensor]:
    """
    Collect sqrt(E[x^2]) per input-channel for Linear layers using forward hooks.
    Returns dict[param_name] -> col_rms (shape [in_features]).
    """
    stats: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    mod_to_pname: Dict[torch.nn.Module, str] = {}
    named_modules = dict(model.named_modules())
    for pname, p in model.named_parameters():
        if not pname.endswith(".weight"):
            continue
        mod_name = pname[:-7]
        m = named_modules.get(mod_name, None)
        if isinstance(m, torch.nn.Linear):
            mod_to_pname[m] = pname

    hooks = []

    def hook_fn(mod: torch.nn.Module, inp, out):
        x = inp[0].detach()
        if x.dim() == 3:
            x = x.reshape(-1, x.size(-1))
        elif x.dim() != 2:
            return
        s = (x.float() ** 2).sum(dim=0)
        pname = mod_to_pname.get(mod)
        if pname is None:
            return
        if pname not in stats:
            stats[pname] = s
            counts[pname] = x.size(0)
        else:
            stats[pname] += s
            counts[pname] += x.size(0)

    for m in mod_to_pname.keys():
        hooks.append(m.register_forward_hook(hook_fn))

    model.eval()
    model.to(device)

    for bi, batch in enumerate(dataloader):
        if bi >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
        _ = model(**batch)

    for h in hooks:
        h.remove()

    out: Dict[str, torch.Tensor] = {}
    for pname, s in stats.items():
        c = max(1, counts[pname])
        out[pname] = torch.sqrt(s / float(c))
    return out


def compute_importance_scores(
    model: torch.nn.Module,
    *,
    dataloader: Optional[Iterable[dict]] = None,
    device: Optional[torch.device] = None,
    max_batches: int = 8,
    cfg: Optional[ImportanceConfig] = None,
) -> Dict[str, torch.Tensor]:
    """
    Returns dict[param_name] -> importance tensor (same shape as param).
    """
    if cfg is None:
        cfg = ImportanceConfig(importance="magnitude")
    include, exclude = build_param_filter(cfg.include_regex, cfg.exclude_regex)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    col_rms: Dict[str, torch.Tensor] = {}
    if cfg.importance == "wanda_like":
        if dataloader is None:
            raise ValueError("wanda_like importance requires dataloader for calibration")
        col_rms = _collect_linear_inputs_rms(model, dataloader, device=device, max_batches=max_batches)

    scores: Dict[str, torch.Tensor] = {}
    for pname, p in model.named_parameters():
        if not include.search(pname) or exclude.search(pname):
            continue
        if cfg.only_linear_weights and not pname.endswith(".weight"):
            continue

        s = p.detach().abs().float().cpu()
        if cfg.importance == "wanda_like":
            rms = col_rms.get(pname, None)
            if rms is not None and s.dim() == 2 and rms.numel() == s.size(1):
                s = s * rms.view(1, -1).cpu()
        scores[pname] = s
    return scores


def scores_to_mask_unstructured(
    scores: Dict[str, torch.Tensor],
    *,
    sparsity: float,
    per_layer: bool = True,
) -> Dict[str, torch.Tensor]:
    masks: Dict[str, torch.Tensor] = {}
    if per_layer:
        for name, s in scores.items():
            flat = s.flatten()
            k = int(round((1.0 - sparsity) * flat.numel()))
            k = max(0, min(flat.numel(), k))
            if k == 0:
                thr = torch.tensor(float("inf"))
            else:
                thr = torch.kthvalue(flat, flat.numel() - k + 1).values
            masks[name] = (s >= thr).to(torch.uint8)
    else:
        all_flat = torch.cat([v.flatten() for v in scores.values()], dim=0)
        k = int(round((1.0 - sparsity) * all_flat.numel()))
        k = max(0, min(all_flat.numel(), k))
        if k == 0:
            thr = torch.tensor(float("inf"))
        else:
            thr = torch.kthvalue(all_flat, all_flat.numel() - k + 1).values
        for name, s in scores.items():
            masks[name] = (s >= thr).to(torch.uint8)
    return masks


def scores_to_mask_random(
    scores: Dict[str, torch.Tensor],
    *,
    sparsity: float,
    seed: int,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    masks: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        u = torch.rand_like(s, generator=g)
        k = int(round((1.0 - sparsity) * u.numel()))
        k = max(0, min(u.numel(), k))
        flat = u.flatten()
        if k == 0:
            thr = torch.tensor(float("inf"))
        else:
            thr = torch.kthvalue(flat, flat.numel() - k + 1).values
        masks[name] = (u >= thr).to(torch.uint8)
    return masks


def scores_to_mask_structured_block(
    scores: Dict[str, torch.Tensor],
    *,
    sparsity: float,
    block_size: Tuple[int, int] = (32, 32),
) -> Dict[str, torch.Tensor]:
    br, bc = block_size
    masks: Dict[str, torch.Tensor] = {}
    for name, s in scores.items():
        if s.dim() != 2:
            masks[name] = scores_to_mask_unstructured(
                {name: s}, sparsity=sparsity, per_layer=True
            )[name]
            continue

        r, c = s.shape
        rb = math.ceil(r / br)
        cb = math.ceil(c / bc)

        block_scores = torch.zeros((rb, cb), dtype=torch.float32)
        for i in range(rb):
            for j in range(cb):
                rs, re = i * br, min(r, (i + 1) * br)
                cs, ce = j * bc, min(c, (j + 1) * bc)
                block_scores[i, j] = s[rs:re, cs:ce].sum()

        flat = block_scores.flatten()
        k = int(round((1.0 - sparsity) * flat.numel()))
        k = max(0, min(flat.numel(), k))
        if k == 0:
            thr = torch.tensor(float("inf"))
        else:
            thr = torch.kthvalue(flat, flat.numel() - k + 1).values

        keep_blocks = block_scores >= thr
        m = torch.zeros_like(s, dtype=torch.uint8)
        for i in range(rb):
            for j in range(cb):
                if not keep_blocks[i, j]:
                    continue
                rs, re = i * br, min(r, (i + 1) * br)
                cs, ce = j * bc, min(c, (j + 1) * bc)
                m[rs:re, cs:ce] = 1
        masks[name] = m
    return masks
