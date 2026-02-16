from __future__ import annotations
from typing import Dict, Tuple, Iterable, Optional, Set
import torch
import torch.nn as nn

LINEAR_LIKE = (nn.Linear,)

def iter_named_linear_modules(model: nn.Module) -> Iterable[Tuple[str, nn.Module]]:
    for name, mod in model.named_modules():
        if isinstance(mod, LINEAR_LIKE):
            yield name, mod

@torch.no_grad()
def collect_linear_input_l2(
    model: nn.Module,
    dataloader,
    device: torch.device,
    max_batches: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Returns:
      in_l2[name] = mean L2 norm per input channel for that Linear, shape [in_features]
    """
    model.eval()
    model.to(device)

    sums: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    hooks = []

    def make_hook(name: str):
        def hook(mod: nn.Module, inp, out):
            # inp is a tuple; Linear gets (x,)
            x = inp[0]
            # x: [B, T, in] or [B, in]
            if x.dim() == 2:
                x2 = x
            else:
                x2 = x.reshape(-1, x.shape[-1])
            l2 = torch.sqrt(torch.clamp((x2 * x2).mean(dim=0), min=1e-12))  # [in]
            if name not in sums:
                sums[name] = l2.detach().clone()
                counts[name] = 1
            else:
                sums[name] += l2.detach()
                counts[name] += 1
        return hook

    for name, mod in iter_named_linear_modules(model):
        hooks.append(mod.register_forward_hook(make_hook(name)))

    seen = 0
    for batch in dataloader:
        seen += 1
        if seen > max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        _ = model(**batch)

    for h in hooks:
        h.remove()

    in_l2 = {k: (sums[k] / counts[k]).cpu() for k in sums.keys()}
    return in_l2

def importance_unstructured_task(
    weight: torch.Tensor,  # [out, in] for Linear
    in_l2: Optional[torch.Tensor],  # [in]
) -> torch.Tensor:
    """
    Task-aware importance proxy:
      score[o,i] = |W[o,i]| * (in_l2[i] if available else 1)
    """
    w = weight.detach().abs()
    if in_l2 is None:
        return w
    scale = in_l2.to(w.device).view(1, -1)
    return w * scale
