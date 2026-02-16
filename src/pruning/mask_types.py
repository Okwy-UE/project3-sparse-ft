from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Literal, Any

MaskType = Literal["unstructured_task", "random", "structured_nm"]

@dataclass
class MaskSpec:
    model: str
    task: str
    mask_type: MaskType
    sparsity: float          # e.g. 0.5 means 50% zeros
    seed: int = 1337
    # structured params
    n: Optional[int] = None
    m: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model": self.model,
            "task": self.task,
            "mask_type": self.mask_type,
            "sparsity": float(self.sparsity),
            "seed": int(self.seed),
            "n": None if self.n is None else int(self.n),
            "m": None if self.m is None else int(self.m),
        }
