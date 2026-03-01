"""
Cerebras callback for sparse training.

For sparse-to-sparse mode, this callback re-applies the sparsity mask
after every training step so that the LoRA-updated base weights stay
exactly zero wherever the original mask says they should be.

For sparse-to-dense mode (the Phoenix default), no per-step masking is
needed because the base weights are frozen; only LoRA parameters train.
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

    Applies masks:
      - After model initialisation (on_train_start)
      - After every step if ``apply_every_step`` is True (sparse-to-sparse)
      - Before checkpoint saves
    """

    def __init__(
        self,
        mask_path: str,
        apply_every_step: bool = False,
        device: str = "cpu",
    ):
        self.mask_path = mask_path
        self.apply_every_step = apply_every_step
        self.device = device

        self.masks, self.metadata = load_mask(mask_path)
        self._key_cache: Dict[str, str] = {}

        print(f"[SparseMaskCallback] Loaded {len(self.masks)} masks "
              f"from {mask_path}")
        print(f"[SparseMaskCallback] Sparsity : "
              f"{self.metadata.get('sparsity', 'unknown')}")
        print(f"[SparseMaskCallback] Every-step: {apply_every_step}")

    def _build_key_cache(self, model):
        """Map mask keys to model module names (once)."""
        if self._key_cache:
            return
        model_names = {n for n, _ in model.named_modules()}
        for mk in self.masks:
            if mk in model_names:
                self._key_cache[mk] = mk
            else:
                for mn in model_names:
                    if mn.endswith(mk) or mk.endswith(mn):
                        self._key_cache[mk] = mn
                        break

    def on_train_start(self, trainer, model):
        print("[SparseMaskCallback] Applying masks at training start")
        self._build_key_cache(model)
        self._apply_masks(model)

    def on_train_step_end(self, trainer, model, outputs):
        if self.apply_every_step:
            self._apply_masks(model)

    def on_save_checkpoint(self, trainer, model, checkpoint_path):
        print(f"[SparseMaskCallback] Applying masks before save -> "
              f"{checkpoint_path}")
        self._apply_masks(model)

    def _apply_masks(self, model):
        for mk, mask_tensor in self.masks.items():
            module_name = self._key_cache.get(mk, mk)
            found = False
            for name, module in model.named_modules():
                if name == module_name:
                    if hasattr(module, "weight") and module.weight is not None:
                        m = mask_tensor.to(module.weight.device)
                        module.weight.data = apply_mask(module.weight.data, m)
                        found = True
                    break
            if not found:
                for name, module in model.named_modules():
                    if mk in name or name in mk:
                        if hasattr(module, "weight") and module.weight is not None:
                            m = mask_tensor.to(module.weight.device)
                            module.weight.data = apply_mask(module.weight.data, m)
                        break


def create_sparse_config(
    base_config: dict,
    mask_path: str,
    lora_config: Optional[dict] = None,
    sparse_mode: str = "sparse_to_dense",
) -> dict:
    """
    Inject sparse-mask callback (and optional LoRA callback) into a
    Cerebras trainer config dictionary.
    """
    config = base_config.copy()

    if "trainer" not in config:
        config["trainer"] = {}
    if "init" not in config["trainer"]:
        config["trainer"]["init"] = {}
    if "callbacks" not in config["trainer"]["init"]:
        config["trainer"]["init"]["callbacks"] = []

    callbacks = config["trainer"]["init"]["callbacks"]

    sparse_cb = {
        "SparseMask": {
            "mask_path": mask_path,
            "apply_every_step": (sparse_mode == "sparse_to_sparse"),
        }
    }
    callbacks.append(sparse_cb)

    if lora_config is not None:
        lora_cb = {"Lora": {"lora_params": lora_config}}
        has_lora = any(
            isinstance(cb, dict) and "Lora" in cb for cb in callbacks
        )
        if has_lora:
            for cb in callbacks:
                if isinstance(cb, dict) and "Lora" in cb:
                    cb["Lora"]["lora_params"].update(lora_config)
                    break
        else:
            callbacks.append(lora_cb)

    return config
