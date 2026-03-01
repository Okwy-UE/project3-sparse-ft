#!/usr/bin/env python3
"""
Generate Cerebras YAML training configs for every
(model, task, sparsity, mode) combination.

Takes the *existing* dense-LoRA config as a base and patches:
  - ckpt_path -> sparse CS checkpoint
  - Adds sparsity / mode metadata (comments + extra labels)
  - For sparse_to_sparse: marks the mode in metadata.

Notes:
  - In this Model Zoo release, "SparseMask" is not a registered callback tag.
    Emitting it causes pydantic union_tag_invalid validation failures.
  - Base weights are frozen during LoRA training, so sparsity is preserved
    without per-step mask re-application callbacks.

Output directory: configs/sparse/{model}/{task}_s{sparsity}_{mode}.yaml
"""

import argparse
import copy
import os
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ======================================================================
# Model / path registry
# ======================================================================

MODELS = ["llama3", "mistral", "mixtral"]
TASKS = ["boolq", "gsm8k", "hellaswag"]
SPARSITIES = [25, 50, 75]
MODES = ["sparse_to_dense", "sparse_to_sparse"]

CS_MODEL_MAP = {
    "llama3": "llama",
    "mistral": "mistral",
}

BASE_CONFIG_DIR = PROJECT_ROOT / "configs" / "cerebras" / "generated"
SPARSE_CONFIG_DIR = PROJECT_ROOT / "configs" / "sparse"

# Models without masked HF checkpoints — use dense CS ckpt + callback masking
CALLBACK_ONLY_MODELS = {
    "mixtral": PROJECT_ROOT / "checkpoints" / "cs" / "mixtral_8x7b" / "model_to_cs-2.5.mdl",
}
HF_SPARSE_BASE = PROJECT_ROOT / "checkpoints" / "hf_sparse"
CS_SPARSE_BASE = PROJECT_ROOT / "checkpoints" / "cs_sparse"

MASK_BASE = PROJECT_ROOT / "masks"

# "SparseMask" is not accepted by R_2.9.0 trainer callback validation.
ALLOW_UNSUPPORTED_SPARSEMASK = (
    os.environ.get("ALLOW_UNSUPPORTED_SPARSEMASK", "0") == "1"
)
MIXTRAL_MICRO_BATCH_SIZE = os.environ.get("MIXTRAL_MICRO_BATCH_SIZE", "4")


# ======================================================================
# YAML helpers
# ======================================================================

def deep_get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def deep_set(d, path, value):
    cur = d
    for k in path[:-1]:
        if k not in cur or cur[k] is None:
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value


def parse_micro_batch_size(value: str):
    if value == "auto":
        return value
    return int(value)


# ======================================================================
# Config generation
# ======================================================================

def generate_config(
    model: str,
    task: str,
    sparsity: int,
    mode: str,
    max_steps: int = 200,
) -> Path:
    """Generate a single sparse training config."""
    base_yaml = BASE_CONFIG_DIR / model / f"{task}.yaml"
    if not base_yaml.exists():
        print(f"  SKIP: base config missing: {base_yaml}")
        return None

    with open(base_yaml) as f:
        cfg = yaml.safe_load(f)

    cs_ckpt_dir = CS_SPARSE_BASE / f"{model}_s{sparsity}"
    cs_ckpt_path = cs_ckpt_dir / "model_to_cs-2.5.mdl"

    deep_set(cfg, ["trainer", "fit", "ckpt_path"], str(cs_ckpt_path))

    mask_path = str(MASK_BASE / model / f"masks_sparsity{sparsity}_unstructured.pt")

    callbacks = deep_get(cfg, ["trainer", "init", "callbacks"], [])
    if callbacks is None:
        callbacks = []

    has_lora = any("Lora" in cb for cb in callbacks if isinstance(cb, dict))
    if not has_lora:
        callbacks.append({
            "Lora": {
                "lora_params": {
                    "r": 16,
                    "alpha": 32,
                    "dropout": 0.05,
                    "target_modules": ["Linear"],
                    "merge_weights": False,
                }
            }
        })

    if mode == "sparse_to_sparse" and ALLOW_UNSUPPORTED_SPARSEMASK:
        has_mask_cb = any(
            "SparseMask" in cb for cb in callbacks if isinstance(cb, dict)
        )
        if not has_mask_cb:
            callbacks.append({
                "SparseMask": {
                    "mask_path": mask_path,
                    "apply_every_step": True,
                }
            })

    deep_set(cfg, ["trainer", "init", "callbacks"], callbacks)

    loop = deep_get(cfg, ["trainer", "init", "loop"], {})
    if isinstance(loop, dict):
        loop["max_steps"] = max_steps
    deep_set(cfg, ["trainer", "init", "loop"], loop)

    if model == "mixtral":
        checkpoint = deep_get(cfg, ["trainer", "init", "checkpoint"], {})
        if isinstance(checkpoint, dict):
            # Avoid creating multi-GB step-0 checkpoints on every retry.
            checkpoint["save_initial_checkpoint"] = False
        deep_set(cfg, ["trainer", "init", "checkpoint"], checkpoint)

        micro_batch_size = parse_micro_batch_size(MIXTRAL_MICRO_BATCH_SIZE)
        for flag_key in ("ScopedTrainFlags", "ScopedValidateFlags"):
            matched = False
            for cb in callbacks:
                if isinstance(cb, dict) and flag_key in cb:
                    cb[flag_key]["csx.performance.micro_batch_size"] = micro_batch_size
                    matched = True
            if not matched:
                callbacks.append(
                    {flag_key: {"csx.performance.micro_batch_size": micro_batch_size}}
                )
        deep_set(cfg, ["trainer", "init", "callbacks"], callbacks)

    out_dir = SPARSE_CONFIG_DIR / model
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task}_s{sparsity}_{mode}.yaml"

    header = (
        f"# Auto-generated sparse LoRA config\n"
        f"# Model: {model}  Task: {task}  Sparsity: {sparsity}%  Mode: {mode}\n"
        f"# Mask : {mask_path}\n"
        f"# Ckpt : {cs_ckpt_path}\n"
    )

    with open(out_path, "w") as f:
        f.write(header)
        yaml.safe_dump(cfg, f, sort_keys=False)

    return out_path


def main():
    ap = argparse.ArgumentParser(description="Generate sparse LoRA YAML configs")
    ap.add_argument("--models", nargs="+", default=MODELS)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--sparsities", nargs="+", type=int, default=SPARSITIES)
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()

    generated = []
    skipped = []

    for model in args.models:
        for task in args.tasks:
            for sparsity in args.sparsities:
                for mode in args.modes:
                    print(f"Generating: {model} / {task} / s{sparsity} / {mode}")
                    path = generate_config(model, task, sparsity, mode,
                                           max_steps=args.max_steps)
                    if path:
                        generated.append(str(path))
                        print(f"  -> {path}")
                    else:
                        skipped.append(f"{model}/{task}/s{sparsity}/{mode}")

    print(f"\n{'='*60}")
    print(f"Generated {len(generated)} configs, skipped {len(skipped)}")
    if skipped:
        print("Skipped:")
        for s in skipped:
            print(f"  {s}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
