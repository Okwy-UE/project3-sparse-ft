import argparse
import os
from pathlib import Path

import yaml

def _first_present(d: dict, keys: list[str]) -> str | None:
    for k in keys:
        if isinstance(d, dict) and k in d:
            return k
    return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_yaml", required=True, help="Path to Model Zoo base config yaml")
    ap.add_argument("--out_yaml", required=True, help="Where to write generated yaml")
    ap.add_argument("--train_hdf5", required=True, help="Train HDF5 dir")
    ap.add_argument("--val_hdf5", required=True, help="Val HDF5 dir")
    ap.add_argument("--model_name", required=True, help="llama3|mistral|mixtral (for tagging)")
    ap.add_argument("--task", required=True, help="boolq|hellaswag|gsm8k")
    ap.add_argument("--msl", type=int, default=2048)
    ap.add_argument("--global_batch", type=int, default=256)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--eval_steps", type=int, default=200)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    args = ap.parse_args()

    with open(args.base_yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # -----------------------
    # Common overrides (best-effort across Model Zoo YAMLs)
    # -----------------------
    # 1) Sequence length
    for path in [
        ("model", "max_position_embeddings"),
        ("model", "max_sequence_length"),
        ("train_input", "max_sequence_length"),
        ("eval_input", "max_sequence_length"),
    ]:
        # only set if present
        cur = cfg
        ok = True
        for k in path[:-1]:
            if not isinstance(cur, dict) or k not in cur:
                ok = False
                break
            cur = cur[k]
        if ok and path[-1] in cur:
            cur[path[-1]] = args.msl

    # 2) Data inputs
    # Many Model Zoo configs use train_input.data_dir / eval_input.data_dir
    if "train_input" not in cfg:
        cfg["train_input"] = {}
    if "eval_input" not in cfg:
        cfg["eval_input"] = {}
    cfg["train_input"]["data_dir"] = args.train_hdf5
    cfg["eval_input"]["data_dir"] = args.val_hdf5

    # 3) Run schedule
    if "runconfig" not in cfg:
        cfg["runconfig"] = {}
    cfg["runconfig"]["max_steps"] = args.max_steps
    cfg["runconfig"]["eval_steps"] = args.eval_steps
    cfg["runconfig"]["checkpoint_steps"] = args.save_steps

    # 4) Optimizer LR
    if "optimizer" not in cfg:
        cfg["optimizer"] = {}
    if "learning_rate" in cfg["optimizer"]:
        cfg["optimizer"]["learning_rate"] = args.lr
    else:
        cfg["optimizer"]["learning_rate"] = args.lr

    # 5) Batch size MUST be configured under the input block for this schema
    # Prefer updating an existing batch key to avoid violating any strict input schemas.
    if "train_input" not in cfg:
        cfg["train_input"] = {}
    batch_key = _first_present(cfg["train_input"], ["batch_size", "batch_size_per_csx", "micro_batch_size"])
    if batch_key is None:
        batch_key = "batch_size"
    cfg["train_input"][batch_key] = args.global_batch

    # 6) LoRA block (based on Cerebras LoraConfig fields: r/alpha/dropout/target_modules). :contentReference[oaicite:13]{index=13}
    # The exact insertion point can vary by model; we place under cfg["model"]["lora"].
    if "model" not in cfg:
        cfg["model"] = {}
    lora_key = _first_present(cfg["model"], ["lora", "lora_config", "lora_params"])
    if lora_key is None:
        lora_key = "lora"
    cfg["model"][lora_key] = {
        "r": args.lora_r,
        "alpha": args.lora_alpha,
        "dropout": args.lora_dropout,
        # Conservative default: target decoder layer linear modules. Adjust after inspecting base model module names.
        "target_modules": ["TransformerDecoderLayer", "Linear"],
        # Keep unmerged during training; Week-3 will handle sparsity-aware merge.
        "merge_weights": False,
    }

    Path(os.path.dirname(args.out_yaml)).mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[OK] Wrote: {args.out_yaml}")

if __name__ == "__main__":
    main()
