import argparse
import os
from pathlib import Path

import yaml

def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def _get(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur

def _set(d, path, value):
    cur = d
    for k in path[:-1]:
        if k not in cur or cur[k] is None:
            cur[k] = {}
        cur = cur[k]
    cur[path[-1]] = value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_yaml", required=True, help="Path to Model Zoo base config yaml")
    ap.add_argument("--out_yaml", required=True, help="Where to write generated yaml")
    ap.add_argument("--train_hdf5", required=True, help="Train HDF5 dir")
    ap.add_argument("--val_hdf5", required=True, help="Val HDF5 dir")
    ap.add_argument("--model_name", required=True, help="llama3|mistral|mixtral (for tagging)")
    ap.add_argument("--task", required=True, help="boolq|hellaswag|gsm8k")
    ap.add_argument("--msl", type=int, default=2048)
    ap.add_argument("--dataloader_batch", type=int, default=None)
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

    # Detect whether this is a trainer-style config (LLaMA3/Mistral/Mixtral in 2.6.0 often are)
    is_trainer = isinstance(cfg, dict) and "trainer" in cfg and isinstance(cfg["trainer"], dict)

    # Always avoid adding random top-level keys that might violate strict schemas:
    # remove legacy keys if we previously injected them.
    for k in ["train_input", "eval_input", "optimizer", "model", "metadata", "runconfig"]:
        if k in cfg and is_trainer:
            cfg.pop(k, None)

    if is_trainer:
        # -----------------------
        # TRAINER-STYLE OVERRIDES
        # -----------------------
        # 1) Patch dataloader paths (this fixes your current error)
        fit_train = _get(cfg, ["trainer", "fit", "train_dataloader"], {})
        if isinstance(fit_train, dict):
            fit_train["data_dir"] = args.train_hdf5
            if args.dataloader_batch is not None and "batch_size" in fit_train:
                fit_train["batch_size"] = args.dataloader_batch
            _set(cfg, ["trainer", "fit", "train_dataloader"], fit_train)

        fit_val = _get(cfg, ["trainer", "fit", "val_dataloader"], None)
        fit_val_list = _as_list(fit_val)
        if fit_val_list:
            for vd in fit_val_list:
                if isinstance(vd, dict):
                    vd["data_dir"] = args.val_hdf5
                    if args.dataloader_batch is not None and "batch_size" in vd:
                        vd["batch_size"] = args.dataloader_batch
            _set(cfg, ["trainer", "fit", "val_dataloader"], fit_val_list)

        v1 = _get(cfg, ["trainer", "validate", "val_dataloader"], None)
        if isinstance(v1, dict):
            v1["data_dir"] = args.val_hdf5
            if args.dataloader_batch is not None and "batch_size" in v1:
                v1["batch_size"] = args.dataloader_batch
            _set(cfg, ["trainer", "validate", "val_dataloader"], v1)

        v_all = _get(cfg, ["trainer", "validate_all", "val_dataloaders"], None)
        v_all_list = _as_list(v_all)
        if v_all_list:
            for vd in v_all_list:
                if isinstance(vd, dict):
                    vd["data_dir"] = args.val_hdf5
                    if args.dataloader_batch is not None and "batch_size" in vd:
                        vd["batch_size"] = args.dataloader_batch
            _set(cfg, ["trainer", "validate_all", "val_dataloaders"], v_all_list)

        # 2) Loop / checkpoint cadence (trainer-style)
        loop = _get(cfg, ["trainer", "init", "loop"], {})
        if isinstance(loop, dict):
            loop["max_steps"] = args.max_steps
            loop["eval_steps"] = args.eval_steps
            _set(cfg, ["trainer", "init", "loop"], loop)

        ckpt = _get(cfg, ["trainer", "init", "checkpoint"], {})
        if isinstance(ckpt, dict):
            # in trainer configs, checkpoint cadence is commonly "steps"
            ckpt["steps"] = args.save_steps
            _set(cfg, ["trainer", "init", "checkpoint"], ckpt)

        # 3) LR: try to set end_learning_rate on LinearLR (safe), without breaking schema
        scheds = _get(cfg, ["trainer", "init", "schedulers"], None)
        if isinstance(scheds, list):
            for s in scheds:
                if isinstance(s, dict) and "LinearLR" in s and isinstance(s["LinearLR"], dict):
                    s["LinearLR"]["end_learning_rate"] = float(args.lr)

        # 4) Remove known-unneeded optimizer key that triggers warnings (optional but clean)
        opt = _get(cfg, ["trainer", "init", "optimizer"], None)
        if isinstance(opt, dict) and "Adam" in opt and isinstance(opt["Adam"], dict):
            opt["Adam"].pop("correct_bias", None)

        # 5) LoRA: MUST be injected under trainer.init.model (top-level model block is ignored here)
        model = _get(cfg, ["trainer", "init", "model"], {})
        if isinstance(model, dict):
            model["lora"] = {
                "r": int(args.lora_r),
                "alpha": int(args.lora_alpha),
                "dropout": float(args.lora_dropout),
                "target_modules": ["Linear"],  # safest initial target; widen later if needed
                "merge_weights": False,
            }
            _set(cfg, ["trainer", "init", "model"], model)

        # 6) Sequence length: set max_position_embeddings if present
        model = _get(cfg, ["trainer", "init", "model"], None)
        if isinstance(model, dict) and "max_position_embeddings" in model:
            model["max_position_embeddings"] = int(args.msl)

    else:
        # -----------------------
        # LEGACY (non-trainer) OVERRIDES
        # -----------------------
        if "train_input" not in cfg:
            cfg["train_input"] = {}
        if "eval_input" not in cfg:
            cfg["eval_input"] = {}
        cfg["train_input"]["data_dir"] = args.train_hdf5
        cfg["eval_input"]["data_dir"] = args.val_hdf5

        if "runconfig" not in cfg:
            cfg["runconfig"] = {}
        cfg["runconfig"]["max_steps"] = args.max_steps
        cfg["runconfig"]["eval_steps"] = args.eval_steps
        cfg["runconfig"]["checkpoint_steps"] = args.save_steps

        # optimizer/lora placement depends on model; keep minimal here

    Path(os.path.dirname(args.out_yaml)).mkdir(parents=True, exist_ok=True)
    with open(args.out_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[OK] Wrote: {args.out_yaml}")

if __name__ == "__main__":
    main()
