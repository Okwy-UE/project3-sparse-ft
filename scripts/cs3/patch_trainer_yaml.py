#!/usr/bin/env python3
import argparse
import copy
import yaml

def as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]

def ensure_paths(cfg, train_dir, val_dir):
    trainer = cfg.get("trainer", {})
    # fit.train_dataloader
    td = trainer.get("fit", {}).get("train_dataloader", {})
    if isinstance(td, dict):
        td["data_dir"] = train_dir
        trainer.setdefault("fit", {})["train_dataloader"] = td

    # fit.val_dataloader (often list)
    vd = trainer.get("fit", {}).get("val_dataloader", None)
    vd_list = as_list(vd)
    if vd_list:
        for d in vd_list:
            if isinstance(d, dict):
                d["data_dir"] = val_dir
        trainer.setdefault("fit", {})["val_dataloader"] = vd_list

    # validate.val_dataloader
    v1 = trainer.get("validate", {}).get("val_dataloader", {})
    if isinstance(v1, dict):
        v1["data_dir"] = val_dir
        trainer.setdefault("validate", {})["val_dataloader"] = v1

    # validate_all.val_dataloaders (often list)
    v_all = trainer.get("validate_all", {}).get("val_dataloaders", None)
    v_all_list = as_list(v_all)
    if v_all_list:
        for d in v_all_list:
            if isinstance(d, dict):
                d["data_dir"] = val_dir
        trainer.setdefault("validate_all", {})["val_dataloaders"] = v_all_list

    cfg["trainer"] = trainer
    return cfg

def ensure_eval_frequency(cfg):
    trainer = cfg.get("trainer", {})
    init = trainer.get("init", {})
    loop = init.get("loop", {})
    if isinstance(loop, dict):
        # If eval_frequency exists, set it to eval_steps (or a sane default)
        if "eval_steps" in loop:
            loop["eval_frequency"] = loop["eval_steps"]
        else:
            loop.setdefault("eval_steps", 200)
            loop["eval_frequency"] = loop["eval_steps"]
    init["loop"] = loop
    trainer["init"] = init
    cfg["trainer"] = trainer
    return cfg

def remove_model_lora(cfg):
    trainer = cfg.get("trainer", {})
    init = trainer.get("init", {})
    model = init.get("model", {})
    if isinstance(model, dict) and "lora" in model:
        model.pop("lora", None)
    init["model"] = model
    trainer["init"] = init
    cfg["trainer"] = trainer
    return cfg

def ensure_lora_callback(cfg, r, alpha, dropout, target_modules, merge_weights):
    trainer = cfg.get("trainer", {})
    init = trainer.get("init", {})
    callbacks = init.get("callbacks", [])
    if callbacks is None:
        callbacks = []
    if not isinstance(callbacks, list):
        callbacks = [callbacks]

    # Remove any existing Lora/LoRA callbacks to avoid duplicates
    def is_lora(cb):
        return isinstance(cb, dict) and ("Lora" in cb or "LoRA" in cb)

    callbacks = [cb for cb in callbacks if not is_lora(cb)]

    lora_cb = {
        "Lora": {
            "lora_params": {
                "r": int(r),
                "alpha": int(alpha),
                "dropout": float(dropout),
                "target_modules": list(target_modules),
                "merge_weights": bool(merge_weights),
            }
        }
    }

    # Insert after LoadCheckpointStates if present
    inserted = False
    for i, cb in enumerate(callbacks):
        if isinstance(cb, dict) and "LoadCheckpointStates" in cb:
            callbacks.insert(i + 1, lora_cb)
            inserted = True
            break
    if not inserted:
        callbacks.append(lora_cb)

    init["callbacks"] = callbacks
    trainer["init"] = init
    cfg["trainer"] = trainer
    return cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yaml", required=True)
    ap.add_argument("--train_dir", required=True)
    ap.add_argument("--val_dir", required=True)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", default="Linear")  # comma-separated
    ap.add_argument("--merge_weights", action="store_true")
    args = ap.parse_args()

    with open(args.yaml, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Only patch trainer-style configs (these are the ones that error with model.config.lora)
    if not isinstance(cfg, dict) or "trainer" not in cfg:
        print(f"[SKIP] Not a trainer-style YAML: {args.yaml}")
        return

    targets = [t.strip() for t in args.target_modules.split(",") if t.strip()]

    cfg = ensure_paths(cfg, args.train_dir, args.val_dir)
    cfg = ensure_eval_frequency(cfg)
    cfg = remove_model_lora(cfg)
    cfg = ensure_lora_callback(cfg, args.lora_r, args.lora_alpha, args.lora_dropout, targets, args.merge_weights)

    with open(args.yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"[OK] Patched trainer YAML: {args.yaml}")

if __name__ == "__main__":
    main()
