#!/usr/bin/env python
import argparse
import os
import torch
from accelerate import Accelerator
from pathlib import Path
from src.train.gpu_dense_sft import train_and_eval_week4
from src.utils.run_logging import (
    make_run_id, make_run_dir, snapshot_env, append_run_registry, write_json
)

DEFAULT_MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
}

def _has_adapter(run_dir: str) -> bool:
    """
    True if run_dir contains a usable adapter payload (LoRA or full model) in run_dir/adapter.
    For LoRA we expect adapter_config.json and adapter_model.* to exist.
    """
    ad = Path(run_dir) / "adapter"
    if not ad.is_dir():
        return False
    # LoRA adapter
    if (ad / "adapter_config.json").is_file():
        # adapter weights can be .safetensors or .bin depending on setup
        if any(ad.glob("adapter_model.*")):
            return True
    # Dense save fallback (not "adapter" in PEFT sense, but still a usable checkpoint)
    if (ad / "pytorch_model.bin").is_file() or any(ad.glob("model*.safetensors")):
        return True
    return False

def _find_existing_run_with_adapter(runs_root: str, prefix: str):
    """
    Find the newest run directory under runs_root whose basename starts with `prefix`
    and that contains an adapter. Returns (run_id, run_dir) or (None, None).
    """
    root = Path(runs_root)
    if not root.exists():
        return None, None

    candidates = []
    for p in root.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(prefix):
            continue
        if _has_adapter(str(p)):
            try:
                mtime = p.stat().st_mtime
            except Exception:
                mtime = 0.0
            candidates.append((mtime, p.name, str(p)))

    if not candidates:
        return None, None

    # Newest by mtime, then by name (run_id contains timestamp so lexicographic is also useful)
    candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
    _, run_id, run_dir = candidates[0]
    return run_id, run_dir

def main():
    accelerator = Accelerator()
    
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(DEFAULT_MODELS.keys()))
    ap.add_argument("--task", required=True, choices=["boolq", "hellaswag", "gsm8k"])
    ap.add_argument("--model_id", default=None, help="Override HF model id")
    ap.add_argument("--contract", default="configs/contracts/week4_dense_h100.yaml")
    ap.add_argument("--deepspeed", default="auto", choices=["auto","on","off"])
    ap.add_argument("--runs_root", default="results/runs")
    ap.add_argument("--registry_csv", default="results/run_registry.csv")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    model_id = args.model_id or DEFAULT_MODELS[args.model]
    # If an adapter already exists for this model/task, skip training and go straight to eval.
    prefix = f"gpu-dense-{args.model}-{args.task}-"

    # Create/select run_id/run_dir on rank0 only, then broadcast to all ranks
    # obj_list: [run_id, run_dir, eval_only_flag]
    obj_list = [None, None, None]
    if accelerator.is_main_process:
        existing_run_id, existing_run_dir = _find_existing_run_with_adapter(args.runs_root, prefix)
        if existing_run_id is not None and existing_run_dir is not None:
            run_id, run_dir = existing_run_id, existing_run_dir
            eval_only = True
            print(f"[CACHE] Found existing adapter; skipping training. run_id={run_id} run_dir={run_dir}")
        else:
            run_id = make_run_id(prefix=f"gpu-dense-{args.model}-{args.task}")
            run_dir = make_run_dir(args.runs_root, run_id)
            eval_only = False
        obj_list = [run_id, run_dir, eval_only]

    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(obj_list, src=0)
    run_id, run_dir, eval_only = obj_list[0], obj_list[1], obj_list[2]
    if run_id is None or run_dir is None or eval_only is None:
        raise RuntimeError("Failed to broadcast run_id/run_dir from rank0.")

    # Optional: override contract deepspeed.enabled at runtime
    # auto => on for mixtral, off otherwise
    import yaml
    with open(args.contract, "r") as f:
        contract_obj = yaml.safe_load(f)
    if args.deepspeed == "on":
        contract_obj["deepspeed"]["enabled"] = True
    elif args.deepspeed == "off":
        contract_obj["deepspeed"]["enabled"] = False
    else:
        # auto
        contract_obj["deepspeed"]["enabled"] = (args.model == "mixtral-8x7b")
    # write resolved contract into run_dir for provenance
    resolved_contract_path = os.path.join(run_dir, "week4_contract_runtime.yaml")
    if accelerator.is_main_process:
        with open(resolved_contract_path, "w") as f:
            yaml.safe_dump(contract_obj, f, sort_keys=False)

    # snapshot env early
    if accelerator.is_main_process:
        snapshot_env(run_dir)
        write_json(os.path.join(run_dir, "run_spec.json"), {
            "run_id": run_id,
            "model_alias": args.model,
            "model_id": model_id,
            "task": args.task,
            "contract": args.contract,
            "notes": args.notes,
        })

    run_result = train_and_eval_week4(
        model_id=model_id,
        task=args.task,
        contract_yaml=resolved_contract_path,
        run_dir=run_dir,
    )

    # Append registry row (schema-agnostic)
    row = {
        "run_id": run_id,
        "system": "gpu",
        "stage": "week4_dense",
        "model": args.model,
        "model_id": model_id,
        "task": args.task,
        "seq_len": run_result.get("seq_len"),
        "peft_mode": run_result.get("peft_mode"),
        "micro_batch_final": run_result.get("micro_batch_final"),
        "train_tokens_per_s": run_result.get("train_tokens_per_s"),
        "train_time_s": run_result.get("train_time_s"),
        "loss_last": run_result.get("loss_last"),
        "run_dir": run_dir,
        "notes": args.notes,
    }
    
    try:
        if accelerator.is_main_process:
            append_run_registry(args.registry_csv, row)
    
        if accelerator.is_main_process:
            print(f"[OK] run_id={run_id}")
            print(f"[OK] run_dir={run_dir}")
    except:
        pass

if __name__ == "__main__":
    main()
