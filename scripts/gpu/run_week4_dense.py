#!/usr/bin/env python
import argparse
import os
import torch
from accelerate import Accelerator

from src.train.gpu_dense_sft import train_and_eval_week4
from src.train.gpu_dense_sft import resolve_week4_config  # reuse existing config mapping
from src.eval.eval_phoenix_tasks import run_lm_eval_harness
from src.utils.run_logging import (
    make_run_id, make_run_dir, snapshot_env, append_run_registry, write_json
)

def _latest_matching_run_dir(runs_root: str, prefix: str) -> str | None:
    if not os.path.isdir(runs_root):
        return None
    cands = []
    for name in os.listdir(runs_root):
        path = os.path.join(runs_root, name)
        if os.path.isdir(path) and name.startswith(prefix):
            cands.append(path)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def _eval_only(model_id: str, task: str, run_dir: str, contract_fallback: str):
    # Prefer the resolved runtime contract saved in the run_dir (best provenance).
    import yaml
    runtime_contract = os.path.join(run_dir, "week4_contract_runtime.yaml")
    contract_path = runtime_contract if os.path.exists(runtime_contract) else contract_fallback
    with open(contract_path, "r") as f:
        contract_obj = yaml.safe_load(f)
    print(f"[EVAL-ONLY] Resolving config")
    cfg = resolve_week4_config(contract_obj, task)

    adapter_dir = os.path.abspath(os.path.join(run_dir, "adapter"))
    eval_path = os.path.join(run_dir, "lm_eval.json")

    # Detect LoRA vs "full model saved" in adapter_dir.
    # - LoRA: adapter_config.json exists => eval base HF model + peft adapter.
    # - Dense: no adapter_config.json => eval model from adapter_dir directly.
    is_lora = os.path.exists(os.path.join(adapter_dir, "adapter_config.json"))
    base_for_eval = model_id if is_lora else adapter_dir
    peft_path = adapter_dir if is_lora else None

    # Ensure single-process eval
    print(f"[EVAL-ONLY] Destroying process group")
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    os.environ["WORLD_SIZE"] = "1"
    os.environ["RANK"] = "0"
    os.environ["LOCAL_RANK"] = "0"

    print(f"[EVAL-ONLY] Starting evaluation")
    eval_out = run_lm_eval_harness(
        base_model_id=base_for_eval,
        peft_adapter_path=peft_path,
        tasks=[cfg.eval_task_name],
        out_json_path=eval_path,
        batch_size=cfg.eval_batch_size,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
        extra_model_args=None,
        write_results=True,
    )
    # Lightweight provenance
    write_json(os.path.join(run_dir, "eval_only_result.json"), {
        "task": task,
        "eval_task_name": cfg.eval_task_name,
        "batch_size": cfg.eval_batch_size,
        "is_lora": is_lora,
        "base_model_id_for_eval": base_for_eval,
        "peft_adapter_path": peft_path,
        "lm_eval": eval_out,
    })
    return eval_out

DEFAULT_MODELS = {
    "llama3.1-8b": "meta-llama/Llama-3.1-8B",
    "mistral-7b": "mistralai/Mistral-7B-v0.3",
    "mixtral-8x7b": "mistralai/Mixtral-8x7B-v0.1",
}

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
    prefix = f"gpu-dense-{args.model}-{args.task}"

    # Decide: eval-only or train+eval
    eval_only = False
    run_dir = None
    run_id = None
    if accelerator.is_main_process:
        found = _latest_matching_run_dir(args.runs_root, prefix=prefix)
        if found is not None:
            run_dir = found
            run_id = os.path.basename(os.path.abspath(run_dir))
            eval_only = True

    # Broadcast decision + run_dir to all ranks
    obj_list = [run_id, run_dir, eval_only]
    if accelerator.num_processes > 1:
        torch.distributed.broadcast_object_list(obj_list, src=0)
    run_id, run_dir, eval_only = obj_list
    if run_id is None or run_dir is None:
        raise RuntimeError("Failed to determine/broadcast run_id/run_dir from rank0.")

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

    if eval_only:
        run_result = {"lm_eval": None}
        if accelerator.is_main_process:
            print(f"[EVAL-ONLY] Using existing run_dir={run_dir}")
            eval_out = _eval_only(
                model_id=model_id,
                task=args.task,
                run_dir=run_dir,
                contract_fallback=resolved_contract_path,
            )
            run_result["lm_eval"] = eval_out
    else:
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
