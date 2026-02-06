from __future__ import annotations
import json
import os
from pathlib import Path
import torch
from typing import Dict, Any, List, Optional

def run_lm_eval_harness(
    base_model_id: str,
    peft_adapter_path: Optional[str],
    tasks: List[str],
    out_json_path: str,
    batch_size: str = "auto",
    device: str = "cuda:0",
    extra_model_args: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    Uses lm-eval-harness (EleutherAI) HuggingFace backend.
    Supports evaluating PEFT adapters by passing `peft=...` in model_args.
    """
    try:
        from lm_eval import evaluator
    except Exception as e:
        raise RuntimeError(
            "lm-eval-harness is not installed. Install with: pip install 'lm_eval[hf]'\n"
            f"Import error: {e}"
        )

    dtype = "float16"
    if extra_model_args and "dtype" in extra_model_args:
        dtype = extra_model_args["dtype"]

    model_args_parts = [f"pretrained={base_model_id}", f"dtype={dtype}"]

    # If multiple GPUs are visible and we're evaluating on CUDA, shard model across GPUs.
    if device.startswith("cuda") and torch.cuda.device_count() > 1:
        model_args_parts.append("device_map=auto")

    if peft_adapter_path is not None:
        peft_abs = str(Path(peft_adapter_path).resolve())
        model_args_parts.append(f"peft={peft_abs}")
    if extra_model_args:
        for k, v in extra_model_args.items():
            if k=="dtype":
                continue
            model_args_parts.append(f"{k}={v}")

    model_args = ",".join(model_args_parts)

    # NOTE: HFLM constructed above does not automatically load PEFT adapter.
    # In python API, easiest is to use evaluator.simple_evaluate with model_args string.
    res = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        batch_size=batch_size,
        device=device,
        # You can add num_fewshot here if you want to match a specific Phoenix setting.
    )

    print("Done with evaluation")

    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w") as f:
        json.dump(res, f, indent=2, sort_keys=True)

    return res
