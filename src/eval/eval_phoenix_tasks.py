from __future__ import annotations
import json
import os
from pathlib import Path
import numpy as np
import torch
from typing import Dict, Any, List, Optional
import torch.distributed as dist

def _is_rank0() -> bool:
    # Works whether dist is initialized or not
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return int(os.environ.get("RANK", "0")) == 0

def _json_sanitize(obj: Any) -> Any:
    """
    Convert non-JSON-serializable objects (torch dtype, numpy scalars, etc.)
    into JSON-safe primitives recursively.
    """
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj

    # torch dtype / device
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)

    # numpy types
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return obj.item()
    if isinstance(obj, (np.dtype,)):
        return str(obj)

    # containers
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]

    # fallback: stringify unknown objects
    return str(obj)

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
    def _is_distributed_env() -> bool:
        if dist.is_available() and dist.is_initialized():
            return dist.get_world_size() > 1
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
        
    if device.startswith("cuda") and torch.cuda.device_count() > 1 and not _is_distributed_env():
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
        # Add num_fewshot here to match a specific Phoenix setting.
    )

    print("Done with evaluation")

    if _is_rank0():
        try:
            with open(out_json_path, "w") as f:
                json.dump(_json_sanitize(res), f, indent=2, sort_keys=True)
        except (TypeError, ValueError) as e:
            pickle_path = out_json_path + ".pkl"
            import pickle
            with open(pickle_path, "wb") as f:
                pickle.dump(res, f)
    
    return _json_sanitize(res)


