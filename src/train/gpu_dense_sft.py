from __future__ import annotations

import gc
import math
from typing import Any
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

import torch
from torch.utils.data import DataLoader
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler

from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

from src.data.phoenix_sft_tasks import build_sft_dataset, make_tokenize_fn
from src.eval.eval_phoenix_tasks import run_lm_eval_harness
from src.utils.run_logging import write_json

import shutil
import traceback

from tqdm.auto import tqdm
import sys

def is_main_process(accelerator) -> bool:
    return getattr(accelerator, "is_main_process", True)

def tqdm_if_main(accelerator, iterable, **kwargs):
    # Avoid multi-bar spam + avoid aggressive log updates in Slurm output
    disable = (not is_main_process(accelerator)) or (not sys.stdout.isatty())
    return tqdm(iterable, disable=disable, mininterval=2.0, **kwargs)

def cuda_mem(msg: str):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserv = torch.cuda.memory_reserved() / 1024**3
    max_alloc = torch.cuda.max_memory_allocated() / 1024**3
    print(f"[CUDA] {msg:>18s} | alloc={alloc:6.2f} GB | reserv={reserv:6.2f} GB | max_alloc={max_alloc:6.2f} GB")

def _supports_bf16() -> bool:
    """
    BF16 is only supported on Ampere+ GPUs (SM80+). V100 (SM70) does NOT support bf16.
    """
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability(0)
    return major >= 8

def _effective_precision(cfg, run_dir: str):
    """
    Return (mixed_precision_str, torch_dtype) adjusted for hardware.
    """
    mp = cfg.mixed_precision
    td = torch.bfloat16 if mp == "bf16" else torch.float16 if mp == "fp16" else torch.float32
    if mp == "bf16" and not _supports_bf16():
        mp = "fp16"
        td = torch.float16
        write_json(os.path.join(run_dir, "precision_override.json"), {
            "requested_mixed_precision": cfg.mixed_precision,
            "effective_mixed_precision": mp,
            "reason": "GPU does not support bf16 (e.g., V100 SM70). Switching to fp16.",
        })
    return mp, td

def _hard_cuda_cleanup(objs: Dict[str, Any], accelerator=None) -> None:
    """
    Try very hard to return GPU memory to the driver before running lm-eval.
    On some stacks, del(model) is not sufficient because wrappers keep references.
    """
    try:
        if accelerator is not None:
            try:
                accelerator.wait_for_everyone()
            except Exception:
                pass
            # accelerate >= 0.26
            if hasattr(accelerator, "free_memory"):
                try:
                    accelerator.free_memory()
                except Exception:
                    pass
    except Exception:
        pass

    # Drop references we know about
    for k in list(objs.keys()):
        try:
            del objs[k]
        except Exception:
            pass

    # Drop accelerator last (it can hold model/optim refs)
    try:
        if accelerator is not None:
            del accelerator
    except Exception:
        pass

    # Force GC + CUDA cache cleanup
    try:
        gc.collect()
    except Exception:
        pass
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

def _cuda_toolkit_available() -> bool:
    """
    DeepSpeed op loader needs CUDA toolkit (CUDA_HOME and nvcc).
    On some HPC setups you have driver libs but not toolkit, so disable DS.
    """
    cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = shutil.which("nvcc")
    return (cuda_home is not None and os.path.isdir(cuda_home)) or (nvcc is not None)

def _make_accelerator(cfg, run_dir: str):
    """
    Create an Accelerator. If DeepSpeed is requested but not available, fall back
    to non-DeepSpeed accelerator and log the reason to run_dir.
    """
    from accelerate import Accelerator

    if not getattr(cfg, "deepspeed_enabled", False):
        return Accelerator(mixed_precision="fp16")

    # DeepSpeed requested
    if not _cuda_toolkit_available():
        # fallback
        write_json(os.path.join(run_dir, "deepspeed_disabled.json"), {
            "requested": True,
            "enabled": False,
            "reason": "CUDA toolkit not found (CUDA_HOME/CUDA_PATH and nvcc missing). "
                      "module load cuda or export CUDA_HOME, or run without deepspeed.",
            "CUDA_HOME": os.environ.get("CUDA_HOME"),
            "CUDA_PATH": os.environ.get("CUDA_PATH"),
            "nvcc": shutil.which("nvcc"),
        })
        return Accelerator(mixed_precision="fp16")

    # Try building deepspeed plugin
    try:
        from accelerate.utils import DeepSpeedPlugin
        with open(cfg.deepspeed_config_path, "r") as f:
            ds_cfg = json.load(f)
        ds_plugin = DeepSpeedPlugin(hf_ds_config=ds_cfg)
        return Accelerator(mixed_precision="fp16", deepspeed_plugin=ds_plugin)
    except Exception as e:
        write_json(os.path.join(run_dir, "deepspeed_disabled.json"), {
            "requested": True,
            "enabled": False,
            "reason": "DeepSpeed failed to import/initialize; falling back to non-DeepSpeed",
            "error": str(e),
            "traceback": traceback.format_exc(),
        })
        return Accelerator(mixed_precision="fp16")

@dataclass
class Week4Config:
    seed: int
    max_seq_len: int
    max_steps: int
    warmup_ratio: float
    lr: float
    weight_decay: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    grad_clip_norm: float
    mixed_precision: str  # "bf16"
    # peft
    peft_mode: str  # "lora" or "none"
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: List[str]
    # data
    max_train_samples: Optional[int]
    max_eval_samples: Optional[int]
    num_proc: int
    # deepspeed
    deepspeed_enabled: bool
    deepspeed_config_path: str
    # throughput
    tp_warmup_steps: int
    tp_measure_steps: int
    tp_points: int
    # eval
    eval_use_lm_eval: bool
    eval_task_name: str
    eval_batch_size: str


def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_contract_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_week4_config(contract: Dict[str, Any], task: str) -> Week4Config:
    peft = contract["peft"]
    data = contract["data"]
    ds = contract["deepspeed"]
    tp = contract["throughput"]
    ev = contract["eval"]

    return Week4Config(
        seed=int(contract["seed"]),
        max_seq_len=int(contract["max_seq_len"]),
        max_steps=int(contract["max_steps"][task]),
        warmup_ratio=float(contract["warmup_ratio"]),
        lr=float(contract["lr"]),
        weight_decay=float(contract["weight_decay"]),
        adam_beta1=float(contract["adam_beta1"]),
        adam_beta2=float(contract["adam_beta2"]),
        adam_eps=float(contract["adam_eps"]),
        grad_clip_norm=float(contract["grad_clip_norm"]),
        mixed_precision=str(contract["mixed_precision"]),
        peft_mode=str(peft["mode"]),
        lora_r=int(peft["r"]),
        lora_alpha=int(peft["alpha"]),
        lora_dropout=float(peft["dropout"]),
        lora_target_modules=list(peft["target_modules"]),
        max_train_samples=data.get("max_train_samples", None),
        max_eval_samples=data.get("max_eval_samples", None),
        num_proc=int(data.get("num_proc", 1)),
        deepspeed_enabled=bool(ds["enabled"]),
        deepspeed_config_path=str(ds["config_path"]),
        tp_warmup_steps=int(tp["warmup_steps"]),
        tp_measure_steps=int(tp["measure_steps"]),
        tp_points=int(tp["points"]),
        eval_use_lm_eval=bool(ev["use_lm_eval"]),
        eval_task_name=str(ev["tasks_map"][task]),
        eval_batch_size=str(ev["batch_size"]),
    )


def maybe_apply_lora(model, cfg: Week4Config):
    if cfg.peft_mode != "lora":
        return model, None

    try:
        from peft import LoraConfig, get_peft_model
    except Exception as e:
        raise RuntimeError(
            "peft is required for LoRA baseline. Install: pip install peft\n"
            f"Import error: {e}"
        )

    lora_cfg = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=cfg.lora_target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, lora_cfg


def build_loaders(task: str, tokenizer, cfg: Week4Config, micro_batch: int):
    ds_train = build_sft_dataset(task, "train", max_samples=cfg.max_train_samples, seed=cfg.seed)
    ds_eval = build_sft_dataset(task, "eval", max_samples=cfg.max_eval_samples, seed=cfg.seed)

    tok_fn = make_tokenize_fn(tokenizer, cfg.max_seq_len)
    ds_train = ds_train.map(tok_fn, remove_columns=ds_train.column_names, num_proc=cfg.num_proc, desc="tokenize train")
    ds_eval = ds_eval.map(tok_fn, remove_columns=ds_eval.column_names, num_proc=cfg.num_proc, desc="tokenize eval")

    cols = ["input_ids", "attention_mask", "labels"]
    ds_train.set_format(type="torch", columns=cols)
    ds_eval.set_format(type="torch", columns=cols)

    dl_train = DataLoader(ds_train, batch_size=micro_batch, shuffle=True, drop_last=True, num_workers=2, pin_memory=True)
    dl_eval = DataLoader(ds_eval, batch_size=micro_batch, shuffle=False, drop_last=False, num_workers=2, pin_memory=True)
    return ds_train, ds_eval, dl_train, dl_eval

@dataclass
class BenchContext:
    tokenizer: Any
    model: Any
    accelerator: Any

def _init_bench_context(model_id: str, cfg: Week4Config, run_dir: str) -> BenchContext:
    torch.cuda.empty_cache()
    gc.collect()
    set_seed(cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model, _ = maybe_apply_lora(model, cfg)

    accelerator = _make_accelerator(cfg, run_dir)
    return BenchContext(tokenizer=tokenizer, model=model, accelerator=accelerator)

def profile_train_steps(model, optim, sched, batch, accelerator, logdir: str, steps: int = 3):
    print("profiling has begun")
    prof = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=0, warmup=1, active=steps, repeat=1),
        on_trace_ready=tensorboard_trace_handler(logdir),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,   # gives file/line for ops (can be heavy but great for attribution)
    )

    model.train()
    torch.cuda.reset_peak_memory_stats()

    # Track where we are so failures are attributable.
    phase = "init"
    i = -1

    try:
        with prof:
            for i in range(1 + steps):  # 1 warmup + active
                with accelerator.accumulate(model):
                    phase = "fwd"
                    with torch.profiler.record_function("profile_train_steps/fwd"):
                        out = model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                            labels=batch["labels"],
                        )
                    cuda_mem("Profile fwd pass")
                    phase = "loss"
                    with torch.profiler.record_function("profile_train_steps/loss"):
                        loss = out.loss
                    cuda_mem("Profile loss")
                    phase = "backward"
                    with torch.profiler.record_function("profile_train_steps/backward"):
                        accelerator.backward(loss)
                    cuda_mem("Profile backward")
                    phase = "optim.step"
                    with torch.profiler.record_function("profile_train_steps/optim_step"):
                        optim.step()
                    cuda_mem("Profile optimize")
                    phase = "sched.step"
                    with torch.profiler.record_function("profile_train_steps/sched_step"):
                        sched.step()
                    phase = "zero_grad"
                    with torch.profiler.record_function("profile_train_steps/zero_grad"):
                        optim.zero_grad(set_to_none=True)
                    cuda_mem("Profile optim reset")
    
                accelerator.wait_for_everyone()
                prof.step()

        print(
            prof.key_averages()
                .table(sort_by="self_cuda_memory_usage", row_limit=30)
        )
    except Exception as e:
        print(f"[PROFILE ERROR] profile_train_steps failed at loop i={i} phase='{phase}'")
        print(f"[PROFILE ERROR] exception: {repr(e)}")
        tb = traceback.format_exc()
        print("[PROFILE ERROR] traceback:")
        print(tb)
        pass

def bench_throughput_single(
    model_id: str,
    task: str,
    run_dir: str,
    cfg: Week4Config,
    micro_batch: int,
    bench_ctx=None
) -> Dict[str, Any]:
    """
    Bench a short train loop to measure tokens/s at a given micro_batch.
    Uses DeepSpeed (if enabled) for Mixtral feasibility.
    """
    if bench_ctx is None:
        bench_ctx = _init_bench_context(model_id, cfg, run_dir)
    tokenizer = bench_ctx.tokenizer
    model = bench_ctx.model
    accelerator = bench_ctx.accelerator

    _, _, dl_train, _ = build_loaders(task, tokenizer, cfg, micro_batch=micro_batch)

    # Optimizer over trainable params only (LoRA => small)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    num_warmup = max(1, int(cfg.warmup_ratio * cfg.max_steps))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup,
        num_training_steps=cfg.max_steps,
    )

    model, optim, dl_train, sched = accelerator.prepare(model, optim, dl_train, sched)
    model.train()

    # Warmup + measure
    total_steps = cfg.tp_warmup_steps + cfg.tp_measure_steps
    step_times = []
    tokens_per_step = micro_batch * accelerator.num_processes * cfg.max_seq_len

    it = iter(dl_train)
    for step in tqdm_if_main(accelerator, range(total_steps), desc=f"bench(bs={micro_batch})"):
        torch.cuda.reset_peak_memory_stats()
        cuda_mem("start_step")
        
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_train)
            batch = next(it)
        
        t0 = time.perf_counter()
        with accelerator.accumulate(model):
            if step == 0:
                print(f"profiling for throughput of batch_size {micro_batch} started (step=0 only)")
                profile_train_steps(
                    model, optim, sched, batch, accelerator,
                    logdir=os.path.join(run_dir, "tb_prof"),
                    steps=3,
                )
                print("profiling for throughput ended")
            out = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            cuda_mem("after_fwd")
            
            loss = out.loss
            accelerator.backward(loss)
            cuda_mem("after_bwd")
            if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
            optim.step()
            cuda_mem("after_optim")
            sched.step()
            optim.zero_grad(set_to_none=True)
            cuda_mem("after_zero_grad")

        accelerator.wait_for_everyone()
        dt = time.perf_counter() - t0
        if step >= cfg.tp_warmup_steps:
            step_times.append(dt)

    mean_dt = sum(step_times) / max(1, len(step_times))
    tok_s = tokens_per_step / mean_dt if mean_dt > 0 else 0.0

    peak_mem = None
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3)

    metrics = {
        "micro_batch": micro_batch,
        "world_size": accelerator.num_processes,
        "seq_len": cfg.max_seq_len,
        "tokens_per_step": tokens_per_step,
        "mean_step_time_s": mean_dt,
        "tokens_per_s": tok_s,
        "peak_mem_gb": peak_mem,
    }

    return metrics


def pick_throughput_points(max_micro: int, k: int) -> List[int]:
    # choose descending points: max, ~half, ~quarter (unique, >=1)
    pts = []
    cur = max_micro
    while len(pts) < k and cur >= 1:
        if cur not in pts:
            pts.append(cur)
        cur = max(1, cur // 2)
        if cur == 1 and 1 in pts:
            break
    # ensure at least 3 points if possible
    if len(pts) < k:
        for b in [1, 2, 4, 8, 16]:
            if b <= max_micro and b not in pts:
                pts.append(b)
            if len(pts) >= k:
                break
    return pts[:k]


def auto_find_max_micro_batch(model_id: str, task: str, run_dir: str, cfg: Week4Config, start: int = 16, cap: int = 32, bench_ctx=None) -> int:
    """
    Conservative OOM-based search: try 1,2,4,... until fail.
    Returns largest successful micro_batch.
    """
    best = 1
    b = start
    while b <= cap:
        try:
            _ = bench_throughput_single(model_id, task, run_dir, cfg, micro_batch=b, bench_ctx=bench_ctx)
            best = b
            b *= 2
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda oom" in msg:
                print(f"best size is {best}")
                break
            # non-OOM error should surface
            raise
        finally:
            torch.cuda.empty_cache()
            gc.collect()
    return best


def train_and_eval_week4(
    model_id: str,
    task: str,
    contract_yaml: str,
    run_dir: str,
    eval_device: str = "cuda:0",
) -> Dict[str, Any]:
    contract = _load_contract_yaml(contract_yaml)
    cfg = resolve_week4_config(contract, task)
    set_seed(cfg.seed)
    mp, torch_dtype = _effective_precision(cfg, run_dir)

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    write_json(os.path.join(run_dir, "week4_contract_resolved.json"), contract)

    # ---- Auto micro-batch
    bench_ctx = _init_bench_context(model_id, cfg, run_dir)
    max_micro = auto_find_max_micro_batch(model_id, task, run_dir, cfg, start=1, cap=64, bench_ctx=bench_ctx)
    print(f"Max Micro is {max_micro}")
    tp_points = pick_throughput_points(max_micro, cfg.tp_points)

    # ---- Throughput bench
    print("Now beginning batch size throughputting")
    tp_metrics = []
    for b in tp_points:
        m = bench_throughput_single(model_id, task, run_dir, cfg, micro_batch=b, bench_ctx=bench_ctx)
        tp_metrics.append(m)

    write_json(os.path.join(run_dir, "throughput_scaling.json"), {"points": tp_metrics})

    try:
        _hard_cuda_cleanup(
            {
                "bench_tokenizer": getattr(bench_ctx, "tokenizer", None),
                "bench_model": getattr(bench_ctx, "model", None),
            },
            accelerator=getattr(bench_ctx, "accelerator", None),
        )
    except Exception:
        pass
    bench_ctx = None

    # ---- Final training run at max_micro (fresh model)
    torch.cuda.empty_cache()
    gc.collect()
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model, lora_cfg = maybe_apply_lora(model, cfg)

    accelerator = _make_accelerator(cfg, run_dir)

    ds_train, ds_eval, dl_train, dl_eval = build_loaders(task, tokenizer, cfg, micro_batch=max_micro)

    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        betas=(cfg.adam_beta1, cfg.adam_beta2),
        eps=cfg.adam_eps,
        weight_decay=cfg.weight_decay,
    )

    num_warmup = max(1, int(cfg.warmup_ratio * cfg.max_steps))
    sched = get_cosine_schedule_with_warmup(
        optim,
        num_warmup_steps=num_warmup,
        num_training_steps=cfg.max_steps,
    )

    model, optim, dl_train, sched = accelerator.prepare(model, optim, dl_train, sched)
    model.train()

    train_log = {"loss_last": None, "steps": cfg.max_steps}
    tokens_per_step = max_micro * accelerator.num_processes * cfg.max_seq_len
    t_train0 = time.perf_counter()

    it = iter(dl_train)
    for step in tqdm_if_main(accelerator, range(cfg.max_steps), desc="train"):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(dl_train)
            batch = next(it)
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        loss = out.loss
        accelerator.backward(loss)
        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            accelerator.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)

        if (step + 1) % 50 == 0 and accelerator.is_main_process:
            write_json(os.path.join(run_dir, "train_progress.json"), {"step": step + 1, "loss": float(loss.detach().cpu())})

        train_log["loss_last"] = float(loss.detach().cpu())

    accelerator.wait_for_everyone()
    t_train1 = time.perf_counter()
    train_s = t_train1 - t_train0
    tokens_total = tokens_per_step * cfg.max_steps
    train_tok_s = tokens_total / train_s if train_s > 0 else 0.0

    # ---- Save adapter (main process only)
    adapter_dir = os.path.abspath(os.path.join(run_dir, "adapter"))
    if accelerator.is_main_process:
        os.makedirs(adapter_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        while hasattr(unwrapped, "module"):
            unwrapped = unwrapped.module
        if cfg.peft_mode == "lora":
            # Ensure we save PEFT adapter files: adapter_config.json + adapter_model.*
            try:
                from peft import PeftModel
                is_peft = isinstance(unwrapped, PeftModel) or hasattr(unwrapped, "peft_config")
                if not is_peft:
                    raise RuntimeError(
                        f"Expected a PEFT model for LoRA save, got: {type(unwrapped)}"
                    )
                unwrapped.save_pretrained(adapter_dir)
            except Exception as e:
                # Do NOT silently fall back to pytorch_model.bin; that breaks lm-eval.
                write_json(os.path.join(run_dir, "adapter_save_error.json"), {
                    "error": str(e),
                    "unwrapped_type": str(type(unwrapped)),
                })
                raise
        else:
            # Full dense model save
            unwrapped.save_pretrained(adapter_dir)
        
        tokenizer.save_pretrained(adapter_dir)

    accelerator.wait_for_everyone()

    write_json(os.path.join(run_dir, "train_summary.json"), {
        "micro_batch": max_micro,
        "world_size": accelerator.num_processes,
        "seq_len": cfg.max_seq_len,
        "steps": cfg.max_steps,
        "train_time_s": train_s,
        "train_tokens_per_s": train_tok_s,
        "loss_last": train_log["loss_last"],
        "tokens_total": tokens_total,
    })

    # ---- HARD cleanup BEFORE lm-eval-harness loads a new model
    _hard_cuda_cleanup(
        {
            "model": model,
            "optim": optim,
            "sched": sched,
            "dl_train": dl_train,
            "dl_eval": dl_eval,
            "ds_train": ds_train,
            "ds_eval": ds_eval,
            "tokenizer": tokenizer,
        },
        accelerator=accelerator,
    )

    # ---- Tear down distributed process
    try:
        accelerator.wait_for_everyone()
    except Exception:
        pass

    if dist.is_available() and dist.is_initialized():
        # one last barrier at torch.distributed level
        dist.barrier()
        dist.destroy_process_group()

    # ---- Eval (lm-eval-harness)
    eval_out = {}
    if cfg.eval_use_lm_eval and accelerator.is_main_process:
        eval_path = os.path.join(run_dir, "lm_eval.json")
        try:
            # Sanity check: adapter_config.json must exist for LoRA
            if cfg.peft_mode == "lora":
                cfg_path = os.path.join(adapter_dir, "adapter_config.json")
                if not os.path.isfile(cfg_path):
                    write_json(os.path.join(run_dir, "lm_eval_adapter_missing.json"), {
                        "adapter_dir": adapter_dir,
                        "missing": "adapter_config.json",
                        "files": sorted(os.listdir(adapter_dir)) if os.path.isdir(adapter_dir) else None,
                    })
                    raise RuntimeError(f"adapter_config.json missing in {adapter_dir}")

            eval_out = run_lm_eval_harness(
            base_model_id=model_id,
            peft_adapter_path=adapter_dir if cfg.peft_mode == "lora" else None,
            tasks=[cfg.eval_task_name],
            out_json_path=eval_path,
            batch_size=cfg.eval_batch_size,
            device=eval_device,
            extra_model_args=None,
        )
        except torch.cuda.OutOfMemoryError as e:
            # Record GPU OOM and fall back to CPU so the run completes.
            write_json(os.path.join(run_dir, "lm_eval_oom.json"), {
                "device_requested": eval_device,
                "fallback_device": "cpu",
                "error": str(e),
            })
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
            eval_out = run_lm_eval_harness(
                base_model_id=model_id,
                peft_adapter_path=adapter_dir if cfg.peft_mode == "lora" else None,
                tasks=[cfg.eval_task_name],
                out_json_path=eval_path,
                batch_size="1" if cfg.eval_batch_size == "auto" else cfg.eval_batch_size,
                device="cpu",
                extra_model_args={"dtype": "float32"},
            )

    accelerator.wait_for_everyone()

    run_result = {
        "model_id": model_id,
        "task": task,
        "seq_len": cfg.max_seq_len,
        "peft_mode": cfg.peft_mode,
        "micro_batch_final": max_micro,
        "throughput_points": tp_metrics,
        "train_tokens_per_s": train_tok_s,
        "train_time_s": train_s,
        "loss_last": train_log["loss_last"],
        "lm_eval": eval_out,
        "adapter_dir": adapter_dir,
    }

    write_json(os.path.join(run_dir, "run_result.json"), run_result)
    return run_result
