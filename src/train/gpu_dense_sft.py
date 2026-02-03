# src/train/gpu_dense_sft.py
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)

from src.data.phoenix_sft_tasks import TASKS, build_sft_features
from src.utils.run_logging import snapshot_env

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return default

def get_rank() -> int:
    return _env_int("RANK", 0)

def get_world_size() -> int:
    return _env_int("WORLD_SIZE", 1)

def is_main_process() -> bool:
    return get_rank() == 0

def _dist_info():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return local_rank, rank, world_size

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)

def cuda_peak_mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024**3)

@dataclass
class RunMeta:
    week: int
    run_kind: str  # bench|train
    group_id: str  # ties bench/train/eval for a model-task
    model_name: str
    task: str
    seq_len: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    max_steps: int
    learning_rate: float
    weight_decay: float
    warmup_steps: int
    precision: str
    world_size: int
    hostname: str
    gpu_name: str
    timestamp: float

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows: List[Dict[str, torch.Tensor]]):
        self.rows = rows
    def __len__(self) -> int:
        return len(self.rows)
    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        return self.rows[i]

def build_dataset(task: str, split: str, tokenizer, seq_len: int, max_examples: Optional[int]) -> SFTDataset:
    spec = TASKS[task]
    path, name = spec.dataset_id
    ds = load_dataset(path, name, split=split)

    if max_examples is not None:
        ds = ds.select(range(min(max_examples, len(ds))))

    rows: List[Dict[str, torch.Tensor]] = []
    for ex in ds:
        prompt, target = spec.formatter(ex)
        feats = build_sft_features(tokenizer, prompt, target, max_length=seq_len)
        rows.append({k: torch.tensor(v, dtype=torch.long) for k, v in feats.items()})
    return SFTDataset(rows)

def bench_one_step(
    model,
    seq_len: int,
    micro_bsz: int,
    warmup_steps: int,
    measured_steps: int,
    device: torch.device,
    lr: float,
) -> Dict[str, Any]:
    assert measured_steps >= 1
    torch.cuda.reset_peak_memory_stats()
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    vocab = model.get_input_embeddings().weight.shape[0]
    input_ids = torch.randint(0, vocab, (micro_bsz, seq_len), device=device)
    attention_mask = torch.ones((micro_bsz, seq_len), device=device)
    labels = input_ids.clone()

    for _ in range(warmup_steps):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(measured_steps):
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        out.loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    dt = (t1 - t0) / measured_steps
    return {
        "micro_bsz": micro_bsz,
        "seq_len": seq_len,
        "step_time_s": dt,
        "samples_per_s": micro_bsz / dt,
        "tokens_per_s": (micro_bsz * seq_len) / dt,
        "peak_mem_gb": cuda_peak_mem_gb(),
        "effective_global_bsz": micro_bsz * get_world_size(),
    }

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--group_id", type=str, required=True)
    p.add_argument("--model_name", type=str, required=True)
    p.add_argument("--task", type=str, choices=list(TASKS.keys()), required=True)

    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--per_device_train_batch_size", type=int, default=1)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--learning_rate", type=float, default=2e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=10)

    p.add_argument("--bf16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--attn_implementation", type=str, default=None)  # e.g. flash_attention_2
    p.add_argument("--deepspeed", type=str, default=None)

    p.add_argument("--max_train_examples", type=int, default=None)
    p.add_argument("--max_eval_examples", type=int, default=None)

    p.add_argument("--bench_only", action="store_true")
    p.add_argument("--bench_warmup_steps", type=int, default=1)
    p.add_argument("--bench_measured_steps", type=int, default=1)

    p.add_argument("--seed", type=int, default=1234)
    return p.parse_args()

def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    local_rank, rank, world_size = _dist_info()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        gpu_name = torch.cuda.get_device_name(local_rank)
    else:
        device = torch.device("cpu")
        gpu_name = "cpu"

    if rank == 0:
        print(f"[ddp] host={socket.gethostname()} world_size={world_size}")
    print(f"[ddp] rank={rank} local_rank={local_rank} device={device}", flush=True)

    # Tokenizer / model
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
        device_map=None,
    )

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    model.to(device)

    meta = RunMeta(
        week=4,
        run_kind="bench" if args.bench_only else "train",
        group_id=args.group_id,
        model_name=args.model_name,
        task=args.task,
        seq_len=args.seq_len,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        precision="bf16" if args.bf16 else "fp32",
        world_size=get_world_size(),
        hostname=platform.node(),
        gpu_name=gpu_name,
        timestamp=time.time(),
    )

    if is_main_process():
        write_json(run_dir / "run_meta.json", asdict(meta))
        snapshot_env(run_dir)

    if args.bench_only:
        bench = bench_one_step(
            model=model,
            seq_len=args.seq_len,
            micro_bsz=args.per_device_train_batch_size,
            warmup_steps=args.bench_warmup_steps,
            measured_steps=args.bench_measured_steps,
            device=device,
            lr=args.learning_rate,
        )
        if is_main_process():
            write_json(run_dir / "throughput.json", bench)
        return

    # Dataset
    train_split = TASKS[args.task].train_split
    eval_split = TASKS[args.task].eval_split
    train_ds = build_dataset(args.task, train_split, tok, args.seq_len, args.max_train_examples)
    eval_ds = build_dataset(args.task, eval_split, tok, args.seq_len, args.max_eval_examples)

    targs = TrainingArguments(
        output_dir=str(run_dir / "hf_out"),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=max(1, args.per_device_train_batch_size),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        logging_steps=10,
        save_steps=max(50, args.max_steps // 4),
        eval_steps=max(50, args.max_steps // 4),
        evaluation_strategy="steps",
        save_total_limit=2,
        report_to=[],
        deepspeed=args.deepspeed,
        dataloader_num_workers=2,
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tok,
    )

    torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()

    if is_main_process():
        summary = {
            "train_loss": float(train_result.training_loss) if train_result.training_loss is not None else None,
            "global_steps": int(trainer.state.global_step),
            "train_runtime_s": float(trainer.state.log_history[-1].get("train_runtime", 0.0)) if trainer.state.log_history else None,
            "peak_mem_gb": cuda_peak_mem_gb(),
            "effective_global_bsz": args.per_device_train_batch_size * get_world_size() * args.gradient_accumulation_steps,
        }
        write_json(run_dir / "train_summary.json", summary)
        write_json(run_dir / "checkpoint.json", {"hf_out_dir": str(run_dir / "hf_out")})

if __name__ == "__main__":
    main()
