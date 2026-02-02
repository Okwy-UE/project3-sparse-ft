# scripts/gpu/bench_throughput_one_step.py
from __future__ import annotations
import argparse, json, os, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--seq_len", type=int, default=2048)
    ap.add_argument("--batch_sizes", type=str, default="1,2,4")
    ap.add_argument("--warmup_steps", type=int, default=3)
    ap.add_argument("--timed_steps", type=int, default=1)
    ap.add_argument("--dtype", type=str, default="bf16", choices=["bf16","fp16"])
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    ap.add_argument("--out_json", required=True)
    return ap.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        device_map={"": 0},
    ).to(device)
    model.train()

    if args.use_lora:
        lconf = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=args.target_modules.split(","),
        )
        model = get_peft_model(model, lconf)

    # only optimize trainable params (LoRA adapters if enabled)
    optim = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    vocab = tok.vocab_size
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    results = {
        "model_id": args.model_id,
        "seq_len": args.seq_len,
        "dtype": args.dtype,
        "use_lora": bool(args.use_lora),
        "points": [],
    }

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        point = {"batch_size": bs, "status": "ok"}
        try:
            input_ids = torch.randint(low=0, high=vocab, size=(bs, args.seq_len), device=device)
            labels = input_ids.clone()

            # warmup
            for _ in range(args.warmup_steps):
                optim.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                optim.step()
            torch.cuda.synchronize()

            # timed
            t0 = time.time()
            for _ in range(args.timed_steps):
                optim.zero_grad(set_to_none=True)
                out = model(input_ids=input_ids, labels=labels)
                out.loss.backward()
                optim.step()
            torch.cuda.synchronize()
            t1 = time.time()

            dt = max(t1 - t0, 1e-9)
            point["elapsed_sec"] = dt
            point["samples_per_sec"] = (bs * args.timed_steps) / dt
            point["tokens_per_sec"] = (bs * args.seq_len * args.timed_steps) / dt
            point["peak_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)

        except torch.cuda.OutOfMemoryError:
            point["status"] = "oom"
            point["peak_mem_gb"] = torch.cuda.max_memory_allocated() / (1024**3)
        results["points"].append(point)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[OK] wrote {args.out_json}")

if __name__ == "__main__":
    main()
