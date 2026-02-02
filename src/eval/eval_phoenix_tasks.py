from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data.phoenix_sft_tasks import TASKS, format_example


def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True))
    tmp.replace(path)


@torch.no_grad()
def logprob_of_completion(model, tokenizer, prompt: str, completion: str, device: torch.device) -> float:
    """
    Computes sum log p(completion | prompt) (teacher-forced).
    Works for multi-token completions.
    """
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # No special tokens to avoid double BOS; we control exact strings
    p_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    c_ids = tokenizer(completion, add_special_tokens=False).input_ids

    input_ids = torch.tensor([p_ids + c_ids], device=device)
    # labels: ignore prompt tokens, score only completion tokens
    labels = torch.full_like(input_ids, -100)
    labels[0, len(p_ids):] = torch.tensor(c_ids, device=device)

    out = model(input_ids=input_ids, labels=labels)
    # Trainer loss is mean over scored tokens; convert to sum logprob:
    # loss = - (1/N) sum log p ; so sum log p = -loss*N
    n = max(1, len(c_ids))
    return float(-out.loss.item() * n)


def extract_gsm8k_final(s: str) -> str:
    # Prefer GSM8K "####" convention
    if "####" in s:
        tail = s.split("####")[-1]
        m = re.search(r"[-+]?\d[\d,]*\.?\d*", tail)
        if m:
            return m.group(0).replace(",", "").strip()
    # Fallback: last number in string
    nums = re.findall(r"[-+]?\d[\d,]*\.?\d*", s)
    if nums:
        return nums[-1].replace(",", "").strip()
    return ""


@torch.no_grad()
def eval_boolq(model, tok, max_examples: int | None, device: torch.device) -> Dict[str, Any]:
    ds = load_dataset(*TASKS["boolq"].dataset_id, split=TASKS["boolq"].eval_split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    for ex in ds:
        prompt, _ = format_example("boolq", ex)
        lp_yes = logprob_of_completion(model, tok, prompt, " yes", device)
        lp_no = logprob_of_completion(model, tok, prompt, " no", device)
        pred = (lp_yes > lp_no)
        if bool(ex["answer"]) == pred:
            correct += 1

    return {"n": len(ds), "accuracy": correct / max(1, len(ds))}


@torch.no_grad()
def eval_hellaswag(model, tok, max_examples: int | None, device: torch.device) -> Dict[str, Any]:
    ds = load_dataset(*TASKS["hellaswag"].dataset_id, split=TASKS["hellaswag"].eval_split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    for ex in ds:
        prompt, _ = format_example("hellaswag", ex)
        lps = []
        for opt in [" A", " B", " C", " D"]:
            lps.append(logprob_of_completion(model, tok, prompt, opt, device))
        pred = int(max(range(4), key=lambda i: lps[i]))
        if pred == int(ex["label"]):
            correct += 1
    return {"n": len(ds), "accuracy": correct / max(1, len(ds))}


@torch.no_grad()
def eval_gsm8k(model, tok, max_examples: int | None, device: torch.device) -> Dict[str, Any]:
    ds = load_dataset(*TASKS["gsm8k"].dataset_id, split=TASKS["gsm8k"].eval_split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    for ex in ds:
        prompt, _ = format_example("gsm8k", ex)
        gold = extract_gsm8k_final(ex["answer"])
        inputs = tok(prompt, return_tensors="pt").to(device)
        out = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tok.eos_token_id,
        )
        gen = tok.decode(out[0], skip_special_tokens=True)
        pred = extract_gsm8k_final(gen)
        if pred != "" and gold != "" and pred == gold:
            correct += 1

    return {"n": len(ds), "accuracy": correct / max(1, len(ds))}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_or_path", type=str, required=True, help="HF model name or local hf_out dir")
    p.add_argument("--task", type=str, choices=list(TASKS.keys()), required=True)
    p.add_argument("--out_json", type=str, required=True)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--max_eval_examples", type=int, default=None)
    p.add_argument("--attn_implementation", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    tok = AutoTokenizer.from_pretrained(args.model_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if args.bf16 else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_or_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        device_map=None,
    ).to(device)
    model.eval()

    t0 = time.time()
    if args.task == "boolq":
        res = eval_boolq(model, tok, args.max_eval_examples, device)
    elif args.task == "hellaswag":
        res = eval_hellaswag(model, tok, args.max_eval_examples, device)
    elif args.task == "gsm8k":
        res = eval_gsm8k(model, tok, args.max_eval_examples, device)
    else:
        raise ValueError(args.task)
    res["elapsed_s"] = time.time() - t0
    res["task"] = args.task
    res["model_or_path"] = args.model_or_path

    write_json(Path(args.out_json), res)


if __name__ == "__main__":
    main()
