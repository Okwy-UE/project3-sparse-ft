# src/eval/eval_phoenix_tasks.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.data.phoenix_sft_tasks import TASKS, extract_gsm8k_final

def write_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True))

@torch.inference_mode()
def score_candidates(
    model,
    tok,
    prompts: List[str],
    candidates: List[List[str]],
    device: torch.device,
    max_length: int,
) -> List[int]:
    """
    For each prompt, pick candidate with highest conditional log-prob.
    candidates[i] = list of candidate strings (already include newline if you want).
    """
    # Flatten (prompt, cand) pairs
    flat_pairs: List[Tuple[int, str, str]] = []
    for i, p in enumerate(prompts):
        for c in candidates[i]:
            flat_pairs.append((i, p, c))

    # Batch over flattened pairs
    batch_size = 8
    scores = [0.0 for _ in flat_pairs]

    for s in range(0, len(flat_pairs), batch_size):
        chunk = flat_pairs[s:s+batch_size]
        texts = [p + c for (_, p, c) in chunk]

        enc = tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        ).to(device)

        # Also get prompt lengths to mask
        p_enc = tok(
            [p for (_, p, _) in chunk],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )

        out = model(**enc)
        logits = out.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]

        logp = torch.log_softmax(logits, dim=-1)
        token_logp = torch.gather(logp, 2, labels.unsqueeze(-1)).squeeze(-1)

        # mask: only score tokens after prompt length
        # prompt length in tokens for each item
        p_lens = (p_enc["attention_mask"].sum(dim=1) - 1).tolist()  # -1 because shifted
        for j in range(len(chunk)):
            start = max(int(p_lens[j]), 0)
            # sum logp from start onward, excluding padding
            attn = enc["attention_mask"][j, 1:]
            mask = (attn == 1)
            m2 = mask.clone()
            if start < m2.numel():
                m2[:start] = False
            score = float(token_logp[j][m2].sum().item())
            scores[s + j] = score

    # reduce: for each original i, pick argmax candidate
    best = [-1 for _ in prompts]
    best_score = [-1e30 for _ in prompts]

    idx = 0
    for i in range(len(prompts)):
        for c in candidates[i]:
            sc = scores[idx]
            if sc > best_score[i]:
                best_score[i] = sc
                best[i] = candidates[i].index(c)
            idx += 1

    return best

@torch.inference_mode()
def eval_boolq(model, tok, device, max_examples: Optional[int], max_length: int) -> Dict[str, Any]:
    spec = TASKS["boolq"]
    ds = load_dataset(*spec.dataset_id, split=spec.eval_split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    prompts, gold = [], []
    for ex in ds:
        p, _ = spec.formatter(ex)
        prompts.append(p)
        gold.append(0 if bool(ex["answer"]) else 1)  # 0=yes,1=no

    cands = [["yes\n", "no\n"] for _ in prompts]
    pred_idx = score_candidates(model, tok, prompts, cands, device, max_length=max_length)

    correct = 0
    for i, pi in enumerate(pred_idx):
        if pi == gold[i]:
            correct += 1
    return {"task": "boolq", "n": len(prompts), "accuracy": correct / max(1, len(prompts))}

@torch.inference_mode()
def eval_hellaswag(model, tok, device, max_examples: Optional[int], max_length: int) -> Dict[str, Any]:
    spec = TASKS["hellaswag"]
    ds = load_dataset(*spec.dataset_id, split=spec.eval_split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    prompts, gold = [], []
    for ex in ds:
        p, _ = spec.formatter(ex)
        prompts.append(p)
        gold.append(int(ex["label"]))

    cands = [["A\n", "B\n", "C\n", "D\n"] for _ in prompts]
    pred_idx = score_candidates(model, tok, prompts, cands, device, max_length=max_length)

    correct = sum(1 for i, pi in enumerate(pred_idx) if pi == gold[i])
    return {"task": "hellaswag", "n": len(prompts), "accuracy": correct / max(1, len(prompts))}

@torch.inference_mode()
def eval_gsm8k(model, tok, device, max_examples: Optional[int], max_new_tokens: int) -> Dict[str, Any]:
    spec = TASKS["gsm8k"]
    ds = load_dataset(*spec.dataset_id, split=spec.eval_split)
    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))

    correct = 0
    total = 0

    for ex in ds:
        prompt, _ = spec.formatter(ex)
        gold = extract_gsm8k_final(ex["answer"])
        if gold is None:
            continue

        enc = tok(prompt, return_tensors="pt").to(device)
        gen = model.generate(
            **enc,
            do_sample=False,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
        text = tok.decode(gen[0], skip_special_tokens=True)
        pred = extract_gsm8k_final(text)
        if pred is not None and pred.strip() == gold.strip():
            correct += 1
        total += 1

    return {"task": "gsm8k", "n": total, "accuracy": correct / max(1, total)}

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_or_path", required=True)
    ap.add_argument("--task", choices=list(TASKS.keys()), required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--attn_implementation", default=None)
    ap.add_argument("--max_eval_examples", type=int, default=None)
    ap.add_argument("--max_length", type=int, default=2048)
    ap.add_argument("--gsm8k_max_new_tokens", type=int, default=256)
    return ap.parse_args()

def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.bf16 else None

    tok = AutoTokenizer.from_pretrained(args.model_or_path, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_or_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation=args.attn_implementation,
    ).to(device)
    model.eval()

    if args.task == "boolq":
        m = eval_boolq(model, tok, device, args.max_eval_examples, args.max_length)
    elif args.task == "hellaswag":
        m = eval_hellaswag(model, tok, device, args.max_eval_examples, args.max_length)
    else:
        m = eval_gsm8k(model, tok, device, args.max_eval_examples, args.gsm8k_max_new_tokens)

    write_json(Path(args.out_json), m)

if __name__ == "__main__":
    main()
