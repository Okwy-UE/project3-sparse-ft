# src/data/phoenix_sft_tasks.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable
import re

@dataclass(frozen=True)
class TaskSpec:
    # (path, name) for datasets.load_dataset(path, name)
    dataset_id: Tuple[str, Optional[str]]
    train_split: str
    eval_split: str
    # example -> (prompt, target)
    formatter: Callable[[Dict[str, Any]], Tuple[str, str]]

def _boolq_formatter(ex: Dict[str, Any]) -> Tuple[str, str]:
    passage = ex["passage"].strip()
    question = ex["question"].strip()
    ans = "yes" if bool(ex["answer"]) else "no"
    prompt = (
        "### Passage:\n"
        f"{passage}\n\n"
        "### Question:\n"
        f"{question}\n\n"
        "### Answer (yes/no):\n"
    )
    target = f"{ans}\n"
    return prompt, target

def _hellaswag_formatter(ex: Dict[str, Any]) -> Tuple[str, str]:
    # Rowan/hellaswag: ctx + endings + label (string "0"-"3")
    ctx = ex["ctx"].strip()
    endings = ex.get("endings", [])
    endings = [e.strip() for e in endings]
    label = int(ex["label"]) if str(ex.get("label", "")).strip() != "" else -1
    letters = ["A", "B", "C", "D"]
    opts = "\n".join([f"{letters[i]}) {endings[i]}" for i in range(min(4, len(endings)))])
    prompt = (
        "### Context:\n"
        f"{ctx}\n\n"
        "### Choose the best ending:\n"
        f"{opts}\n\n"
        "### Answer (A/B/C/D):\n"
    )
    target = f"{letters[label]}\n" if 0 <= label < 4 else "A\n"
    return prompt, target

def _gsm8k_formatter(ex: Dict[str, Any]) -> Tuple[str, str]:
    # openai/gsm8k main: answer contains rationale + "#### <final>"
    q = ex["question"].strip()
    a = ex["answer"].strip()
    prompt = (
        "### Problem:\n"
        f"{q}\n\n"
        "### Solution:\n"
    )
    # Train on full rationale + final, to match common GSM8K SFT.
    target = f"{a}\n"
    return prompt, target

TASKS: Dict[str, TaskSpec] = {
    "boolq": TaskSpec(
        dataset_id=("google/boolq", None),
        train_split="train",
        eval_split="validation",
        formatter=_boolq_formatter,
    ),
    "hellaswag": TaskSpec(
        dataset_id=("Rowan/hellaswag", None),
        train_split="train",
        eval_split="validation",
        formatter=_hellaswag_formatter,
    ),
    "gsm8k": TaskSpec(
        dataset_id=("openai/gsm8k", "main"),
        train_split="train",
        eval_split="test",
        formatter=_gsm8k_formatter,
    ),
}

def build_sft_features(tokenizer, prompt: str, target: str, max_length: int) -> Dict[str, Any]:
    """
    SFT: loss only on target tokens.
    Pads/truncates to fixed max_length (Week4 requires 2048).
    """
    # Ensure pad token exists (LLaMA-family often lacks it)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + target, add_special_tokens=False).input_ids
    # Add EOS
    full_ids = (full_ids + [tokenizer.eos_token_id])[:max_length]

    # Labels: mask prompt portion
    labels = [-100] * len(full_ids)
    p = min(len(prompt_ids), len(full_ids))
    for i in range(p, len(full_ids)):
        labels[i] = full_ids[i]

    # Pad
    pad_len = max_length - len(full_ids)
    input_ids = full_ids + [tokenizer.pad_token_id] * pad_len
    attention_mask = [1] * len(full_ids) + [0] * pad_len
    labels = labels + [-100] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

_FINAL_NUM_RE = re.compile(r"####\s*([-+]?\d+)")
_LAST_INT_RE = re.compile(r"([-+]?\d+)\s*$")

def extract_gsm8k_final(text: str) -> Optional[str]:
    """
    Prefer GSM8K canonical '#### <int>' format; else last integer in string.
    """
    m = _FINAL_NUM_RE.search(text)
    if m:
        return m.group(1)
    m = _LAST_INT_RE.search(text.strip())
    if m:
        return m.group(1)
    return None
