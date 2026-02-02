from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

# HF dataset IDs (Phoenix-aligned task suite)
BOOLQ_DATASET = ("google/boolq", None)
HELLASWAG_DATASET = ("Rowan/hellaswag", "default")
GSM8K_DATASET = ("openai/gsm8k", "main")


@dataclass(frozen=True)
class TaskSpec:
    name: str
    dataset_id: Tuple[str, Optional[str]]
    train_split: str
    eval_split: str


TASKS: Dict[str, TaskSpec] = {
    "boolq": TaskSpec("boolq", BOOLQ_DATASET, "train", "validation"),
    "hellaswag": TaskSpec("hellaswag", HELLASWAG_DATASET, "train", "validation"),
    "gsm8k": TaskSpec("gsm8k", GSM8K_DATASET, "train", "test"),
}


def format_example(task: str, ex: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (prompt, answer_text). The training text will be prompt+answer.
    We keep answers short for BoolQ/HellaSwag to match classifier-style scoring.
    """
    task = task.lower()
    if task == "boolq":
        # ex: question, passage, answer (bool)
        prompt = (
            f"Passage: {ex['passage']}\n"
            f"Question: {ex['question']}\n"
            f"Answer:"
        )
        answer = " yes" if bool(ex["answer"]) else " no"
        return prompt, answer

    if task == "hellaswag":
        # ex: ctx, endings (list[str]), label (int 0..3)
        endings = ex["endings"]
        prompt = (
            f"Context: {ex['ctx']}\n"
            "Choose the most plausible ending:\n"
            f"A) {endings[0]}\n"
            f"B) {endings[1]}\n"
            f"C) {endings[2]}\n"
            f"D) {endings[3]}\n"
            "Answer:"
        )
        label = int(ex["label"])
        answer = f" {'ABCD'[label]}"
        return prompt, answer

    if task == "gsm8k":
        # ex: question, answer (contains reasoning + #### final)
        prompt = f"Question: {ex['question']}\nAnswer:"
        answer = "\n" + ex["answer"].strip()
        return prompt, answer

    raise ValueError(f"Unknown task: {task}")


def build_sft_features(
    tokenizer,
    prompt: str,
    answer: str,
    max_length: int,
) -> Dict[str, Any]:
    """
    Tokenize to fixed max_length with padding='max_length' so throughput is comparable.
    Labels: ignore prompt tokens + pad tokens.
    """
    if tokenizer.pad_token_id is None:
        # LLaMA-family frequently has no pad token by default
        tokenizer.pad_token = tokenizer.eos_token

    # Tokenize prompt and full text without special tokens (we control BOS/EOS)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    full_ids = tokenizer(prompt + answer, add_special_tokens=False).input_ids

    # Add BOS if model uses it; always terminate with EOS if available.
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    input_ids = bos + full_ids + eos
    prompt_len = len(bos) + len(prompt_ids)  # prompt length within input_ids

    # Truncate
    input_ids = input_ids[:max_length]
    # Pad
    attn_mask = [1] * len(input_ids)
    if len(input_ids) < max_length:
        pad_len = max_length - len(input_ids)
        input_ids = input_ids + [tokenizer.pad_token_id] * pad_len
        attn_mask = attn_mask + [0] * pad_len

    # Labels: copy input_ids, but mask prompt portion and pad
    labels = input_ids.copy()
    for i in range(min(prompt_len, max_length)):
        labels[i] = -100
    for i, m in enumerate(attn_mask):
        if m == 0:
            labels[i] = -100

    return {
        "input_ids": input_ids,
        "attention_mask": attn_mask,
        "labels": labels,
    }
