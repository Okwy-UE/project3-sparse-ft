from __future__ import annotations
from typing import Dict, Tuple, Any, Optional

import re
from datasets import load_dataset, Dataset

# ---- Task -> HF dataset identifiers
HF_DATASETS = {
    "boolq": ("google/boolq", None),
    "hellaswag": ("Rowan/hellaswag", None),
    "gsm8k": ("gsm8k", "main"),
}

CHOICE_LETTERS = ["A", "B", "C", "D"]


def format_sft_example(task: str, ex: Dict[str, Any]) -> Tuple[str, str]:
    """
    Returns (prompt, completion) for SFT-style fine-tuning.
    We keep prompts simple + deterministic for cross-hardware parity.
    """
    if task == "boolq":
        passage = ex["passage"].strip()
        question = ex["question"].strip()
        ans = "yes" if bool(ex["answer"]) else "no"
        prompt = (
            "Answer the question with 'yes' or 'no'.\n\n"
            f"Passage:\n{passage}\n\n"
            f"Question: {question}\n"
            "Answer:"
        )
        completion = f" {ans}"
        return prompt, completion

    if task == "hellaswag":
        # HellaSwag fields can vary; Rowan/hellaswag includes ctx and endings
        ctx = (ex.get("ctx") or "").strip()
        endings = ex.get("endings")
        if endings is None:
            # some variants have 'endings' nested in another structure
            endings = ex.get("endings", ["", "", "", ""])
        label = int(ex["label"])
        prompt = (
            "Choose the best ending for the context. Answer with A, B, C, or D.\n\n"
            f"Context:\n{ctx}\n\n"
            + "\n".join([f"{CHOICE_LETTERS[i]}) {endings[i]}" for i in range(4)])
            + "\nAnswer:"
        )
        completion = f" {CHOICE_LETTERS[label]}"
        return prompt, completion

    if task == "gsm8k":
        q = ex["question"].strip()
        a = ex["answer"].strip()
        prompt = (
            "Solve the problem. Show reasoning. Give the final answer after '####'.\n\n"
            f"Question: {q}\n"
            "Answer:"
        )
        completion = f" {a}"
        return prompt, completion

    raise ValueError(f"Unknown task: {task}")


def load_splits(task: str) -> Tuple[Dataset, Dataset]:
    ds_name, ds_cfg = HF_DATASETS[task]
    if ds_cfg is None:
        ds = load_dataset(ds_name)
    else:
        ds = load_dataset(ds_name, ds_cfg)

    # normalize to train/validation
    train = ds["train"]
    if "validation" in ds:
        val = ds["validation"]
    elif "test" in ds:
        val = ds["test"]
    else:
        raise ValueError(f"No validation/test split found for task={task}")

    return train, val


def build_sft_dataset(task: str, split: str, max_samples: Optional[int] = None, seed: int = 1337) -> Dataset:
    train, val = load_splits(task)
    base = train if split == "train" else val

    if max_samples is not None:
        base = base.shuffle(seed=seed).select(range(min(max_samples, len(base))))

    def _map(ex):
        p, c = format_sft_example(task, ex)
        return {"prompt": p, "completion": c}

    return base.map(_map, remove_columns=base.column_names)


def make_tokenize_fn(tokenizer, max_seq_len: int):
    # LLaMA-family tokenizers often have no pad_token by default
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tok(ex):
        prompt = ex["prompt"]
        completion = ex["completion"]

        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        full = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]

        # Add EOS if not present
        if len(full) == 0 or full[-1] != tokenizer.eos_token_id:
            full = full + [tokenizer.eos_token_id]

        labels = full[:]
        # mask prompt tokens
        for i in range(min(len(prompt_ids), len(labels))):
            labels[i] = -100

        # pad/truncate to fixed 2048
        if len(full) > max_seq_len:
            full = full[:max_seq_len]
            labels = labels[:max_seq_len]

        attn = [1] * len(full)

        pad_len = max_seq_len - len(full)
        if pad_len > 0:
            full = full + [tokenizer.pad_token_id] * pad_len
            attn = attn + [0] * pad_len
            labels = labels + [-100] * pad_len

        return {"input_ids": full, "attention_mask": attn, "labels": labels}

    return tok
