import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import yaml
from datasets import load_dataset

# -------------------------
# Prompt templates (must match contract v1)
# -------------------------
BOOLQ_TMPL = """Passage:
{passage}

Question: {question}
Answer (yes/no):"""
HELLASWAG_TMPL = """Context:
{context}

Choose the best continuation:
A) {ending_a}
B) {ending_b}
C) {ending_c}
D) {ending_d}

Answer (A/B/C/D):"""
GSM8K_TMPL = """Problem:
{question}

Final answer (number only):"""

LETTER = ["A", "B", "C", "D"]

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def write_jsonl(path: str, rows: Iterable[dict]) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n

def normalize_gsm8k_answer(ans: str) -> str:
    """
    HF gsm8k 'answer' contains rationale + '#### <final>'.
    We extract final and normalize:
      - strip spaces
      - remove commas
      - keep leading '-' and digits; allow decimal point
    """
    if "####" in ans:
        ans = ans.split("####")[-1]
    ans = ans.strip().replace(",", "")
    # keep simple numeric tokenization
    # (Phoenix-like eval usually compares normalized strings)
    return ans

def deterministic_split_indices(ids: List[str], val_frac: float, seed: int) -> Tuple[set, set]:
    """
    Deterministic hash split: no RNG, stable across machines.
    Use sha1(id + seed) to bucket into [0,1).
    """
    import hashlib
    val = set()
    train = set()
    for _id in ids:
        key = f"{_id}:{seed}".encode("utf-8")
        hv = hashlib.sha1(key).hexdigest()
        u = int(hv[:8], 16) / 0xFFFFFFFF
        if u < val_frac:
            val.add(_id)
        else:
            train.add(_id)
    return train, val

def build_boolq(contract: dict, out_root: str) -> Dict[str, Dict]:
    ds = load_dataset("super_glue", "boolq")
    # official: train/validation
    def row(split, ex):
        _id = f"boolq::{split}::{ex['idx']}"
        prompt = BOOLQ_TMPL.format(passage=ex["passage"], question=ex["question"])
        completion = "yes" if ex["label"] else "no"
        return {"prompt": prompt, "completion": completion, "task": "boolq", "id": _id}

    task_dir = os.path.join(out_root, "boolq")
    os.makedirs(task_dir, exist_ok=True)

    train_path = os.path.join(task_dir, "train.jsonl")
    val_path = os.path.join(task_dir, "val.jsonl")

    n_train = write_jsonl(train_path, (row("train", ex) for ex in ds["train"]))
    n_val = write_jsonl(val_path, (row("validation", ex) for ex in ds["validation"]))

    return {
        "boolq": {
            "source": "super_glue/boolq",
            "splits": {
                "train": {"path": train_path, "count": n_train, "sha256": sha256_file(train_path)},
                "val": {"path": val_path, "count": n_val, "sha256": sha256_file(val_path)},
            },
        }
    }

def build_hellaswag(contract: dict, out_root: str) -> Dict[str, Dict]:
    ds = load_dataset("hellaswag")
    # official: train/validation
    def row(split, ex):
        _id = f"hellaswag::{split}::{ex['ind']}"
        prompt = HELLASWAG_TMPL.format(
            context=ex["ctx"],
            ending_a=ex["endings"][0],
            ending_b=ex["endings"][1],
            ending_c=ex["endings"][2],
            ending_d=ex["endings"][3],
        )
        completion = LETTER[int(ex["label"])]
        return {"prompt": prompt, "completion": completion, "task": "hellaswag", "id": _id}

    task_dir = os.path.join(out_root, "hellaswag")
    os.makedirs(task_dir, exist_ok=True)

    train_path = os.path.join(task_dir, "train.jsonl")
    val_path = os.path.join(task_dir, "val.jsonl")

    n_train = write_jsonl(train_path, (row("train", ex) for ex in ds["train"]))
    n_val = write_jsonl(val_path, (row("validation", ex) for ex in ds["validation"]))

    return {
        "hellaswag": {
            "source": "hellaswag",
            "splits": {
                "train": {"path": train_path, "count": n_train, "sha256": sha256_file(train_path)},
                "val": {"path": val_path, "count": n_val, "sha256": sha256_file(val_path)},
            },
        }
    }

def build_gsm8k(contract: dict, out_root: str) -> Dict[str, Dict]:
    seed = int(contract["randomness"]["seed"])
    val_frac = float(contract["randomness"]["gsm8k_val_frac"])

    ds = load_dataset("gsm8k", "main")
    # official: train/test; we create val slice deterministically from train
    train_ids = [f"gsm8k::train::{i}" for i in range(len(ds["train"]))]
    train_set, val_set = deterministic_split_indices(train_ids, val_frac=val_frac, seed=seed)

    def row(split, ex, i):
        _id = f"gsm8k::{split}::{i}"
        prompt = GSM8K_TMPL.format(question=ex["question"])
        completion = normalize_gsm8k_answer(ex["answer"])
        return {"prompt": prompt, "completion": completion, "task": "gsm8k", "id": _id}

    task_dir = os.path.join(out_root, "gsm8k")
    os.makedirs(task_dir, exist_ok=True)

    train_path = os.path.join(task_dir, "train.jsonl")
    val_path = os.path.join(task_dir, "val.jsonl")
    test_path = os.path.join(task_dir, "test.jsonl")

    def train_rows():
        for i, ex in enumerate(ds["train"]):
            _id = f"gsm8k::train::{i}"
            if _id in train_set:
                yield row("train", ex, i)

    def val_rows():
        for i, ex in enumerate(ds["train"]):
            _id = f"gsm8k::train::{i}"
            if _id in val_set:
                yield row("val", ex, i)

    def test_rows():
        for i, ex in enumerate(ds["test"]):
            yield row("test", ex, i)

    n_train = write_jsonl(train_path, train_rows())
    n_val = write_jsonl(val_path, val_rows())
    n_test = write_jsonl(test_path, test_rows())

    return {
        "gsm8k": {
            "source": "gsm8k/main",
            "val_policy": {"method": "deterministic_hash", "val_frac": val_frac, "seed": seed},
            "splits": {
                "train": {"path": train_path, "count": n_train, "sha256": sha256_file(train_path)},
                "val": {"path": val_path, "count": n_val, "sha256": sha256_file(val_path)},
                "test": {"path": test_path, "count": n_test, "sha256": sha256_file(test_path)},
            },
        }
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contract", required=True, help="Path to week2 contract YAML")
    ap.add_argument("--out_root", default="data/sft", help="Output root for JSONL files")
    ap.add_argument("--manifest_out", default="results/artifacts/data_manifest.json", help="Manifest path")
    args = ap.parse_args()

    with open(args.contract, "r", encoding="utf-8") as f:
        contract = yaml.safe_load(f)

    os.makedirs(os.path.dirname(args.manifest_out), exist_ok=True)
    os.makedirs(args.out_root, exist_ok=True)

    manifest = {
        "contract_path": args.contract,
        "prompt_template_version": contract["prompt_templates"]["version"],
        "datasets": {},
        "builder": {
            "script": "src/data/build_week2_sft.py",
        },
    }

    # Build tasks
    manifest["datasets"].update(build_boolq(contract, args.out_root))
    manifest["datasets"].update(build_hellaswag(contract, args.out_root))
    manifest["datasets"].update(build_gsm8k(contract, args.out_root))

    # Record library versions (important for reproducibility)
    import datasets as datasets_pkg
    import transformers as transformers_pkg
    manifest["versions"] = {
        "datasets": datasets_pkg.__version__,
        "transformers": transformers_pkg.__version__,
    }

    with open(args.manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"[OK] Wrote manifest: {args.manifest_out}")

if __name__ == "__main__":
    main()
