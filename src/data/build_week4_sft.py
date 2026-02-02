# src/data/build_week4_sft.py
from __future__ import annotations
import argparse, os
from datasets import load_dataset, DatasetDict

def build_boolq() -> DatasetDict:
    ds = load_dataset("google/boolq")  # train/validation :contentReference[oaicite:6]{index=6}
    def map_ex(ex):
        prompt = (
            "Passage:\n" + ex["passage"].strip() + "\n\n"
            "Question: " + ex["question"].strip() + "\n"
            "Answer (yes/no):"
        )
        completion = " yes" if ex["answer"] else " no"
        return {"prompt": prompt, "completion": completion}
    return ds.map(map_ex, remove_columns=ds["train"].column_names)

def build_hellaswag() -> DatasetDict:
    ds = load_dataset("Rowan/hellaswag")  # train/validation/test :contentReference[oaicite:7]{index=7}
    def map_ex(ex):
        # Canonical context is ctx_a + ctx_b in this dataset
        ctx = (ex.get("ctx_a","") + ex.get("ctx_b","")).strip()
        endings = ex["endings"]
        label = int(ex["label"])
        prompt = "Context:\n" + ctx + "\n\nContinuation:"
        completion = " " + endings[label].strip()
        return {"prompt": prompt, "completion": completion}
    return ds.map(map_ex, remove_columns=ds["train"].column_names)

def build_gsm8k() -> DatasetDict:
    ds = load_dataset("openai/gsm8k", "main")  # train/test :contentReference[oaicite:8]{index=8}
    # lm-eval GSM8K parsing usually expects the "####" delimiter;
    # keeping the original answer string preserves that.
    def map_ex(ex):
        prompt = "Problem:\n" + ex["question"].strip() + "\n\nSolution:\n"
        completion = ex["answer"]
        # Ensure a leading space so tokenizer separation is stable
        if not completion.startswith(" "):
            completion = " " + completion
        return {"prompt": prompt, "completion": completion}
    return ds.map(map_ex, remove_columns=ds["train"].column_names)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    builders = {
        "boolq": build_boolq,
        "hellaswag": build_hellaswag,
        "gsm8k": build_gsm8k,
    }
    for name, fn in builders.items():
        out = os.path.join(args.out_dir, name)
        ds = fn()
        ds.save_to_disk(out)
        print(f"[OK] wrote {name} -> {out}")

if __name__ == "__main__":
    main()
