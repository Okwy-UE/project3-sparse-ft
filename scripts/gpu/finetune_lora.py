
import argparse, json, os, time

from datasets import load_dataset

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling

from peft import LoraConfig, get_peft_model



# Phoenix baseline target modules and hyperparams (baseline track)

PHOENIX_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]



def gsm8k_prompt(example):

    q = example["question"].strip()

    a = example["answer"].strip()

    prompt = f"Question: {q}\nAnswer:"

    completion = f" {a}"

    return {"text": prompt + completion}



def boolq_prompt(example):

    passage = example["passage"].strip()

    question = example["question"].strip()

    label = "yes" if example["answer"] else "no"

    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

    completion = f" {label}"

    return {"text": prompt + completion}



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)

    ap.add_argument("--task", choices=["gsm8k", "boolq"], required=True)

    ap.add_argument("--out", required=True)

    ap.add_argument("--max_seq_len", type=int, default=2048)

    ap.add_argument("--lr", type=float, default=2e-4)

    ap.add_argument("--epochs", type=float, default=0.1)

    ap.add_argument("--per_device_train_bs", type=int, default=1)

    ap.add_argument("--grad_accum", type=int, default=8)

    ap.add_argument("--seed", type=int, default=1)

    ap.add_argument("--lora_r", type=int, default=16)

    ap.add_argument("--lora_alpha", type=int, default=64)

    args = ap.parse_args()



    os.makedirs(args.out, exist_ok=True)

    with open(os.path.join(args.out, "args.json"), "w") as f:

        json.dump(vars(args), f, indent=2)



    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    if tok.pad_token is None:

        tok.pad_token = tok.eos_token



    model = AutoModelForCausalLM.from_pretrained(

        args.model,

        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,

        device_map="auto",

    )



    lora_cfg = LoraConfig(

        r=args.lora_r,

        lora_alpha=args.lora_alpha,

        target_modules=PHOENIX_TARGET_MODULES,

        lora_dropout=0.0,

        bias="none",

        task_type="CAUSAL_LM",

    )

    model = get_peft_model(model, lora_cfg)



    if args.task == "gsm8k":

        ds = load_dataset("openai/gsm8k", "main")

        ds_train = ds["train"].map(gsm8k_prompt, remove_columns=ds["train"].column_names)

    else:

        ds = load_dataset("google/boolq")

        ds_train = ds["train"].map(boolq_prompt, remove_columns=ds["train"].column_names)



    def tokenize(batch):

        return tok(batch["text"], truncation=True, max_length=args.max_seq_len)



    ds_train = ds_train.map(tokenize, batched=True, remove_columns=["text"])

    collator = DataCollatorForLanguageModeling(tok, mlm=False)



    targs = TrainingArguments(

        output_dir=args.out,

        per_device_train_batch_size=args.per_device_train_bs,

        gradient_accumulation_steps=args.grad_accum,

        learning_rate=args.lr,

        num_train_epochs=args.epochs,

        logging_steps=10,

        save_steps=200,

        bf16=torch.cuda.is_available(),

        seed=args.seed,

        report_to="none",

    )



    trainer = Trainer(model=model, args=targs, train_dataset=ds_train, data_collator=collator)



    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t0 = time.time()

    trainer.train()

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    t1 = time.time()



    metrics = {

        "wallclock_sec": t1 - t0,

        "train_samples": len(ds_train),

        "effective_global_batch": args.per_device_train_bs * args.grad_accum,

        "max_seq_len": args.max_seq_len,

        "lr": args.lr,

    }

    with open(os.path.join(args.out, "train_metrics.json"), "w") as f:

        json.dump(metrics, f, indent=2)



if __name__ == "__main__":

    main()

