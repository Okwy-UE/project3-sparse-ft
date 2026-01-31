def gsm8k_prompt_completion_read_hook(x, data_keys, **kwargs):
    q = x[data_keys["question_key"]].strip()
    a = x[data_keys["answer_key"]].strip()
    prompt = f"Question: {q}\nAnswer:"
    completion = f" {a}"
    # Semantic Data Array: list of turns with type + content
    return [
        {"type": "prompt", "content": [{"text": prompt}]},
        {"type": "completion", "content": [{"text": completion}]},
    ]

def boolq_prompt_completion_hook(x, **kwargs):
    passage = x["passage"].strip()
    question = x["question"].strip()
    label = "yes" if x["answer"] else "no"
    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
    completion = f" {label}"
    return {"prompt": prompt, "completion": completion}

