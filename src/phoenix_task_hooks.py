
def gsm8k_prompt_completion_hook(x, **kwargs):

    q = x["question"].strip()

    a = x["answer"].strip()

    prompt = f"Question: {q}\nAnswer:"

    completion = f" {a}"

    return {"prompt": prompt, "completion": completion}



def boolq_prompt_completion_hook(x, **kwargs):

    passage = x["passage"].strip()

    question = x["question"].strip()

    label = "yes" if x["answer"] else "no"

    prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"

    completion = f" {label}"

    return {"prompt": prompt, "completion": completion}

