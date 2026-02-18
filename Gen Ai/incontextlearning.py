import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, pipeline

################
# Model setup
################

model = T5ForConditionalGeneration.from_pretrained(
    "google/flan-t5-base",
    torch_dtype="auto"
)
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-base")

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=10,
    num_beams=1,
    do_sample=False,
    temperature=0.7,
    device = "cpu"
)

################
# Dataset
################

TEST = [
    ("This cookie is amazing. I want another one.", "positive"),
    ("It was fine, nothing special.", "neutral"),
    ("Terrible cookie. I threw it away.", "negative"),
    ("Not bad, but not great either.", "neutral"),
    ("Absolutely loved it.", "positive"),
    ("I regret buying this.", "negative"),
]

#Allowed labels:
LABELS = ["negative", "neutral", "positive"]

################
# Functions for evaluation
################

def normalize_label(text: str) -> str:
    #delete leading and trailing whitespace and convert to lowercase
    t = text.lower().strip(" \n\t.,:;!?'\"")
    # Iterate through all allowed LABELS and return the first one that appears in the text
    for lab in LABELS:
        if lab in t:
            return lab
    return "unknown"

def build_prompt(text: str, condition: str, k: int = 3):
    if condition == "zero":
        return zero_shot_prompt(text)

    if condition == "good":
        return few_shot_prompt(text, demos=GOOD_DEMOS, k=k)

    if condition == "poison":
        return few_shot_prompt(text, demos=POISON_DEMOS, k=k)

    if condition == "drift":
        return few_shot_prompt_format_drift(text)

    if condition == "best":
        return few_shot_prompt_best_practice(text)

    raise ValueError(f"Unknown condition: {condition}")

def evaluate(name: str, condition: str, k: int = 3):
    correct = 0
    print(f"\n=== {name} ===")

    for text, gold in TEST:
        prompt = build_prompt(text, condition=condition, k=k)
        out = generator(prompt)[0]["generated_text"]
        pred = normalize_label(out)
        correct += int(pred == gold)

        print(f"Text: {text}")
        print(f"Gold: {gold:8s} Pred: {pred:8s} Raw: {out}\n") #:8s means to format as string with at least 8 characters, padding with spaces if needed 

    acc = correct / len(TEST)
    print(f"Accuracy: {correct}/{len(TEST)} = {acc:.2%}") #format as percentage with 2 decimal places   
    return acc

###################
# Prompt templates
###################

def zero_shot_prompt(text: str) -> str:
    return (
        "Classify the sentiment of the text as negative, neutral, or positive.\n"
        f"Text: {text}\n"
        "Sentiment:"
    )


evaluate("1) Zero-shot", condition="zero")

###We will further develop this code in the next class 