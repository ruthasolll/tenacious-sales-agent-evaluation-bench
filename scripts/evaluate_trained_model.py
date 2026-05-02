from transformers import AutoModelForCausalLM, AutoTokenizer
from unsloth import FastLanguageModel
import json

MODEL_PATH = "outputs"

model, tokenizer = FastLanguageModel.from_pretrained(MODEL_PATH)

def generate(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

with open("data/splits/dev.json") as f:
    data = json.load(f)

results = []

for t in data:
    prompt = t["input"]["company"] + " " + t["input"]["signal"]
    pred = generate(prompt)

    results.append({
        "task_id": t["task_id"],
        "prediction": pred
    })

print("DONE:", len(results))