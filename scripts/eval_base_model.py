import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

MODEL_NAME = "unsloth/qwen2.5-1.5b"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def generate(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=150)
    return tokenizer.decode(output[0], skip_special_tokens=True)

with open("data/splits/dev.json") as f:
    data = json.load(f)

results = []

for task in data:
    prompt = str(task["input"])
    output = generate(prompt)

    results.append({
        "task_id": task["task_id"],
        "output": output
    })

print(json.dumps(results[:3], indent=2))