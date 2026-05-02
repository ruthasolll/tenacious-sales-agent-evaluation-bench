import json

def load_ids(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return set(t["task_id"] for t in data if "task_id" in t)

train = load_ids("data/splits/train.json")
dev = load_ids("data/splits/dev.json")
test = load_ids("data/splits/held_out.json")

print("train ∩ dev:", len(train & dev))
print("train ∩ test:", len(train & test))
print("dev ∩ test:", len(dev & test))