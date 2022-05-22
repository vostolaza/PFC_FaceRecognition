import json
with open("train_set.json") as f:
        individuals = json.load(f)

for dir, set in individuals.items():
    print(dir)
    print(set)
    break