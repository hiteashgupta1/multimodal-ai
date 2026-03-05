import pandas as pd
import json
import os

def build_dataset():

    if not os.path.exists("backend/experiments/feedback.csv"):
        print("No feedback data")
        return

    fb = pd.read_csv("backend/experiments/feedback.csv")

    # Keep only good rated samples
    good = fb[fb["rating"] >= 4]

    dataset = []

    for _, row in good.iterrows():
        dataset.append({
            "instruction": row["input_text"],
            "output": row["comment"] if pd.notna(row["comment"]) else ""
        })

    os.makedirs("training/data", exist_ok=True)

    with open("training/data/instruction_dataset.json", "w") as f:
        json.dump(dataset, f, indent=4)

    print("Dataset built with", len(dataset), "samples")

if __name__ == "__main__":
    build_dataset()