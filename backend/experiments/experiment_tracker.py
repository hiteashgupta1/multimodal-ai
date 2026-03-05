import json
import os
from datetime import datetime

EXPERIMENT_FILE = "backend/experiments/experiments.json"

def log_experiment(agent, confidence, latency, hallucination, model_version="v2"):


    os.makedirs(os.path.dirname(EXPERIMENT_FILE), exist_ok=True)

    experiment = {
        "timestamp": str(datetime.now()),
        "agent": agent,
        "confidence": confidence,
        "latency": latency,
        "hallucination": hallucination,
        "model_version": model_version
    }

    if os.path.exists(EXPERIMENT_FILE):

        with open(EXPERIMENT_FILE, "r") as f:
            data = json.load(f)

    else:
        data = []

    data.append(experiment)

    with open(EXPERIMENT_FILE, "w") as f:
        json.dump(data, f, indent=2)
