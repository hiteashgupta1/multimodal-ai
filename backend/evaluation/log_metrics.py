import csv
import os
from datetime import datetime

METRICS_FILE = "experiments/metrics.csv"

def log_metric(agent, confidence, latency, hallucination, object_count=None, model_version="v1"):

    os.makedirs("experiments", exist_ok=True)

    file_exists = os.path.isfile(METRICS_FILE)

    with open(METRICS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "agent",
                "model_version",
                "confidence",
                "latency",
                "hallucination",
                "object_count"
            ])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            agent,
            model_version,
            confidence,
            latency,
            hallucination,
            object_count
        ])