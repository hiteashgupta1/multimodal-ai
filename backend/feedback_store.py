import csv
import os
from datetime import datetime

FEEDBACK_FILE = "experiments/feedback.csv"

def save_feedback(input_text, rating, comment):

    os.makedirs("experiments", exist_ok=True)

    file_exists = os.path.isfile(FEEDBACK_FILE)

    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Write header only if file does not exist
        if not file_exists:
            writer.writerow(["timestamp", "input_text", "rating", "comment"])

        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_text,
            rating,
            comment
        ])