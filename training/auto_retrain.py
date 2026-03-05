import json
import datetime
import pandas as pd
import os


FEEDBACK_FILE = "backend/experiments/feedback.csv"

def check_retrain():

    df = pd.read_csv(FEEDBACK_FILE)

    avg_rating = df["rating"].mean()

    if avg_rating < 3:

        print("Model quality dropped. Retraining...")

        os.system("python training/dataset_builder.py")

        os.system("python training/lora_train.py")