import requests
import base64
import time
from evaluation.log_metrics import log_metric
from experiments.experiment_tracker import log_experiment

COLAB_URL = "https://galeiform-cathleen-unloveably.ngrok-free.dev"

def generate_image(prompt):
    start = time.time()
    latency = round(time.time() - start, 2)

    # ✅ Log metrics
    log_metric(
        agent="image_gen",
        confidence=0.99,
        object_count=None,
        latency=latency,
        hallucination=False
    )

    log_experiment(
        agent="image_gen",
        confidence=0.99,
        latency=latency,
        hallucination=False,
        model_version="img_v1"
    )

    try:
        response = requests.post(
            f"{COLAB_URL}/generate",
            json={"prompt": prompt}
        )

        if response.status_code == 200:
            image_base64 = base64.b64encode(response.content).decode("utf-8")
            return image_base64, 0.85
        else:
            print("COLAB ERROR:", response.text)
            return None, 0.1

    except Exception as e:
        print("COLAB CONNECTION ERROR:", str(e))
        return None, 0.1
