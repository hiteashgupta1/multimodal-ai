import requests
import cv2
import numpy as np
import time
from config import HF_TOKEN, VISION_MODEL
from rag.memory_store import store_memory
from evaluation.log_metrics import log_metric
from experiments.experiment_tracker import log_experiment

def detect_objects(image_bytes):

    start_time = time.time()

    url = f"https://router.huggingface.co/hf-inference/models/{VISION_MODEL}"

    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "image/jpeg"
    }

    response = requests.post(url, headers=headers, data=image_bytes)

    if response.status_code != 200:
        print("VISION ERROR:", response.text)
        return None

    detections = response.json()

    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    detected_objects = []

    for obj in detections:
        box = obj["box"]
        label = obj["label"]
        score = round(obj["score"], 2)

        x1 = int(box["xmin"])
        y1 = int(box["ymin"])
        x2 = int(box["xmax"])
        y2 = int(box["ymax"])

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        text = f"{label} ({score})"

        cv2.putText(
            img,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

        detected_objects.append({
            "label": label,
            "confidence": score
        })

    # store_memory(f"Detected objects: {detected_objects}")
    
    latency = round(time.time() - start_time, 2)

    _, buffer = cv2.imencode(".jpg", img)

    # ✅ Calculate average confidence
    if len(detected_objects) > 0:
        avg_conf = round(
            sum(obj["confidence"] for obj in detected_objects) / len(detected_objects),
            2
        )
    else:
        avg_conf = 0.0

    # ✅ Log metrics
    log_metric(
        agent="vision",
        model_version="yolo_v1",
        confidence=avg_conf,
        object_count=len(detected_objects),
        latency=latency,
        hallucination=False
    )
    log_experiment(
        agent="vision",
        confidence=avg_conf,
        latency=latency,
        hallucination=False,
        model_version="yolo_v1"
    )

    return {
        "image": buffer.tobytes(),
        "objects": detected_objects,
        "latency": latency,
        "confidence": avg_conf
    }