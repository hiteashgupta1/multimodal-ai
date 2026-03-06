import os
import torch
import fitz
import time
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from agents.rag_agent import retrieve_context
from rag.memory_store import store_memory, retrieve_memory
from evaluation.log_metrics import log_metric
from experiments.experiment_tracker import log_experiment

BASE_MODEL = "distilgpt2"


# ----------------------------
# Resolve project root
# ----------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# agents → backend → project root
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

def get_latest_model():

    versions = []

    if not os.path.exists(MODEL_DIR):
        print("Models directory not found")
        return None

    for name in os.listdir(MODEL_DIR):

        if name.startswith("orchestrator_v"):

            try:
                v = int(name.replace("orchestrator_v", ""))
                versions.append(v)
            except:
                pass

    if not versions:
        print("No trained models found")
        return None

    latest = max(versions)

    latest_path = os.path.join(MODEL_DIR, f"orchestrator_v{latest}")

    print("Latest model detected:", latest_path)

    return latest_path


print("Project root:", PROJECT_ROOT)
print("Model directory:", MODEL_DIR)


# Load Adapter


tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL)

ADAPTER_PATH = get_latest_model()

if ADAPTER_PATH and os.path.exists(os.path.join(ADAPTER_PATH, "adapter_config.json")):

    print("Loading LoRA adapter:", ADAPTER_PATH)

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

else:

    print("No LoRA adapter found — using base model")

    model = base_model

model.eval()


def extract_pdf_text(pdf_bytes):
    text = ""
    pdf = fitz.open(stream=pdf_bytes, filetype="pdf")
    for page in pdf:
        text += page.get_text()
    return text


def detect_hallucination(original_text, summary):

    original_words = set(original_text.lower().split())
    summary_words = summary.lower().split()

    unseen_words = [w for w in summary_words if w not in original_words]

    hallucination_ratio = len(unseen_words) / max(1, len(summary_words))

    return hallucination_ratio > 0.4, round(hallucination_ratio, 2)


def calculate_confidence(original_text, summary):

    original_length = len(original_text.split())
    summary_length = len(summary.split())

    if original_length == 0:
        return 0.1

    compression_ratio = summary_length / original_length

    score = 1 - abs(compression_ratio - 0.3)

    return round(max(0.1, min(score, 1.0)), 2)


def summarize(input_text=None, pdf_bytes=None):

    start = time.time()
    # Extract input
    if pdf_bytes:
        text = extract_pdf_text(pdf_bytes)
    else:
        text = input_text

    if not text:
        return {
            "summary": "No input provided.",
            "confidence": 0.0,
            "hallucinated": False,
            "hallucination_ratio": 0.0
        }

    # RAG context
    context = retrieve_context(text)
    combined_input = context + "\n\nSummarize:\n" + text[:1000]


    inputs = tokenizer(combined_input, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=120,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract summary cleanly
    if "Summarize:" in decoded:
        summary = decoded.split("Summarize:")[-1].strip()
    else:
        summary = decoded.strip()

    # Compute metrics
    confidence = calculate_confidence(text, summary)
    hallucinated, ratio = detect_hallucination(text, summary)

    if confidence > 0.7 and not hallucinated:
        store_memory(text)
        store_memory(summary)

    latency = round(time.time() - start, 2)

    # ✅ Log metrics
    log_metric(
        agent="summarizer",
        model_version="orchestrator_v3",
        confidence=confidence,
        object_count=None,
        latency=latency,
        hallucination=hallucinated
    )

    log_experiment(
        agent="summarizer",
        confidence=confidence,
        latency=latency,
        hallucination=hallucinated,
        model_version="orchestrator_v3"
    )

    return {
        "summary": summary,
        "confidence": confidence,
        "hallucinated": hallucinated,
        "hallucination_ratio": ratio
    }
