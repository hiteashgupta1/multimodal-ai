from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

AGENTS = {
    "summarizer": ["summarize", "summary", "document", "pdf"],
    "vision": ["detect", "object", "image"],
    "tts": ["speech", "audio", "voice"],
    "text2image": ["generate image", "create image", "draw"]
}

def route_query(text):

    text = text.lower()

    scores = {}

    for agent, keywords in AGENTS.items():
        scores[agent] = sum(k in text for k in keywords)

    return max(scores, key=scores.get)