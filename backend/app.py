from fastapi import FastAPI, UploadFile, File, Form, Body
from agents.summarizer import summarize
from agents.vision import detect_objects
from agents.tts import text_to_speech
from fastapi.responses import JSONResponse
from fastapi.responses import Response
from agents.text2image import generate_image
from agents.feedback import log_feedback
from agents.orchestrator import smart_orchestrate
from agents.router_agent import route_query
from pydantic import BaseModel
from feedback_store import save_feedback
import sqlite3


app = FastAPI()

@app.post("/query")
async def query_api(text: str = Form(None), file: UploadFile = File(None)):

    agent = route_query(text)

    if agent == "summarizer":
        return summarize(input_text=text)

    if agent == "vision":
        image_bytes = await file.read()
        return detect_objects(image_bytes)

    if agent == "tts":
        return text_to_speech(text)

    if agent == "text2image":
        return generate_image(text)

@app.post("/summarize")
async def summarize_api(
    text: str = Form(None),
    file: UploadFile = File(None)
):

    if file:
        pdf_bytes = await file.read()
        summary = summarize(pdf_bytes=pdf_bytes)

    elif text:
        summary = summarize(input_text=text)

    else:
        return {"error": "No input provided"}

    return summary

@app.post("/detect")
async def detect_api(image: UploadFile = File(...)):

    image_bytes = await image.read()

    result = detect_objects(image_bytes)

    if not result:
        return {"error": "Detection failed"}

    return {
        "image": result["image"].hex(),
        "objects": result["objects"],
        "latency": result["latency"]
    }

from fastapi.responses import Response

@app.post("/text-to-speech")
async def tts(text: str):
    audio = text_to_speech(text)

    if audio:
        return Response(content=audio, media_type="audio/wav")
    else:
        return {"error": "TTS failed"}


@app.post("/generate-image")
async def generate(prompt: str):

    img = generate_image(prompt)

    if img:
        return {"image": img}
    else:
        return {"error": "Image generation failed"}

@app.post("/analyze")
async def analyze(
    text: str = Form(None),
    image: UploadFile = File(None),
    image_prompt: str = Form(None)
):

    image_bytes = None

    if image:
        image_bytes = await image.read()

    results, agents_used = orchestrate(
        text=text,
        image=image_bytes,
        generate_img_prompt=image_prompt
    )

    final_response = {
        "analysis": results,
        "agents_used": agents_used
    }

    return final_response


class FeedbackRequest(BaseModel):
    input_text: str | None = None
    rating: int
    comment: str | None = None
    agent: str
@app.post("/feedback")
async def submit_feedback(data: FeedbackRequest):

    # If empty, replace with placeholder
    input_text = data.input_text if data.input_text else "N/A"
    comment = data.comment if data.comment else ""

    # Log feedback (your existing function)
    save_feedback(input_text, data.rating, comment)

    return {"status": "Feedback saved successfully"}