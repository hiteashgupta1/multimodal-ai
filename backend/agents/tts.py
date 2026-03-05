from gtts import gTTS
import io
import time
from evaluation.log_metrics import log_metric
from experiments.experiment_tracker import log_experiment

def text_to_speech(text):
    start = time.time()
    latency = round(time.time() - start, 2)
    # ✅ Log metrics
    log_metric(
        agent="tts",
        confidence=0.95,
        object_count=None,
        latency=latency,
        hallucination=False
    )
    log_experiment(
        agent="tts",
        confidence=0.95,
        latency=latency,
        hallucination=False,
        model_version="v1"
    )

    try:
        tts = gTTS(text=text, lang="en")
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.read()
    except Exception as e:
        print("gTTS ERROR:", str(e))
        return None
    
