import whisper
from app.config import WHISPER_MODEL
import tempfile
import os

# Load model once
model = whisper.load_model(WHISPER_MODEL)

def transcribe_audio(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(file.file.read())
        temp_path = tmp.name

    try:
      result = model.transcribe(temp_path)
    finally:
      os.remove(temp_path)

    return result["text"]
