from gtts import gTTS
from app.config import TTS_LANG, TTS_SLOW
import os
import uuid

OUTPUT_DIR = "outputs/audio"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def synthesize_speech(text):
    filename = f"{OUTPUT_DIR}/{uuid.uuid4().hex}.mp3"
    tts = gTTS(text=text, lang=TTS_LANG,slow=TTS_SLOW)
    tts.save(filename)
    return f"/audio/{os.path.basename(filename)}"
