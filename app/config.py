from dotenv import load_dotenv
import os

load_dotenv()

# ASR
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")

# Intent model
INTENT_MODEL_PATH = os.getenv("INTENT_MODEL_PATH", "models/intent_model")

# TTS
TTS_LANG = os.getenv("TTS_LANG", "en")
TTS_SLOW = os.getenv("TTS_SLOW", "false").lower() == "true"

# App
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", 5))
