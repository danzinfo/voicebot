from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from app.config import MAX_FILE_SIZE_MB
import os
import json

# Services
from app.services.asr_service import transcribe_audio
from app.services.intent_service import predict_intent
from app.services.response_service import generate_response
from app.services.tts_service import synthesize_speech

from app.utils.logger import get_logger
logger = get_logger()

# -----------------------------
# App setup
# -----------------------------
app = FastAPI(title="AI Voice Bot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Create audio folder
os.makedirs("outputs/audio", exist_ok=True)
os.makedirs("templates",exist_ok=True) #Frontend Templates folder

#Serve audio files
app.mount("/audio", StaticFiles(directory="outputs/audio"), name="audio")


# Serve static frontend files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="templates/static"), name="static")


# -----------------------------
# Routes
# -----------------------------

@app.get("/")
def serve_index():
    return FileResponse("templates/index.html")

@app.get("/api")
def root():
    return {"message": "AI Voice Bot API is running. Use /voicebot to send audio."}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/voicebot")
async def voicebot(file: UploadFile = File(...)):
    
    #Validate file type
    if not file.content_type.startswith("audio/"):
        return JSONResponse({"error": "Only audio files are allowed"}, status_code=400)

    #Validate file size
    contents = await file.read()
    file_size_mb = len(contents) / (1024 * 1024)

    if file_size_mb > MAX_FILE_SIZE_MB:
        return JSONResponse({"error": "File too large"}, status_code=400)

    
    file.file.seek(0)  # reset pointer
    try:
        #ASR
        text = transcribe_audio(file)
        logger.info(f"Transcript: {text}")

        if not text or len(text.strip()) < 2:
           return JSONResponse({"error": "No valid speech detected"}, status_code=400)


        #Intent
        intent, confidence = predict_intent(text)

        #Response
        if confidence < 0.5:
            response_text = "Sorry, I didn’t understand that. Please repeat."
        else:
            response_text = generate_response(intent)

        #TTS
        audio_path = synthesize_speech(response_text)

        return {
            "transcript": text,
            "intent": intent,
            "confidence": confidence,
            "response": response_text,
            "audio_path": audio_path
        }

    except Exception as e:
        logger.error(f"Voicebot error: {str(e)}")
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/metrics") #from training train_intent_model
def get_metrics():
    try:
        with open("outputs/metrics.json", "r") as f:
            metrics = json.load(f)
        return metrics
    except Exception as e:
        logger.error(f"Metrics error: {str(e)}")
        return JSONResponse({"error": "Metrics not found"}, status_code=500)
    

# -----------------------------
# Individual Pipeline Endpoints
# -----------------------------

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        text = transcribe_audio(file)
        return {"transcript": text}
    except Exception as e:
        logger.error(f"Transcribe error: {str(e)}")
        return JSONResponse({"error": "ASR failed"}, status_code=500)


@app.post("/predict-intent")
async def predict(text: str):
    try:
        intent, confidence = predict_intent(text)
        return {
            "intent": intent,
            "confidence": confidence
        }
    except Exception as e:
        logger.error(f"Intent error: {str(e)}")
        return JSONResponse({"error": "Intent prediction failed"}, status_code=500)


@app.post("/generate-response")
async def respond(intent: str):
    try:
        response = generate_response(intent)
        return {"response": response}
    except Exception as e:
        logger.error(f"Response error: {str(e)}")
        return JSONResponse({"error": "Response generation failed"}, status_code=500)


@app.post("/synthesize")
async def synthesize(text: str):
    try:
        audio_path = synthesize_speech(text)
        return {"audio_path": audio_path}
    except Exception as e:
        logger.error(f"TTS error: {str(e)}")
        return JSONResponse({"error": "TTS failed"}, status_code=500)




# -----------------------------
# Run server
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)




