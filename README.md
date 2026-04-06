#Voice Bot
A web-based AI voice bot that converts speech to text, predicts intent, generates responses, and returns audio using TTS.

## Features
- 🎤 Microphone recording + file upload  
- 📝 Automatic speech recognition (Whisper)  
- 🧠 Intent prediction (BERT)  
- 💬 Rule-based response generation  
- 🔊 Text-to-Speech (gTTS)  
- 📂 Audio response saved in `outputs/audio`  

## Architecture
```text
[User] 
   │
   ▼
[Frontend (index.html)]  ---> Uploads audio or records via microphone
   │
   ▼
[FastAPI /voicebot API]
   │
   ├─▶ [Whisper ASR] ---> Transcribed text
   │
   ├─▶ [BERT Intent Model] ---> Predicted intent
   │
   ├─▶ [Response Service] ---> Generated text response
   │
   └─▶ [gTTS] ---> Audio response
   │
   ▼
[Frontend]
   │
   └─▶ Plays audio response, displays transcript & intent


Setup:
1. Clone repo
git clone <repo-url>
cd voicebot
2. Install Python dependencies
pip install -r requirements.txt
⚠ Make sure ffmpeg binary is installed on your system:
sudo apt install ffmpeg
3. Run server
uvicorn main:app --reload
4. Open in browser
Navigate to http://localhost:8000
________________________________________
Docker
Build Docker image
docker build -t voicebot .
Run Docker container
docker run -p 8000:8000 voicebot
________________________________________
Project Structure
voicebot/
├─ requirements.txt
├─ Dockerfile
├─ templates/
│   └─ index.html
├─ outputs/
│   └─ audio/
├─ models/
    └─ intent_model/
├─ templatess/
├─ app/
│   ├─ config.py
    ├─ main.py
│   ├─ services/
│   │   ├─ asr_service.py
│   │   ├─ intent_service.py
│   │   ├─ response_service.py
│   │   └─ tts_service.py
│   └─ utils/
│       └─ logger.py
________________________________________
Note:
•	Whisper ASR requires ffmpeg. The Dockerfile already installs it. 
•	BERT intent model must be downloaded and placed at INTENT_MODEL_PATH. 
•	API endpoints: 
o	/voicebot – Send audio & get response 
o	/transcribe, /predict-intent, /generate-response, /synthesize – Individual pipeline steps 
•	Frontend is served at / 
________________________________________
License
MIT License
