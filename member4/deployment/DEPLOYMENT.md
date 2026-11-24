# SarcasTone Production Deployment & Execution Guide

This document provides exact, reproducible commands for running, testing, containerizing, and deploying the complete four-member SarcasTone system.

---

## 1. Local Environment Setup

### Prerequisites
- Python 3.11
- FFmpeg (for MP3/M4A audio decoding): `brew install ffmpeg` (macOS) or `apt-get install ffmpeg` (Ubuntu)
- Libsndfile: `brew install libsndfile` (macOS) or `apt-get install libsndfile1` (Ubuntu)

### Install Dependencies
```bash
# Recommended: Create a dedicated Python 3.11 environment
conda create -n sarcastone python=3.11 -y
conda activate sarcastone

# Install project packages
pip install --upgrade pip
pip install -r requirements.txt
pip install fastapi uvicorn streamlit python-multipart pydantic PyYAML altair
```

---

## 2. Running the System Locally

### Step 1: Start the FastAPI Backend Server
```bash
# From repository root
PYTHONPATH=. python3.11 -m uvicorn member4.backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- API Docs (Swagger): `http://localhost:8000/docs`
- Healthcheck: `http://localhost:8000/health`
- Model Metadata: `http://localhost:8000/model-info`

### Step 2: Start the Streamlit Dashboard
```bash
# From repository root (in a separate terminal)
PYTHONPATH=. streamlit run member4/frontend/app.py --server.port 8501
```
- Interactive Web Dashboard: `http://localhost:8501`

---

## 3. Testing REST API Endpoints via cURL

### Full Multimodal Prediction (Audio + Text + Context)
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "audio=@member2_audio/tests/fixtures/test_sample.wav" \
  -F "text=It's just a privilege to watch your mind at work." \
  -F "context=[\"I think I solved the puzzle.\"]" \
  -F "speaker=SHELDON"
```

### Text-Only Sarcasm Prediction
```bash
curl -X POST "http://localhost:8000/predict/text" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Oh brilliant, another great idea that completely broke the database.",
    "context": ["The migration script finished executing."],
    "speaker": "CHANDLER"
  }'
```

### Audio-Only Sarcasm Prediction (Automatic ASR)
```bash
curl -X POST "http://localhost:8000/predict/audio" \
  -F "audio=@member2_audio/tests/fixtures/test_sample.wav" \
  -F "speaker=SHELDON"
```

---

## 4. Docker & Docker Compose Deployment

### Build & Run with Docker Compose
```bash
# Build and start both Backend (port 8000) and Frontend (port 8501)
docker-compose up --build -d

# View service logs
docker-compose logs -f

# Stop containers
docker-compose down
```

### Single Container Docker Build (Backend)
```bash
docker build -t sarcastone-backend -f member4/docker/Dockerfile .
docker run -p 8000:8000 sarcastone-backend
```

---

## 5. Google Colab GPU Execution

To train or evaluate on Google Colab with GPU acceleration:
```python
# Cell 1: Clone and install dependencies
!git clone https://github.com/muskan/SarcasTone.git
%cd SarcasTone
!pip install -r requirements.txt
!pip install fastapi uvicorn streamlit pyngrok

# Cell 2: Verify 3-Way Sync
!PYTHONPATH=. python3.11 verify_all_members_sync.py

# Cell 3: Run Full Multimodal Inference
import sys
sys.path.insert(0, ".")
from member4.backend.pipeline import MultimodalPipeline

pipeline = MultimodalPipeline()
result = pipeline.predict(
    audio_path="member2_audio/tests/fixtures/test_sample.wav",
    text="It's just a privilege to watch your mind at work.",
    speaker="SHELDON"
)
print("Classification:", result.label)
print("Confidence:", result.confidence)
print("PSID Incongruence Score:", result.psid_dissonance_score)
```

---

## 6. Automated Testing & Verification
```bash
# Run Member 4 Backend & Pipeline Tests
PYTHONPATH=. python3.11 -m unittest discover member4/tests -v

# Run 3-Way Synchronization Verification across all members
PYTHONPATH=. python3.11 verify_all_members_sync.py

# Run Robustness & Environmental Stress Suite
PYTHONPATH=. python3.11 member4/experiments/robustness/run_robustness.py
```
