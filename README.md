🔍 Fraud Detector API
A production-ready FastAPI backend for real-time fraudulent job posting detection, powered by an ensemble ML model and deployable via Docker.
🏗️ Architecture
Chrome Extension / Client
        ↓
  FastAPI Server (main.py)
        ↓
  Ensemble ML Model (CatBoost + ensemble)
        ↓
  Fraud Score + Risk Signals
✨ Features

Real-time inference via REST API
Ensemble ML model (v33) with multiple lightweight variants (v13: 98MB, v17: 1.3MB)
Chrome Extension integration with CORS support
Docker deployment ready
Feature importance analysis included

🚀 Quick Start
Run locally
bashpip install -r requirements.txt
uvicorn main:app --reload
Run with Docker
bashdocker build -t fraud-detector .
docker run -p 8000:8000 fraud-detector
📡 API Endpoints
POST /predict
{
  "title": "Job Title",
  "description": "Job description...",
  "requirements": "...",
  "benefits": "...",
  "company_profile": "..."
}

→ Response:
{
  "fraud_probability": 0.87,
  "is_fraud": true,
  "risk_signals": [...]
}
🤖 Model

Trained on EMSCAD dataset (17,880 real-world job postings)
Multiple model versions for different deployment constraints
Feature importance visualized in feature_importance_simple_v9.png

🔗 Related

🤗 Live Demo on Hugging Face Spaces
📊 Dataset: EMSCAD — Employment Scam Aegean Dataset

🛠️ Tech Stack
Python FastAPI Docker CatBoost scikit-learn pandas uvicorn
