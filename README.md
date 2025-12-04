✈️ SkyCast AI — Intelligent Flight Price Prediction System

SkyCast AI is a full-stack machine learning application that predicts real-world flight prices using a deep neural network trained on 300,000+ flight listings.
It behaves like an AI-powered travel assistant, helping users identify the cheapest option, the best-value option, and the most time-efficient option in real time.

🌐 Live Links
Service	URL
Frontend	http://bright-banoffee-094a46.netlify.app
Backend API	(https://flight-price-prediction-production-4984.up.railway.app)
⭐ Key Features
🔥 Machine Learning

Deep Neural Network (PyTorch, 256-unit dense layers)

Batch Normalization + Dropout (0.3) for stability

Learning-Rate Scheduler (ReduceLROnPlateau)

Early Stopping for optimized convergence

Complete preprocessing pipeline (scaling + encoding)

⚡ Backend API

FastAPI asynchronous server

Millisecond-level prediction latency

Clean Swagger docs at /docs

Model + scaler + encoders loaded via joblib

🎨 Frontend

Cyberpunk / glassmorphism UI

Fully responsive layout

Real-time price predictions

Smooth animations & clean JS fetch integration

🚀 Deployment

Docker containerization

Backend hosted on Railway

Frontend hosted on Netlify

📁 Project Structure
AI_PROJECT/
│
├── backend/
│   ├── server.py                # FastAPI server
│   ├── Dockerfile               # Railway deployment config
│   ├── requirements.txt         # Dependencies
│   ├── best_flight_brain.pth    # Trained PyTorch model
│   ├── flightdata.csv           # 300k-row dataset
│   ├── scaler.save              # StandardScaler
│   ├── encoder.save             # LabelEncoders
│   └── ...
│
└── frontend/
    └── index.html               # UI Dashboard

🧩 Installation & Local Development
1. Clone the Repository
git clone https://github.com/050025493/Flight-Price-Prediction.git
cd Flight-Price-Prediction

2. Backend Setup
cd backend
pip install -r requirements.txt


Run the FastAPI server:

uvicorn server:app --reload


Your API will be available at:

http://127.0.0.1:8000

3. Frontend Setup

Open:

frontend/index.html


Make sure the API endpoint is set to:

const API_URL = "http://127.0.0.1:8000";


Then simply open index.html in your browser.

🧠 Model Training (Optional)

If you want to retrain the neural network:

Step 1 — Prepare Data

Place flightdata.csv in the backend directory.

Step 2 — Run Preprocessing
python clean_kaggle_flight.py

Step 3 — Train
python train_kaggle_flight.py


Training runs up to 5000 epochs with:

early stopping

learning rate scheduler

automatic best model saving (best_flight_brain.pth)

☁️ Deployment Guide
🚀 Deploying the Backend (Railway)

Push your repository to GitHub

Create a new Railway project

Select your repo

In Railway settings → Set Root Directory to /backend

Deploy

🌐 Deploying the Frontend (Netlify)

Edit the API URL inside index.html:

const API_URL = "https://your-railway-url.up.railway.app";


Drag the frontend/ folder into Netlify Drop

Netlify will host it instantly
