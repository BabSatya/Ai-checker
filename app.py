from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from features.feature_extractor import extract_features

import pickle
import nltk
from nltk.tokenize import sent_tokenize
import os
from datetime import datetime   # ✅ ADDED (logs only)

from firebase_admin import auth
from backend.firebase_admin import db

# ✅ CORRECT ADMIN ROUTER IMPORT
from router.admin import admin_router


# ============================
# App Initialization
# ============================
app = FastAPI(title="Turnitin-Style AI Detector")

# ============================
# Middleware
# ============================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-checker-alpha.vercel.app",  # your Vercel frontend
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
) 
# ============================
# Register Admin Router
# ============================
app.include_router(admin_router)

# ============================
# Download NLTK Tokenizer
# ============================
# nltk.download("punkt")

# ============================
# Load Model
# ============================
MODEL_PATH = "models/xgb_model_.pkl"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found at models/xgb_model_.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# ============================
# Request Schema
# ============================
class TextInput(BaseModel):
    text: str

# ============================
# Heuristic Logic
# ============================
def classify_turnitin(ai_percent, human_percent, polish_percent):

    if ai_percent >= 20 and ai_percent > human_percent:
        return "AI"

    if polish_percent >= 95:
        return "AI"

    if polish_percent < 90:
        return "AI" if ai_percent > human_percent else "Human"

    if human_percent >= ai_percent + 10:
        return "Human"

    if ai_percent >= 12:
        return "AI"

    return "Human"

# ============================
# Prediction API + Token System



# ============================
@app.post("/predict")
def predict(data: TextInput):

    text = (data.text or "").strip()

    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")

    sentences = sent_tokenize(text)

    if not sentences:
        raise HTTPException(status_code=400, detail="No sentences found")

    # ✅ optional safety limit (prevents Render timeouts)
    MAX_SENTENCES = 25
    sentences = sentences[:MAX_SENTENCES]

    results = []
    total_ai = 0
    total_human = 0

    for sent in sentences:

        features = extract_features(sent).reshape(1, -1)
        probs = model.predict_proba(features)[0]

        human_p = float(probs[0] * 100)
        ai_p = float(probs[1] * 100)
        polish_p = float(probs[2] * 100)

        total_ai += ai_p
        total_human += human_p

        final_label = classify_turnitin(ai_p, human_p, polish_p)

        results.append({
            "sentence": sent,
            "human_probability": round(human_p, 2),
            "ai_probability": round(ai_p, 2),
            "over_polished_probability": round(polish_p, 2),
            "final_label": final_label
        })

    avg_ai = total_ai / len(sentences)
    avg_human = total_human / len(sentences)

    final_doc_label = "AI" if avg_ai > avg_human else "Human"

    return {
        "overall_human_probability": round(avg_human, 2),
        "overall_ai_probability": round(avg_ai, 2),
        "final_document_label": final_doc_label,
        "sentences": results
    }
