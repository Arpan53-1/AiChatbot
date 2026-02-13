from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import pandas as pd
import re
from typing import List, Dict
import datetime
import uuid
import os
import logging
import numpy as np

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



# Load model, encoder, etc.
MODEL_PATH = r"C:\Users\arpan\PycharmProjects\AiChatbot\Models\disease_model.pkl"
ENCODER_PATH = r"C:\Users\arpan\PycharmProjects\AiChatbot\Models\symptom_encoder.pkl"
SELECTOR_PATH = r"C:\Users\arpan\PycharmProjects\AiChatbot\Models\feature_selector.pkl"
PRECAUTIONS_PATH = r"C:\Users\arpan\PycharmProjects\AiChatbot\Disease precaution.csv"

try:
    model = pickle.load(open(MODEL_PATH, "rb"))
    mlb = pickle.load(open(ENCODER_PATH, "rb"))
    try:
        selector = pickle.load(open(SELECTOR_PATH, "rb"))
    except FileNotFoundError:
        selector = None
    logger.info("Model and encoder loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or encoder: {e}")
    raise RuntimeError(f"Failed to load model or encoder: {e}")

try:
    precautions_df = pd.read_csv(PRECAUTIONS_PATH)
    precautions_dict: Dict[str, List[str]] = {}
    for _, row in precautions_df.iterrows():
        disease = row["Disease"]
        precautions = row.iloc[1:].dropna().tolist()
        precautions_dict[disease] = precautions
    logger.info("Precautions loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load precautions: {e}")
    raise RuntimeError(f"Failed to load precautions: {e}")

text_clean_pattern = re.compile(r"[^a-z ]")
symptom_map = {s.replace("_", " "): s for s in mlb.classes_}






class NameRequest(BaseModel):
    name: str


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    predicted_disease: str = None
    precautions: List[str] = []
    disclaimer: str = "⚠️ This is not a medical diagnosis. Please consult a doctor."
    error_message: str = None

intensity_pattern = re.compile(r"\b(mild|moderate|severe|high|low|acute|chronic)\b")
non_alpha_pattern = re.compile(r"[^a-z ]")
whitespace_pattern = re.compile(r"\s+")

def clean_symptom(symptom: str) -> str:
    """
    Cleans and normalizes symptom text.
    Ensures consistency between training and prediction.
    """
    if not symptom:
        return ""
    symptom = str(symptom).lower().strip()
    symptom = symptom.replace("_", " ").replace("-", " ")
    symptom = intensity_pattern.sub("", symptom)
    symptom = non_alpha_pattern.sub("", symptom)
    symptom = whitespace_pattern.sub(" ", symptom).strip()
    return symptom


def predict_disease(user_symptoms):
    if len(user_symptoms) < 3:
        return None, None, "Please provide at least 3 symptoms."
    cleaned = [clean_symptom(s) for s in user_symptoms]
    encoded = mlb.transform([cleaned])
    if selector:
        encoded = selector.transform(encoded)
    probs = model.predict_proba(encoded)[0]
    top3_idx = np.argsort(probs)[-3:][::-1]
    results = []
    for idx in top3_idx:
        results.append({
            "disease": model.classes_[idx],
            "probability": round(probs[idx] * 100, 2)
        })
    highest = results[0]
    return highest, results, None


def extract_symptoms(text: str) -> List[str]:
    text = text.lower()
    text = text_clean_pattern.sub(" ", text)
    text_words = set(text.split())
    extracted = []
    for readable, original in symptom_map.items():
        if all(word in text_words for word in readable.split()):
            extracted.append(original)
    return extracted