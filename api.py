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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

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


def extract_symptoms(text: str) -> List[str]:
    text = text.lower()
    text = text_clean_pattern.sub(" ", text)
    text_words = set(text.split())
    extracted = []
    for readable, original in symptom_map.items():
        if all(word in text_words for word in readable.split()):
            extracted.append(original)
    return extracted


def get_user_name(request: Request, response: Response) -> str:
    # Modified: Return "Anonymous" if no cookie is set, instead of raising an exception.
    # This makes /chat and /history not depend on /set_name.
    #print(request)
    #user_name = request.cookies.get("user_name")
    if not user_name:
        #user_name = "Anonymous"  # Default fallback
        logger.info("No user_name cookie found; using 'Anonymous'.")
    return user_name


def get_csv_file(user_name: str) -> str:
    return f"CSVUsersFile/{user_name}_chat_history.csv"


def save_to_csv(csv_file: str, symptoms: List[str], disease: str):
    new_record = pd.DataFrame({
        "symptoms": [",".join(symptoms)],
        "predicted_disease": [disease],
        "date": [datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")]
    })

    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        updated_df = pd.concat([existing_df, new_record], ignore_index=True)
    else:
        updated_df = new_record

    updated_df.to_csv(csv_file, index=False)
    logger.info(f"Saved record to CSV: {csv_file}")

user_name="Anonymous"
@app.post("/set_name", response_model=dict)
def set_name(req: NameRequest, response: Response):
    userName = req.name.strip()
    if not userName:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    # Sanitize name for file safety (remove special chars)
    userName = re.sub(r"[^a-zA-Z0-9_]", "", userName)
    global user_name
    user_name=userName
    response.set_cookie(key="user_name", value=user_name, httponly=True)
    logger.info(f"User name set: {user_name}")
    return {"message": f"Name set to {user_name}. You can now chat."}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, response: Response) -> ChatResponse:
    user_name = get_user_name(request, response)
    print(user_name)# Now uses default if not set
    csv_file = get_csv_file(user_name)
    try:
        logger.info(f"Received message for user {user_name}: {req.message}")
        symptoms = extract_symptoms(req.message)
        logger.info(f"Extracted symptoms: {symptoms}")

        if not symptoms:
            return ChatResponse(
                error_message=(
                    "❗ I couldn't identify any symptoms.\n\n"
                    "Try typing:\n"
                    "• I have fever and headache\n"
                    "• cough with chest pain"
                )
            )

        if len(symptoms) < 2:
            symptom_name = symptoms[0].replace("_", " ").capitalize()
            return ChatResponse(
                error_message=(
                    f"❗ {symptom_name} alone is too common to identify a disease.\n"
                    "Please provide at least two symptoms.\n\n"
                    "Example:\n"
                    "• fever and headache\n"
                    "• fever, cough, fatigue"
                )
            )

        input_vector = mlb.transform([symptoms])
        if selector:
            input_vector = selector.transform(input_vector)

        disease = model.predict(input_vector)[0]
        precautions = precautions_dict.get(disease, [])
        logger.info(f"Predicted disease: {disease}")

        save_to_csv(csv_file, symptoms, disease)

        return ChatResponse(
            predicted_disease=disease,
            precautions=precautions
        )

    except Exception as e:
        logger.error(f"Error in chat for user {user_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@app.get("/history")
def get_history(request: Request, response: Response):
    user_name = get_user_name(request, response)  # Now uses default if not set
    csv_file = get_csv_file(user_name)
    logger.info(f"Fetching history for user: {user_name}")
    try:
        if not os.path.exists(csv_file):
            logger.info(f"CSV file does not exist: {csv_file}")
            return {"history": []}

        df = pd.read_csv(csv_file)
        logger.info(f"CSV loaded with {len(df)} rows")
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        history = [
            {
                "date": row["date"].strftime("%Y-%m-%d %H:%M:%S"),
                "symptoms": row["symptoms"].split(","),
                "predicted_disease": row["predicted_disease"]
            }
            for _, row in df.iterrows()
        ]
        logger.info(f"Returning {len(history)} history items")
        return {"history": history}
    except Exception as e:
        logger.error(f"Error fetching history for user {user_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")