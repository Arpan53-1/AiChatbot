from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import re
import os
import logging

from Prediction_Functions.predictionFunctions import NameRequest, ChatResponse, ChatRequest, extract_symptoms, \
    predict_disease, precautions_dict
from file_and_user_usecases.file_iteration import get_csv_file, save_to_csv

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

def get_user_name(request: Request, response: Response) -> str:
    if not user_name:
        #user_name = "Anonymous"  # Default fallback
        logger.info("No user_name cookie found; using 'Anonymous'.")
    return user_name


@app.post("/set_name", response_model=dict)
def set_name(req: NameRequest, response: Response):
    """
        Setting name for the user in the application
    """
    userName = req.name.strip()
    if not userName:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    userName = re.sub(r"[^a-zA-Z0-9_]", "", userName)
    global user_name
    user_name=userName
    response.set_cookie(key="user_name", value=user_name, httponly=True)
    logger.info(f"User name set: {user_name}")
    return {"message": f"Name set to {user_name}. You can now chat."}

user_name="Anonymous"
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request, response: Response) -> ChatResponse:
    """
    Response for each conversation with the application
    """
    user_name = get_user_name(request, response)
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
        if len(symptoms) < 3:
            return ChatResponse(
                error_message=(
                    "❗ Please provide at least 3 symptoms for better prediction.\n\n"
                    "Example:\n"
                    "• fever, cough, fatigue\n"
                    "• headache, nausea, vomiting"
                )
            )
        highest, top3, error = predict_disease(symptoms)
        if error:
            return ChatResponse(error_message=error)
        predicted_disease = highest["disease"]
        confidence = highest["probability"]
        precautions = precautions_dict.get(predicted_disease, [])
        logger.info(
            f"Predicted disease: {predicted_disease} "
            f"with confidence {confidence}%"
        )
        save_to_csv(csv_file, symptoms, predicted_disease)
        return ChatResponse(
            predicted_disease=predicted_disease,
            confidence_percentage=confidence,   # 👈 add this field in schema
            precautions=precautions
        )
    except Exception as e:
        logger.error(f"Error in chat for user {user_name}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

@app.get("/history")
def get_history(request: Request, response: Response):
    """
        Response for history with the application
    """
    user_name = get_user_name(request, response)
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