import pandas as pd
from typing import List, Dict
import datetime
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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