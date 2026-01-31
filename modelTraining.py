import pandas as pd
import pickle
import re
from collections import Counter
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2  # Added for feature selection
import numpy as np  # Added for potential optimizations

# -------------------------------
# Load Dataset
# -------------------------------
# Make the path configurable for better portability
dataset_path = r"C:/Users/arpan/PycharmProjects/AiChatbot/DiseaseAndSymptoms.csv"
df = pd.read_csv(dataset_path)

# -------------------------------
# Symptom Cleaning Function (Optimized)
# -------------------------------
# Compile regex patterns once for efficiency (avoids recompiling in loops)
intensity_pattern = re.compile(r"\b(mild|moderate|severe|high|low)\b")
non_alpha_pattern = re.compile(r"[^a-z ]")
whitespace_pattern = re.compile(r"\s+")

def clean_symptom(symptom):
    symptom = str(symptom).lower()
    symptom = symptom.replace("_", " ")
    symptom = intensity_pattern.sub("", symptom)
    symptom = non_alpha_pattern.sub("", symptom)
    symptom = whitespace_pattern.sub(" ", symptom).strip()
    return symptom

# -------------------------------
# Combine Symptom Columns (Vectorized)
# -------------------------------
symptom_cols = df.columns[1:]

# Use pandas apply with axis=1 for better performance on rows
df["symptoms"] = df[symptom_cols].apply(
    lambda row: [
        clean_symptom(s)
        for s in row
        if pd.notna(s) and clean_symptom(s)
    ],
    axis=1
)

# -------------------------------
# Remove Weak Records
# -------------------------------
df["symptom_count"] = df["symptoms"].apply(len)
df = df[df["symptom_count"] >= 2]

# -------------------------------
# Remove Rare Diseases (VERY IMPORTANT)
# -------------------------------
# Use pandas value_counts for efficiency instead of Counter
disease_counts = df["Disease"].value_counts()
COMMON_DISEASES = disease_counts[disease_counts >= 15].index

df = df[df["Disease"].isin(COMMON_DISEASES)]

# -------------------------------
# Encode Symptoms
# -------------------------------
mlb = MultiLabelBinarizer()
X = mlb.fit_transform(df["symptoms"])
y = df["Disease"]

# Optional: Feature Selection to Reduce Dimensionality (Improves Efficiency)
# Select top K features based on chi-squared test (adjust K as needed, e.g., 100-500)
# This reduces the number of features, speeding up training without much loss in accuracy
k_features = min(500, X.shape[1])  # Cap at 500 or less if fewer features
selector = SelectKBest(chi2, k=k_features)
X = selector.fit_transform(X, y)

# -------------------------------
# Train / Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -------------------------------
# Train Safer Model (Optimized)
# -------------------------------
# Reduce n_estimators if training is too slow (e.g., from 400 to 200), and add n_jobs for parallelism
model = RandomForestClassifier(
    n_estimators=200,  # Reduced for speed; increase if accuracy is prioritized over time
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1  # Use all available CPU cores for parallel training
)

model.fit(X_train, y_train)

# -------------------------------
# Evaluate Model Accuracy
# -------------------------------
# Calculate accuracy on the test set
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy on Test Set: {accuracy:.4f} ({accuracy * 100:.2f}%)")

# -------------------------------
# Save Model & Encoder
# -------------------------------
# Make paths configurable
model_path = r"C:/Users/arpan/PycharmProjects/AiChatbot/Models/disease_model.pkl"
encoder_path = r"C:/Users/arpan/PycharmProjects/AiChatbot/Models/symptom_encoder.pkl"
selector_path = r"C:/Users/arpan/PycharmProjects/AiChatbot/Models/feature_selector.pkl"  # Save selector if using feature selection

pickle.dump(model, open(model_path, "wb"))
pickle.dump(mlb, open(encoder_path, "wb"))
if 'selector' in locals():  # Save selector only if feature selection was applied
    pickle.dump(selector, open(selector_path, "wb"))

print("✅ Optimized model trained & saved successfully")