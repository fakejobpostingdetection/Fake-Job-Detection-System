# ==========================================
# Fake Job Detection
# Feature Engineering
# ==========================================

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("="*50)
print("FEATURE ENGINEERING")
print("="*50)

# Load Clean Dataset
df = pd.read_csv("data/raw/clean_fake_job_postings.csv")

# Combine Important Text Columns
text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]

df["combined_text"] = ""

for col in text_columns:
    if col in df.columns:
        df["combined_text"] += df[col].astype(str) + " "

# Features and Target
X = df["combined_text"]
y = df["fraudulent"]

print("\nApplying TF-IDF...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

print("TF-IDF Shape :", X.shape)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain Shape :", X_train.shape)
print("Test Shape  :", X_test.shape)

# Save Files
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")
joblib.dump(X_train, "models/X_train.pkl")
joblib.dump(X_test, "models/X_test.pkl")
joblib.dump(y_train, "models/y_train.pkl")
joblib.dump(y_test, "models/y_test.pkl")

print("\nFiles Saved Successfully")

print("="*50)
print("FEATURE ENGINEERING COMPLETED")
print("="*50)