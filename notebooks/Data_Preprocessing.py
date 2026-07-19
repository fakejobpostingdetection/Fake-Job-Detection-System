# ==========================================
# Fake Job Detection
# Data Preprocessing
# ==========================================

import pandas as pd
import numpy as np

print("="*50)
print("DATA PREPROCESSING")
print("="*50)

# Load Dataset
df = pd.read_csv("data/raw/fake_job_postings.csv")

# -----------------------------------
# Remove Duplicate Records
# -----------------------------------

print("\nRemoving Duplicates...")

before = df.shape[0]

df = df.drop_duplicates()

after = df.shape[0]

print("Rows Before :", before)
print("Rows After  :", after)
print("Duplicates Removed :", before-after)

# -----------------------------------
# Fill Missing Values
# -----------------------------------

print("\nHandling Missing Values...")

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits",
    "industry",
    "function",
    "department",
    "location"
]

for col in text_columns:
    if col in df.columns:
        df[col] = df[col].fillna("Unknown")

# -----------------------------------
# Salary Columns
# -----------------------------------

salary_cols = ["salary_range"]

for col in salary_cols:
    if col in df.columns:
        df[col] = df[col].fillna("Not Mentioned")

# -----------------------------------
# Binary Columns
# -----------------------------------

binary_cols = [
    "telecommuting",
    "has_company_logo",
    "has_questions"
]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# -----------------------------------
# Final Missing Values
# -----------------------------------

print("\nRemaining Missing Values")

print(df.isnull().sum())

# -----------------------------------
# Save Clean Dataset
# -----------------------------------

df.to_csv("data/raw/clean_fake_job_postings.csv", index=False)

print("\nClean Dataset Saved Successfully")

print("="*50)
print("PREPROCESSING COMPLETED")
print("="*50)