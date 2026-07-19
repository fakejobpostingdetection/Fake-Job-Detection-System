# ==============================
# Fake Job Detection - EDA
# ==============================

import pandas as pd
import numpy as np

# Load Dataset
df = pd.read_csv("data/raw/fake_job_postings.csv")

# Display first 5 rows
print("\n========== First 5 Rows ==========\n")
print(df.head())

# Dataset Shape
print("\n========== Dataset Shape ==========\n")
print(df.shape)

# Column Names
print("\n========== Column Names ==========\n")
print(df.columns.tolist())

# Data Types
print("\n========== Data Types ==========\n")
print(df.dtypes)
# Missing Values

print("\n========== Missing Values ==========\n")
print(df.isnull().sum())
# ------------------------------------------------
# 6. Duplicate Rows
# ------------------------------------------------
print("\n========== DUPLICATE ROWS ==========\n")
print(df.duplicated().sum())

# ------------------------------------------------
# 7. Statistical Summary
# ------------------------------------------------
print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

# ------------------------------------------------
# 8. Target Variable Distribution
# ------------------------------------------------
print("\n========== TARGET VARIABLE ==========\n")
print(df["fraudulent"].value_counts())

print("\nPercentage:\n")
print(df["fraudulent"].value_counts(normalize=True) * 100)

# ------------------------------------------------
# 9. Unique Values
# ------------------------------------------------
print("\n========== UNIQUE VALUES ==========\n")
print(df.nunique())

# ------------------------------------------------
# 10. Employment Type
# ------------------------------------------------
print("\n========== EMPLOYMENT TYPE ==========\n")
print(df["employment_type"].value_counts(dropna=False))

# ------------------------------------------------
# 11. Required Experience
# ------------------------------------------------
print("\n========== REQUIRED EXPERIENCE ==========\n")
print(df["required_experience"].value_counts(dropna=False))

# ------------------------------------------------
# 12. Required Education
# ------------------------------------------------
print("\n========== REQUIRED EDUCATION ==========\n")
print(df["required_education"].value_counts(dropna=False))

# ------------------------------------------------
# 13. Top Industries
# ------------------------------------------------
print("\n========== TOP INDUSTRIES ==========\n")
print(df["industry"].value_counts().head(10))

# ------------------------------------------------
# 14. Top Functions
# ------------------------------------------------
print("\n========== TOP FUNCTIONS ==========\n")
print(df["function"].value_counts().head(10))

# ------------------------------------------------
# 15. Top Locations
# ------------------------------------------------
print("\n========== TOP LOCATIONS ==========\n")
print(df["location"].value_counts().head(10))

# ------------------------------------------------
# 16. Salary Range Missing
# ------------------------------------------------
print("\n========== SALARY RANGE MISSING ==========\n")
print(df["salary_range"].isnull().sum())

# ------------------------------------------------
# 17. Telecommuting
# ------------------------------------------------
print("\n========== TELECOMMUTING ==========\n")
print(df["telecommuting"].value_counts())

# ------------------------------------------------
# 18. Has Company Logo
# ------------------------------------------------
print("\n========== HAS COMPANY LOGO ==========\n")
print(df["has_company_logo"].value_counts())

# ------------------------------------------------
# 19. Has Questions
# ------------------------------------------------
print("\n========== HAS QUESTIONS ==========\n")
print(df["has_questions"].value_counts())

# ------------------------------------------------
# END
# ------------------------------------------------
print("\n====================================")
print("EDA COMPLETED SUCCESSFULLY")
print("====================================")