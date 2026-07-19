# ==========================================
# Fake Job Detection - EDA Visualization
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

# Load Dataset
df = pd.read_csv("data/raw/fake_job_postings.csv")

# Better plotting style
plt.style.use("ggplot")

# ==========================================================
# 1. Fraudulent vs Genuine Jobs (Bar Chart)
# ==========================================================

plt.figure(figsize=(6,5))

df["fraudulent"].value_counts().plot(
    kind="bar",
    color=["skyblue","red"]
)

plt.title("Fake vs Genuine Job Postings")
plt.xlabel("Class")
plt.ylabel("Count")
plt.xticks([0,1],["Genuine","Fake"],rotation=0)

plt.tight_layout()
plt.savefig("images/fraud_distribution_bar.png")
plt.close()

# ==========================================================
# 2. Fraud Percentage (Pie Chart)
# ==========================================================

plt.figure(figsize=(6,6))

df["fraudulent"].value_counts().plot(
    kind="pie",
    labels=["Genuine","Fake"],
    autopct="%1.1f%%",
    startangle=90
)

plt.ylabel("")
plt.title("Percentage of Fake and Genuine Jobs")

plt.tight_layout()
plt.savefig("images/fraud_distribution_pie.png")
plt.close()

# ==========================================================
# 3. Missing Values Chart
# ==========================================================

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

plt.figure(figsize=(12,6))

missing.plot(kind="bar")

plt.title("Missing Values in Dataset")
plt.xlabel("Columns")
plt.ylabel("Missing Values")
plt.xticks(rotation=90)

plt.tight_layout()
plt.savefig("images/missing_values.png")
plt.close()

print("\nVisualization Part-1 Completed Successfully!")