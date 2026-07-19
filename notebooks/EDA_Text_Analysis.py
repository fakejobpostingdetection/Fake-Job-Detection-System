# ==========================================
# Fake Job Detection - Text Analysis
# ==========================================

import os
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer

# Create images folder
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/raw/fake_job_postings.csv")

# Fill missing values
df["description"] = df["description"].fillna("")

# ==========================================================
# 1. Top 20 Most Frequent Words
# ==========================================================

vectorizer = CountVectorizer(stop_words="english", max_features=20)

X = vectorizer.fit_transform(df["description"])

word_counts = X.toarray().sum(axis=0)

words = vectorizer.get_feature_names_out()

word_df = pd.DataFrame({
    "Word": words,
    "Count": word_counts
})

word_df = word_df.sort_values(by="Count", ascending=True)

plt.figure(figsize=(10,6))
plt.barh(word_df["Word"], word_df["Count"])
plt.title("Top 20 Most Frequent Words")
plt.xlabel("Frequency")
plt.tight_layout()
plt.savefig("images/top_20_words.png")
plt.close()

# ==========================================================
# 2. WordCloud
# ==========================================================

text = " ".join(df["description"])

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(text)

plt.figure(figsize=(12,6))
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud of Job Descriptions")
plt.tight_layout()
plt.savefig("images/wordcloud.png")
plt.close()

# ==========================================================
# 3. Fake vs Genuine Job Comparison
# ==========================================================

fake_jobs = " ".join(
    df[df["fraudulent"] == 1]["description"]
)

real_jobs = " ".join(
    df[df["fraudulent"] == 0]["description"]
)

fake_cloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(fake_jobs)

real_cloud = WordCloud(
    width=800,
    height=400,
    background_color="white"
).generate(real_jobs)

plt.figure(figsize=(10,5))
plt.imshow(fake_cloud)
plt.axis("off")
plt.title("Fake Job WordCloud")
plt.tight_layout()
plt.savefig("images/fake_wordcloud.png")
plt.close()

plt.figure(figsize=(10,5))
plt.imshow(real_cloud)
plt.axis("off")
plt.title("Genuine Job WordCloud")
plt.tight_layout()
plt.savefig("images/real_wordcloud.png")
plt.close()

print("\n===================================")
print("TEXT ANALYSIS COMPLETED")
print("===================================")