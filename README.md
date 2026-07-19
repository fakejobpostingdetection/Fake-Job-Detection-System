# 🛡️ Fake Job Detection System using Machine Learning & Generative AI

## Overview

The Fake Job Detection System is an intelligent web application that detects whether a job posting is **Fake** or **Genuine** using Machine Learning. The application also leverages **Generative AI (Groq Llama 3.1)** to generate a professional explanation of the prediction, identify potential fraud indicators, assess risk level, and provide safety recommendations for users.

---

##  Features

- Fake and Genuine Job Detection
- Machine Learning based Classification
- TF-IDF Feature Extraction
- XGBoost Classifier
- AI-powered Fraud Analysis using Groq API
- Risk Level Prediction
- Interactive Dashboard
- Dataset Statistics
- Fraud Distribution Charts
- Fake & Genuine WordClouds
- Confusion Matrix Visualization
- Professional Streamlit User Interface

---

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- Matplotlib
- SciPy
- Joblib
- Groq API (Llama 3.1)
- HTML
- CSS

---

##  Machine Learning Workflow

1. Data Collection
2. Data Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering
5. TF-IDF Vectorization
6. Model Training
7. Hyperparameter Tuning
8. Model Evaluation
9. Deployment using Streamlit

---

##  Model Performance

| Metric | Score |
|---------|-------|
| Accuracy | **98.6%** |
| Precision | **99.2%** |
| Recall | **71.7%** |
| F1-Score | **83.2%** |
| ROC-AUC | **98.97%** |

---

##  Project Structure

```text
Fake Job Detection/
│
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
├── images/
│   ├── bg.jpeg
│   ├── confusion_matrix.png
│   ├── fraud_distribution_bar.png
│   ├── fraud_distribution_pie.png
│   ├── fake_wordcloud.png
│   ├── real_wordcloud.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   └── model_comparison.csv
│
├── notebooks/
│   ├── Data_Preprocessing.py
│   ├── EDA.py
│   ├── EDA_Text_Analysis.py
│   ├── EDA_Visualization.py
│   ├── Feature_Engineering.py
│   ├── Model_Training.py
│   ├── Model_Evaluation.py
│   └── Prediction.py
│
└── styles/
    └── style.css
```

---

##  Installation

Clone the repository:

```bash
git clone https://github.com/your-username/Fake-Job-Detection.git
```

Go to the project directory:

```bash
cd Fake-Job-Detection
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

## Run the Project

```bash
streamlit run app.py
```

---

##  Dataset

- Fake Job Postings Dataset
- Total Records: **17,880**
- Genuine Jobs: **17,014**
- Fake Jobs: **866**

---

## Future Scope

- Deep Learning based Detection
- Resume and Job Matching
- Real-time Job Scam Detection
- Browser Extension Support
- Multilingual Job Analysis
- Cloud Database Integration

---

##  Developer

Nancy and Pari Singh

---

##  License

This project is developed for **educational and research purposes** only.