# =====================================
# Fake Job Detection - Model Evaluation
# =====================================

import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# -------------------------------
# Load Model & Data
# -------------------------------

model = joblib.load("models/best_model.pkl")

X_test = joblib.load("models/X_test.pkl")
y_test = joblib.load("models/y_test.pkl")

# -------------------------------
# Prediction
# -------------------------------

y_pred = model.predict(X_test)

# -------------------------------
# Classification Report
# -------------------------------

print("="*60)
print("CLASSIFICATION REPORT")
print("="*60)

print(classification_report(y_test, y_pred))

# -------------------------------
# Confusion Matrix
# -------------------------------

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Real","Fake"]
)

disp.plot(cmap="Blues")

plt.title("Confusion Matrix")

plt.savefig("images/confusion_matrix.png")

plt.show()

print("="*60)
print("MODEL EVALUATION COMPLETED")
print("="*60)