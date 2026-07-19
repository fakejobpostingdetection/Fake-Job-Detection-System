# =====================================
# Fake Job Detection - Model Training
# =====================================

import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ------------------------------------
# Load Data
# ------------------------------------

X_train = joblib.load("models/X_train.pkl")
X_test = joblib.load("models/X_test.pkl")
y_train = joblib.load("models/y_train.pkl")
y_test = joblib.load("models/y_test.pkl")

# ------------------------------------
# Models
# ------------------------------------

models = {

    "Logistic Regression":
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ),

    "Naive Bayes":
        MultinomialNB(),

    "Decision Tree":
        DecisionTreeClassifier(
            random_state=42,
            class_weight="balanced"
        ),

    "Random Forest":
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        ),

    "Linear SVM":
        LinearSVC(
            class_weight="balanced",
            random_state=42
        ),

   "XGBoost":
        XGBClassifier(
            random_state=42,
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=1
        )
}

# ------------------------------------
# Variables
# ------------------------------------

results = []

best_accuracy = 0
best_model = None
best_model_name = ""

print("=" * 60)
print("MODEL TRAINING STARTED")
print("=" * 60)

# ------------------------------------
# Train Models
# ------------------------------------

for name, model in models.items():

    print("\n" + "=" * 60)
    print("Training :", name)
    print("=" * 60)
    # Hyperparameter Tuning for XGBoost

    if name == "XGBoost":

        print("\nPerforming Hyperparameter Tuning...")

        param_dist = {

            "n_estimators": randint(100, 500),

            "max_depth": randint(3, 10),

            "learning_rate": uniform(0.01, 0.3),

            "subsample": uniform(0.7, 0.3),

            "colsample_bytree": uniform(0.7, 0.3),

            "gamma": uniform(0, 5),

            "min_child_weight": randint(1, 10)

        }

        random_search = RandomizedSearchCV(

            estimator=model,

            param_distributions=param_dist,

            n_iter=10,

            scoring="f1",

            cv=3,

            verbose=1,

            random_state=42,

            n_jobs=1
        )
        random_search.fit(X_train, y_train)

        model = random_search.best_estimator_

        print("\nBest Parameters Found:")

        print(random_search.best_params_)

    if name != "XGBoost":
        model.fit(X_train, y_train)

    pred = model.predict(X_test)
    # Get prediction probabilities (for ROC Curve)

    if name == "XGBoost":
        y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    pre = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)

    print("Accuracy :", round(acc, 4))
    print("Precision:", round(pre, 4))
    print("Recall   :", round(rec, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report")
    print(classification_report(y_test, pred))
    # ------------------------------------
    # ROC Curve (Only for XGBoost)
    # ------------------------------------

    if name == "XGBoost":

        fpr, tpr, thresholds = roc_curve(y_test, y_prob)

        roc_auc = auc(fpr, tpr)

        print(f"\nROC-AUC Score : {roc_auc:.4f}")

        plt.figure(figsize=(7,6))

        plt.plot(
            fpr,
            tpr,
            color="royalblue",
            linewidth=2,
            label=f"AUC = {roc_auc:.4f}"
        )

        plt.plot(
            [0,1],
            [0,1],
            linestyle="--",
            color="gray"
        )

        plt.xlabel("False Positive Rate")

        plt.ylabel("True Positive Rate")

        plt.title("ROC Curve - XGBoost")

        plt.legend(loc="lower right")

        plt.grid(alpha=0.3)

        plt.savefig("images/roc_curve.png", dpi=300)

        plt.show()

        plt.close()

    print("\nConfusion Matrix")
    print(confusion_matrix(y_test, pred))

    
    # ------------------------------------
    # Precision-Recall Curve (Only for XGBoost)
    # ------------------------------------

    if name == "XGBoost":

        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)

        ap_score = average_precision_score(y_test, y_prob)

        print(f"\nAverage Precision Score : {ap_score:.4f}")

        plt.figure(figsize=(7,6))

        plt.plot(
            recall,
            precision,
            color="darkorange",
            linewidth=2,
            label=f"AP = {ap_score:.4f}"
        )

        plt.xlabel("Recall")

        plt.ylabel("Precision")

        plt.title("Precision-Recall Curve - XGBoost")

        plt.legend(loc="lower left")

        plt.grid(alpha=0.3)

        plt.savefig("images/precision_recall_curve.png", dpi=300)

        plt.show()

        plt.close()


     results.append([
            name,
            acc,
            pre,
            rec,
            f1
        ])
    # Save Logistic Regression separately
    if name == "Logistic Regression":
        joblib.dump(
            model,
            "models/logistic_regression.pkl"
        )

    # Save Best Model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# ------------------------------------
# Save Best Model
# ------------------------------------

joblib.dump(best_model, "models/best_model.pkl")

print("\n")
print("=" * 60)
print("BEST MODEL SAVED")
print("=" * 60)
print("Best Model   :", best_model_name)
print("Best Accuracy:", round(best_accuracy, 4))

# ------------------------------------
# Final Results
# ------------------------------------

print("\n")
print("=" * 60)
print("FINAL RESULTS")
print("=" * 60)
results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

results_df = results_df.sort_values(
    by="F1 Score",
    ascending=False
)

print("\nModel Comparison Table\n")

print(results_df)

results_df.to_csv(
    "models/model_comparison.csv",
    index=False
)

for r in results:
    print(r)