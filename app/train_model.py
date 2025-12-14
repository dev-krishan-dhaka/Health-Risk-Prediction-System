# app/train_model.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

# ---------- 1. Load Dataset ----------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

print("First 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())
print("\nMissing values per column:")
print(df.isna().sum())

# ---------- 2. Simple EDA (Data Science) ----------

print("\nBasic statistics:")
print(df.describe())

# Correlation matrix
corr = df.corr(numeric_only=True)
print("\nCorrelation with Outcome:")
print(corr["Outcome"].sort_values(ascending=False))

plt.figure()
corr["Outcome"].sort_values(ascending=False).plot(kind="bar")
plt.title("Feature correlation with Outcome")
plt.xlabel("Feature")
plt.ylabel("Correlation with Outcome")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "correlation_with_outcome.png"))
plt.close()

# Distribution of target
plt.figure()
df["Outcome"].value_counts().plot(kind="bar")
plt.title("Outcome distribution (0 = No diabetes, 1 = Diabetes)")
plt.xlabel("Outcome")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "outcome_distribution.png"))
plt.close()

print("\nSaved EDA plots to models/ folder.")

# ---------- 3. Prepare Data for ML ----------

# Features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------- 4. Train Logistic Regression Model ----------

log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# ---------- 5. Evaluation ----------

y_pred = log_reg.predict(X_test_scaled)
y_proba = log_reg.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\nModel Performance:")
print(f"Accuracy:  {acc:.3f}")
print(f"Precision: {prec:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"F1-score:  {f1:.3f}")
print(f"ROC AUC:   {auc:.3f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"LogReg (AUC = {auc:.2f})")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODELS_DIR, "roc_curve.png"))
plt.close()

print("Saved ROC curve to models/ folder.")

# ---------- 6. Save Model and Scaler ----------

scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
model_path = os.path.join(MODELS_DIR, "logreg_model.pkl")

joblib.dump(scaler, scaler_path)
joblib.dump(log_reg, model_path)

print(f"\nSaved scaler to {scaler_path}")
print(f"Saved model to {model_path}")
