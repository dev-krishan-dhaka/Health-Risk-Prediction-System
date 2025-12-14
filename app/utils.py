# app/utils.py

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_model_and_scaler():
    """Load trained scaler and logistic regression model."""
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    model_path = os.path.join(MODELS_DIR, "logreg_model.pkl")

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)

    return scaler, model


def probability_to_risk_level(probability: float) -> str:
    """
    Map a probability (0–1) to a risk level string.
    You can tweak the thresholds later if you want.
    """
    if probability < 0.33:
        return "Low"
    elif probability < 0.66:
        return "Medium"
    else:
        return "High"


def make_feature_array(
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age,
):
    """
    Create feature array in the same order as training:
    [Pregnancies, Glucose, BloodPressure, SkinThickness,
     Insulin, BMI, DiabetesPedigreeFunction, Age]
    """
    return np.array(
        [
            [
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                dpf,
                age,
            ]
        ]
    )
