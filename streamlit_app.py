import os
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# from app.db import init_db, insert_patient_record, get_all_patients
from app.db import init_db, insert_patient_record, get_all_patients, get_patients_by_name

from app.utils import (
    load_model_and_scaler,
    probability_to_risk_level,
    make_feature_array,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "diabetes.csv")


# ------------- Setup -------------
# Initialize DB (safe to call multiple times)
init_db()

# Load model & scaler once
scaler, model = load_model_and_scaler()

# Original feature names (same as training)
FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]


# ------------- Sidebar Navigation -------------
st.sidebar.title("Health Risk Prediction")
page = st.sidebar.radio("Go to", ["Predict Risk", "Dashboard"])


# ------------- Page 1: Prediction -------------
if page == "Predict Risk":
    st.title("Diabetes Risk Prediction")

    st.write(
        "Enter patient details below to estimate the probability of having diabetes."
    )
    name = st.text_input("Patient Name")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0.0, max_value=20.0, value=1.0, step=1.0)
        glucose = st.number_input("Glucose", min_value=0.0, max_value=300.0, value=120.0, step=1.0)
        blood_pressure = st.number_input("Blood Pressure", min_value=0.0, max_value=200.0, value=70.0, step=1.0)
        skin_thickness = st.number_input("Skin Thickness", min_value=0.0, max_value=100.0, value=20.0, step=1.0)

    with col2:
        insulin = st.number_input("Insulin", min_value=0.0, max_value=900.0, value=80.0, step=1.0)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0, step=0.1)
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, step=0.01)
        age = st.number_input("Age", min_value=1.0, max_value=120.0, value=30.0, step=1.0)

    if st.button("Predict Risk"):
        if not name.strip():
            st.warning("Please enter the patient's name before predicting.")
        else:
        # Create features array
         X = make_feature_array(
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
        )

        # Scale and predict
        X_scaled = scaler.transform(X)
        prob = float(model.predict_proba(X_scaled)[0, 1])
        risk_level = probability_to_risk_level(prob)

        st.subheader("Prediction Result")
        st.write(f"**Patient:** {name}")
        st.write(f"**Probability of Diabetes:** {prob:.2f}")
        st.write(f"**Risk Level:** :red[{risk_level}] " if risk_level == "High" else f"**Risk Level:** {risk_level}")

        # Save to database
        insert_patient_record(
            name,
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
            probability=prob,
            risk_level=risk_level,
            outcome=None,  # we don't know ground truth at prediction time
        )

        st.success("Record saved to database.")


# ------------- Page 2: Dashboard -------------
elif page == "Dashboard":
    st.title("Health Risk Dashboard")

    st.write("This page shows analytics from the original dataset and stored predictions.")

    # --- Load original dataset for EDA ---
    # if os.path.exists(DATA_PATH):
    #     df = pd.read_csv(DATA_PATH)
    #     st.subheader("Original Dataset Preview")
    #     st.dataframe(df.head())

    #     st.write("Basic statistics:")
    #     st.write(df.describe())

    # else:
    #     st.warning("Dataset not found at data/diabetes.csv")

    # --- Load patient records from DB ---
    rows = get_all_patients()
    if len(rows) == 0:
        st.info("No patient records in the database yet. Go to 'Predict Risk' and add some.")
    else:
        cols = [
            "id",
            "Name",
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DPF",
            "Age",
            "Probability",
            "RiskLevel",
            "Outcome",
        ]
        patients_df = pd.DataFrame(rows, columns=cols)

        st.subheader("Stored Patient Records")
        st.subheader("Search Patient by Name")
        search_name = st.text_input("Enter patient name to search", key="search_name")

        if st.button("Search", key="search_button"):
            if not search_name.strip():
                st.warning("Please enter a name to search.")
            else:
                results = get_patients_by_name(search_name.strip())
                if len(results) == 0:
                    st.info(f"No records found for '{search_name}'.")
                else:
                    search_df = pd.DataFrame(results, columns=cols)
                    st.write(f"Records for **{search_name}**:")
                    st.dataframe(search_df)

        st.dataframe(patients_df)

        # --- Risk level distribution ---
        st.subheader("Risk Level Distribution (from predictions)")

        counts = patients_df["RiskLevel"].value_counts()

        fig1, ax1 = plt.subplots()
        counts.plot(kind="bar", ax=ax1)
        ax1.set_xlabel("Risk Level")
        ax1.set_ylabel("Count")
        ax1.set_title("Risk Level Distribution")
        st.pyplot(fig1)

        # --- Feature importance (Logistic Regression coefficients) ---
        st.subheader("Feature Importance (Model Coefficients)")

        # Get coefficients
        coefs = model.coef_[0]
        importance = pd.Series(coefs, index=FEATURE_NAMES).sort_values()

        fig2, ax2 = plt.subplots()
        importance.plot(kind="barh", ax=ax2)
        ax2.set_xlabel("Coefficient")
        ax2.set_title("Feature Importance (higher = more influence)")
        st.pyplot(fig2)
