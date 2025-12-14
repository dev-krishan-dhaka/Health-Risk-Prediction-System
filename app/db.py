# app/db.py

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "health_risk.db")


def get_connection():
    """Create a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    """Create the patients table if it doesn't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS patients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            pregnancies REAL,
            glucose REAL,
            blood_pressure REAL,
            skin_thickness REAL,
            insulin REAL,
            bmi REAL,
            dpf REAL,
            age REAL,
            probability REAL,
            risk_level TEXT,
            outcome INTEGER
        )
        """
    )

    conn.commit()
    conn.close()


def insert_patient_record(
    name,
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    dpf,
    age,
    probability,
    risk_level,
    outcome=None,
):
    """Insert one patient record into the DB."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO patients (
            name, pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, probability, risk_level, outcome
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            name,
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age,
            probability,
            risk_level,
            outcome,
        ),
    )

    conn.commit()
    conn.close()


def get_all_patients():
    """Return all rows as a list of tuples."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id, name, pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, probability, risk_level, outcome
        FROM patients
        ORDER BY id DESC
        """
    )

    rows = cursor.fetchall()
    conn.close()
    return rows


def get_patients_by_name(name: str):
    """Return all rows for a given patient name."""
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT
            id, name, pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age, probability, risk_level, outcome
        FROM patients
        WHERE name = ?
        ORDER BY id DESC
        """,
        (name,),
    )

    rows = cursor.fetchall()
    conn.close()
    return rows
