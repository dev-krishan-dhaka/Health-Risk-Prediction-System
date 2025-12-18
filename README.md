# 🩺 Health Risk Prediction System (Diabetes)

A Data Science & Machine Learning web application that predicts a patient’s diabetes risk level (Low / Medium / High) based on health parameters.
The system uses Logistic Regression, stores patient records in an SQL (SQLite) database, and provides an interactive Streamlit dashboard for predictions and analytics.


## 🔹 Data Science

Exploratory Data Analysis (EDA)

Feature correlation analysis

Feature importance using model coefficients

Data visualization with Matplotlib

## 🔹 Database (SQL)
SQLite database integration
Stores:
     Patient name
     Health parameters
     Predicted probability
     Risk level
Search patient records by name

## 🔹 Web Application (Streamlit)
User-friendly web interface
Patient data input form
Real-time prediction
Dashboard with:
    Stored patient records
    Risk distribution chart
    Feature importance visualization

## 🧠 Tech Stack
Programming Language: Python
Libraries:
    NumPy
    Pandas
    Matplotlib
    Scikit-learn
    Streamlit
    SQLAlchemy / SQLite
    IDE: VS Code
    Database: SQLite
    Deployment: Streamlit

## 📁 Project Structure
``health_risk_prediction/
│
├── app/
│   ├── __init__.py
│   ├── train_model.py      # Model training & evaluation
│   ├── db.py               # SQLite database operations
│   ├── utils.py            # Model loading & helper functions
│
├── data/
│   └── diabetes.csv        # Dataset
│
├── models/
│   ├── scaler.pkl          # StandardScaler
│   └── logreg_model.pkl    # Trained Logistic Regression model
│
├── streamlit_app.py        # Streamlit web application
├── requirements.txt
├── README.md
└── health_risk.db          # SQLite database (auto-created)``

## 📊 Dataset
Dataset: Diabetes dataset (e.g., Pima Indians Diabetes Dataset)
Features Used:
    Pregnancies
    Glucose
    Blood Pressure
    Skin Thickness
    Insulin
    BMI
    Diabetes Pedigree Function
    Age
Target Variable:
    Outcome (0 = No Diabetes, 1 = Diabetes)

## ⚙️ Installation & Setup
1️⃣ Clone the Repository
`git clone https://github.com/your-username/health-risk-prediction.git
cd health-risk-prediction`

2️⃣ Create Virtual Environment (Optional but Recommended)
`python -m venv venv
venv\Scripts\activate`   # Windows

3️⃣ Install Dependencies
`pip install -r requirements.txt`

## 🏋️ Model Training
Train the ML model and generate evaluation metrics:
`python -m app.train_model`
This will:
    Train Logistic Regression model
    Evaluate performance
    Save trained model & scaler to models/

## 🌐 Run the Streamlit App
`streamlit run streamlit_app.py`

The app will open automatically in your browser.


## 🔮 Future Enhancements
Add user authentication
Deploy on cloud (Streamlit Cloud / AWS)
Support multiple diseases
Improve model using advanced algorithms
Add PDF report generation for patients

## 👨‍💻 Author

Dev Krishan
Final Year B.Tech – Computer Science
GitHub: https://github.com/dev-krishan-dhaka
