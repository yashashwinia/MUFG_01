Heart Disease Detection
Project Overview
This capstone project predicts the presence of heart disease using classification models. It applies machine learning techniques on clinical diagnostic data to support early detection and improve patient outcomes.

Problem Statement
Can we accurately predict whether a patient has heart disease based on their diagnostic test results and clinical measurements?

Dataset
Target Variable: heart_disease (0 = No, 1 = Yes)
Features: Age, Gender, Chest Pain Type, Resting BP, Cholesterol, Fasting Blood Sugar, ECG, Max Heart Rate, Exercise Angina, ST Depression, ST Slope, Major Vessels, Thalassemia

Tech Stack
Python 
Pandas, NumPy → Data handling
Scikit-Learn → Models & Evaluation
Streamlit → Web app deployment
Pickle → Model saving/loading

🖥️ Project Structure
├── data
│   └── heart_disease_dataset.csv      # Dataset
├── models
│   └── heart_disease_model.pkl        # Trained ML model
├── notebooks
│   └── heart_disease_notebook.ipynb   # EDA & model training
├── app.py                             # Streamlit app
├── train_model.py                     # Training script
├── requirements.txt                   # Dependencies
├── Dockerfile                         # For containerization
└── README.md                          # Project documentation

🚀 How to Run
git clone 
cd heart-disease-detection
pip install -r requirements.txt
streamlit run app.py
