Manufacturing Equipment Output Prediction 

Overview
This project predicts the hourly output of injection molding machines using Linear Regression.
It helps optimize production efficiency, plan schedules, and detect underperforming machines.

Dataset
Records: 1000 hourly machine logs
Target: Parts produced per hour
Features: Temperature, Pressure, Cycle Time, Cooling Time, Material Viscosity, Machine Age, Operator Experience, etc.

Tech Stack
Python, Pandas, NumPy
Scikit-Learn (Linear Regression)
Streamlit (Web App)
Pickle (Model storage)

Project Structure
├── app.py                 # Streamlit app
├── linear_reg.ipynb       # Model training notebook
├── linear_pipeline.pkl    # Saved ML pipeline
├── manufacturing_dataset.csv  # Dataset
├── requirements.txt       # Dependencies
├── Dockerfile             # For containerization
└── README.md              # Project documentation

Run the App
git clone 
cd manufacturing-output-prediction
pip install -r requirements.txt
streamlit run app.py
