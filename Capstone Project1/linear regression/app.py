import streamlit as st
import pandas as pd
import pickle

# Load trained pipeline
pipe = pickle.load(open("linear_pipeline.pkl", "rb"))

# Feature lists
cat_feats = ['Shift','Machine_Type','Material_Grade','Day_of_Week']
num_feats = ['Injection_Temperature','Injection_Pressure','Cycle_Time','Cooling_Time',
             'Material_Viscosity','Ambient_Temperature','Machine_Age','Operator_Experience',
             'Maintenance_Hours','Temperature_Pressure_Ratio','Total_Cycle_Time',
             'Efficiency_Score','Machine_Utilization']

st.title("Manufacturing Equipment Output Prediction")

# Collect user input
data = {}
# Categorical inputs
data['Shift'] = st.selectbox('Shift', ['Day','Evening','Night'])
data['Machine_Type'] = st.selectbox('Machine_Type', ['Type_A','Type_B'])
data['Material_Grade'] = st.selectbox('Material_Grade', ['Economy','Standard','Premium'])
data['Day_of_Week'] = st.selectbox('Day_of_Week', ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])

# Numeric inputs
for n in num_feats:
    data[n] = st.number_input(n, value=50.0)

# Predict button
if st.button("Predict"):
    df_input = pd.DataFrame([data])
    pred = pipe.predict(df_input)[0]
    st.success(f"Predicted Parts/Hour: {pred:.2f}")
