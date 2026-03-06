
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

st.title("🛡️ Fraud Detection Demo Dashboard")

st.write(
"""
This dashboard shows how a machine learning model can help detect
suspicious credit card transactions.

You enter information about a transaction and the model estimates
whether it looks **normal** or **possibly fraudulent**.
"""
)

MODEL_PATH = "optimized_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.header("Transaction Input")

col1, col2 = st.columns(2)

amount = col1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
age = col2.slider("Customer Age", 18, 100, 35)

city_pop = col1.number_input("City Population", min_value=0, value=50000)
hour = col2.slider("Hour of Day", 0, 23, 12)

data = pd.DataFrame([{
    "amt": amount,
    "age": age,
    "city_pop": city_pop,
    "hour_of_day": hour,
    "day_of_week": 2
}])

if st.button("Predict Fraud Risk"):

    try:
        pred = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

        label = "🟢 Likely Normal"
        if pred == 1:
            label = "🔴 Possible Fraud"

        st.subheader(label)
        st.progress(float(prob))
        st.write(f"Fraud probability: **{prob:.2%}**")

    except Exception as e:
        st.error("Model input format does not match training features.")
        st.write(e)

st.markdown("---")
st.header("Model Development Journey")

st.write(
"""
During development, multiple models were tested to see how changes
affected performance.

The charts below show how **precision** and **recall** changed as
different models were tried.
"""
)
