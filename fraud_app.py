
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Fraud Detection Demo", layout="wide")

MODEL_PATH = "optimized_model.joblib"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🛡️ Fraud Detection Demo Dashboard")

st.write(
"""
This dashboard demonstrates how a machine learning model can help detect
suspicious credit card transactions.

Enter transaction information and the model will estimate whether the
transaction appears **normal** or **possibly fraudulent**.
"""
)

# ---- Transaction Inputs ----

st.header("Transaction Input")

col1, col2 = st.columns(2)

amount = col1.number_input("Transaction Amount ($)", min_value=0.0, value=100.0)
city_pop = col2.number_input("City Population", min_value=0, value=50000)

age = col1.slider("Customer Age", 18, 100, 35)
hour = col2.slider("Hour of Day", 0, 23, 12)

day = st.selectbox("Day of Week", list(range(7)), format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x])

category_options = {
    "Entertainment": "category_entertainment",
    "Food & Dining": "category_food_dining",
    "Gas & Transport": "category_gas_transport",
    "Grocery (Net)": "category_grocery_net",
    "Grocery (POS)": "category_grocery_pos",
    "Health & Fitness": "category_health_fitness",
    "Home": "category_home",
    "Kids & Pets": "category_kids_pets",
    "Misc (Net)": "category_misc_net",
    "Misc (POS)": "category_misc_pos",
    "Personal Care": "category_personal_care",
    "Shopping (Net)": "category_shopping_net",
    "Shopping (POS)": "category_shopping_pos",
    "Travel": "category_travel"
}

selected_cat = st.selectbox("Transaction Category", list(category_options.keys()))

# ---- Expected Model Columns ----

input_cols = [
    "amt","city_pop","age","day_of_week","hour_of_day",
    "category_entertainment","category_food_dining","category_gas_transport",
    "category_grocery_net","category_grocery_pos","category_health_fitness",
    "category_home","category_kids_pets","category_misc_net","category_misc_pos",
    "category_personal_care","category_shopping_net","category_shopping_pos","category_travel"
]

# ---- Build Input Row ----

input_data = {col: 0 for col in input_cols}

input_data["amt"] = amount
input_data["city_pop"] = city_pop
input_data["age"] = age
input_data["day_of_week"] = day
input_data["hour_of_day"] = hour

# set chosen category
input_data[category_options[selected_cat]] = 1

df = pd.DataFrame([input_data])

# ---- Prediction ----

if st.button("Predict Fraud Risk"):

    try:
        pred = model.predict(df)[0]
        proba = model.predict_proba(df)[0][1]

        label = "🟢 Likely Normal"
        if pred == 1:
            label = "🔴 Possible Fraud"

        st.subheader(label)

        st.progress(float(proba))
        st.write(f"Fraud probability: **{proba:.2%}**")

    except Exception as e:
        st.error("Prediction failed. Model feature mismatch.")
        st.write(e)

st.markdown("---")
st.write("Demo ML deployment for fraud detection.")
