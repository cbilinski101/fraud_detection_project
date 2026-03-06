
# 🛡️ Fraud Detection Model Exploration

This project explores how machine learning models can help detect **credit card fraud**.

The goal of this project is not just to build one model, but to **experiment with multiple models and see how changes affect performance**.

The project includes:

• Model experiments (Models 1–10)  
• Precision–Recall comparison charts  
• A simple interactive dashboard  
• Cloud deployment  

---

# 🚀 Live Dashboard

After deployment your app will be available at:

https://your-render-app.onrender.com

---

# 🧠 Project Idea (Simple Explanation)

Imagine a bank trying to answer this question:

"Does this credit card transaction look normal, or suspicious?"

A machine learning model looks at things like:

• transaction amount  
• time of day  
• customer information  

Then it estimates the **probability that the transaction might be fraud**.

---

# 📊 Model Development

Instead of building only one model, this project compares **10 different models**.

Each model tests different ideas such as:

• feature changes  
• hyperparameter tuning  
• threshold adjustments  

The goal is to understand how model decisions change.

The charts in `model_comparisons/` show how each model performs.

---

# 📉 Why Precision‑Recall Curves?

Fraud detection datasets are **highly imbalanced**.

Example:

Legitimate transactions: 99%  
Fraudulent transactions: 1%

Because of this, **accuracy alone is misleading**.

Precision–Recall curves help us understand:

• how many frauds we catch  
• how many false alarms we create

---

# 🖥️ Dashboard

The Streamlit dashboard allows users to:

1. Enter transaction details
2. Run the trained model
3. See the fraud risk probability

This demonstrates how a machine learning model could be used inside a real system.

---

# 🗂 Project Structure

fraud_detection_project

fraud_app.py  
optimized_model.joblib  
requirements.txt  
render.yaml  
model_comparisons/  
README.md  

---

# ⚙️ Run Locally

pip install -r requirements.txt

streamlit run fraud_app.py

---

# ☁️ Deploy on Render

1. Push the repo to GitHub
2. Go to https://render.com
3. Create **New Web Service**
4. Render will detect `render.yaml`
5. Deploy

---

# 👩‍💻 Author

Christine Bilinski  
GitHub: https://github.com/cbilinski101
