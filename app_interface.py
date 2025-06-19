import streamlit as st
import joblib
import numpy as np

# Load trained model
model = joblib.load('models/fraud_model.pkl')

# App title
st.title("💳 Fraud Transaction Detection System")

st.markdown("""
Welcome! Enter the transaction details below, and we'll predict if it's **fraudulent** or **legit**.
""")

# Input fields
amount = st.number_input("Enter Transaction Amount (₹)", min_value=0.0, step=1.0)
hour = st.slider("Select Transaction Hour (0-23)", 0, 23, 12)
day = st.slider("Select Transaction Day (1-31)", 1, 31, 15)

# Calculate HIGH_AMOUNT feature
high_amount = 1 if amount > 220 else 0

# Prediction button
if st.button("🚨 Detect Fraud"):
    new_tx = np.array([[amount, hour, day, high_amount]])
    prediction = model.predict(new_tx)

    if prediction[0] == 1:
        st.error("⚠️ This transaction is **FRAUDULENT**!")
    else:
        st.success("✅ This transaction is **Legitimate**.")

# Footer
st.markdown("""
---
Made with ❤️ by Pankaj Bhattacharjee
""")
