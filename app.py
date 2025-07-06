import streamlit as st
import subprocess
import sys

# Force reinstall joblib every time app runs (temporary fix)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "joblib"])

import joblib
st.markdown("""
    <style>
    .stApp {
        background-color: #f9f9f9;
        font-family: 'Segoe UI', sans-serif;
    }
    </style>
""", unsafe_allow_html=True)

# Load saved model and tools
model = joblib.load("spam_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")
le = joblib.load("label_encoder.pkl")

# Title
st.title("📨 Spam Detector App")
st.markdown("🔍 Type a message and find out if it's **Spam** or **Ham (Not Spam)**")

# Input box
msg = st.text_area("type your custom message below:")

# Button to predict
if st.button("Check Spam"):
    vect_msg = vectorizer.transform([msg])
    prediction = model.predict(vect_msg)
    result = le.inverse_transform(prediction)[0]
    proba = model.predict_proba(vect_msg)
    spam_score = proba[0][1] * 100
    st.write(f"🧠 Spam Confidence: {spam_score:.2f}%")

    if result == "spam":
        st.error("⚠️ This message is SPAM!")
    else:
        st.success("✅ This message is HAM (Not Spam).")
        
st.markdown("---")

# 🖋️ Footer
st.caption("Built with ❤️ by Ishani Chakravarty")