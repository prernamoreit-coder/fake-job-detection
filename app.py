import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🔥 FAKE JOB DETECTOR")

model = pickle.load(open("model_simple.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

title = st.text_input("Job Title")
description = st.text_area("Job Description")

if st.button("Predict"):
    text = f"{title} {description}"
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    
    col1, col2 = st.columns(2)
    with col1:
        if pred == 1:
            st.error("🚨 FAKE JOB")
        else:
            st.success("✅ REAL JOB")
    with col2:
        st.metric("Confidence", f"{max(prob)*100:.0f}%")