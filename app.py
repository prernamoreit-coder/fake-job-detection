import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("🔥 FAKE JOB DETECTOR")

# Load TF-IDF files
model = pickle.load(open("model_tfidf.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

title = st.text_input("Job Title")
description = st.text_area("Job Description")

if st.button("Predict") and title:
    text = f"{title} {description}"
    X = vectorizer.transform([text])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0]
    
    if pred == 1:
        st.error("🚨 FAKE JOB")
    else:
        st.success("✅ REAL JOB")
    st.metric("Confidence", f"{max(prob)*100:.0f}%")