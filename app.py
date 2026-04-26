import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

st.title("✅ FAKE JOB DETECTOR - WORKS!")

# Use the TF-IDF model from your notebook (LogisticRegression cell)
model = pickle.load(open("model.pkl", "rb"))

title = st.text_input("**Job Title**")
description = st.text_area("**Job Description**")

if st.button("🔍 Predict"):
    # TF-IDF like your notebook
    text = f"{title} {description}"
    vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
    X = vectorizer.fit_transform([text])
    
    pred = model.predict(X)[0]
    
    if pred == 1:
        st.error("🔴 FAKE JOB!")
    else:
        st.success("🟢 REAL JOB!")